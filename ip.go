package turboquant

import (
	"crypto/rand"
	"encoding/binary"
	"math"
	mrand "math/rand"
	"sync"
)

// IPQuantizer provides unbiased inner-product-preserving quantization.
// Uses TurboQuant_mse at bit-width (b-1) + 1-bit QJL on the residual.
// Safe for concurrent use after construction.
type IPQuantizer struct {
	dim      int
	bitWidth int
	seed     int64
	mse      *Quantizer
	proj     []float32 // d×d Gaussian matrix for QJL
	proj8    []float32 // 8-row blocked projection for grouped QJL kernels
	pool     sync.Pool
}

// IPQuantized holds the two-stage quantization result.
type IPQuantized struct {
	MSE     []byte  // packed MSE indices (b-1 bits per coordinate)
	Signs   []byte  // packed QJL sign bits (1 bit per coordinate)
	ResNorm float32 // L2 norm of residual
}

// PreparedQuery precomputes S^T * y for amortized IP queries.
type PreparedQuery struct {
	signLUT []float32 // per-sign-byte QJL contribution table
	mseLUT  []float32 // per-packed-byte MSE contribution table when supported
	rotY    []float32 // y rotated into the MSE domain once

	mseBitWidth uint8
}

type ipScratch struct {
	residual []float32
}

// IPQuantizedSizes returns the byte lengths required for the MSE and sign
// payloads of an IP-quantized vector.
func IPQuantizedSizes(dim, bitWidth int) (mseBytes, signBytes int) {
	panicOnInvalid("turboquant.IPQuantizedSizes", validateDim(dim))
	panicOnInvalid("turboquant.IPQuantizedSizes", validateIPBitWidth(bitWidth))
	return PackedSize(dim, bitWidth-1), (dim + 7) / 8
}

// AllocIPQuantized allocates a reusable storage buffer for a single IP-quantized
// vector. The returned struct owns a single backing allocation shared by MSE and
// Signs.
func AllocIPQuantized(dim, bitWidth int) IPQuantized {
	mseBytes, signBytes := IPQuantizedSizes(dim, bitWidth)
	storage := make([]byte, mseBytes+signBytes)
	return IPQuantized{
		MSE:   storage[:mseBytes],
		Signs: storage[mseBytes:],
	}
}

// NewIP creates an inner-product-optimal quantizer with random seed.
// The default constructor uses the fast structured Walsh-Hadamard rotation in
// the MSE stage.
func NewIP(dim, bitWidth int) *IPQuantizer {
	var seedBytes [8]byte
	rand.Read(seedBytes[:])
	seed := int64(binary.LittleEndian.Uint64(seedBytes[:]))
	return NewIPHadamardWithSeed(dim, bitWidth, seed)
}

// NewIPHadamard creates an inner-product-optimal quantizer whose MSE stage uses
// a structured Walsh-Hadamard rotation.
func NewIPHadamard(dim, bitWidth int) *IPQuantizer {
	var seedBytes [8]byte
	rand.Read(seedBytes[:])
	seed := int64(binary.LittleEndian.Uint64(seedBytes[:]))
	return NewIPHadamardWithSeed(dim, bitWidth, seed)
}

// NewIPDense creates an inner-product-optimal quantizer with the legacy dense
// QR rotation in the MSE stage.
func NewIPDense(dim, bitWidth int) *IPQuantizer {
	var seedBytes [8]byte
	rand.Read(seedBytes[:])
	seed := int64(binary.LittleEndian.Uint64(seedBytes[:]))
	return NewIPDenseWithSeed(dim, bitWidth, seed)
}

// NewIPWithSeed creates a deterministic inner-product-optimal quantizer using
// the default fast structured Walsh-Hadamard rotation in the MSE stage.
func NewIPWithSeed(dim, bitWidth int, seed int64) *IPQuantizer {
	return NewIPHadamardWithSeed(dim, bitWidth, seed)
}

// NewIPDenseWithSeed creates a deterministic inner-product-optimal quantizer
// with the legacy dense QR rotation in the MSE stage.
func NewIPDenseWithSeed(dim, bitWidth int, seed int64) *IPQuantizer {
	if dim < 2 {
		panic("turboquant: dim must be >= 2")
	}
	if bitWidth < 2 {
		panic("turboquant: IP quantizer bitWidth must be >= 2")
	}
	mseSeed := seed
	return newIPQuantizer(dim, bitWidth, seed, NewDenseWithSeed(dim, bitWidth-1, mseSeed))
}

// NewIPHadamardWithSeed creates a deterministic inner-product-optimal quantizer
// whose MSE stage uses a structured Walsh-Hadamard rotation.
func NewIPHadamardWithSeed(dim, bitWidth int, seed int64) *IPQuantizer {
	if dim < 2 {
		panic("turboquant: dim must be >= 2")
	}
	if bitWidth < 2 {
		panic("turboquant: IP quantizer bitWidth must be >= 2")
	}
	mseSeed := seed
	return newIPQuantizer(dim, bitWidth, seed, NewHadamardWithSeed(dim, bitWidth-1, mseSeed))
}

func newIPQuantizer(dim, bitWidth int, seed int64, mseQ *Quantizer) *IPQuantizer {
	panicOnInvalid("turboquant.NewIP", validateDim(dim))
	panicOnInvalid("turboquant.NewIP", validateIPBitWidth(bitWidth))
	// CRITICAL: MSE rotation and QJL projection must be independent.
	qjlSeed := seed ^ 0x5DEECE66D
	qjlRng := mrand.New(mrand.NewSource(qjlSeed))
	proj := generateGaussianMatrix(dim, qjlRng)
	return newIPQuantizerWithProjection(dim, bitWidth, seed, mseQ, proj)
}

func newIPQuantizerWithProjection(dim, bitWidth int, seed int64, mseQ *Quantizer, proj []float32) *IPQuantizer {
	q := &IPQuantizer{
		dim:      dim,
		bitWidth: bitWidth,
		seed:     seed,
		mse:      mseQ,
		proj:     proj,
		proj8:    blockProjectionRows8(proj, dim),
	}
	q.pool.New = func() any {
		return &ipScratch{
			residual: make([]float32, dim),
		}
	}
	return q
}

func (q *IPQuantizer) Dim() int      { return q.dim }
func (q *IPQuantizer) BitWidth() int { return q.bitWidth }
func (q *IPQuantizer) Seed() int64   { return q.seed }
func (q *IPQuantizer) RotationKind() string {
	return q.mse.RotationKind()
}

// Quantize returns the two-stage quantization result.
func (q *IPQuantizer) Quantize(vec []float32) IPQuantized {
	qx := AllocIPQuantized(q.dim, q.bitWidth)
	q.QuantizeTo(&qx, vec)
	return qx
}

// QuantizeTo compresses vec into caller-owned storage. dst must have been
// allocated for this quantizer's dimension and bit width.
func (q *IPQuantizer) QuantizeTo(dst *IPQuantized, vec []float32) {
	if dst == nil {
		panic("(*IPQuantizer).QuantizeTo: turboquant: nil destination")
	}
	panicOnInvalid("(*IPQuantizer).QuantizeTo", ValidateVector(q.dim, vec))
	panicOnInvalid("(*IPQuantizer).QuantizeTo", ValidateIPQuantized(q.dim, q.bitWidth, *dst))

	mseBuf := q.mse.pool.Get().(*scratchBuf)
	buf := q.pool.Get().(*ipScratch)
	defer q.mse.pool.Put(mseBuf)
	defer q.pool.Put(buf)

	mseNorm := q.mse.quantizeToBuf(dst.MSE, vec, mseBuf)
	q.mse.dequantizeToBuf(buf.residual, dst.MSE, mseBuf)

	scale := float32(1.0)
	if mseNorm > 1e-12 {
		scale = 1.0 / mseNorm
	}
	for i := range vec {
		buf.residual[i] = vec[i]*scale - buf.residual[i]
	}

	dst.ResNorm = qjlProjectBlocked(dst.Signs, buf.residual, q.proj, q.proj8, q.dim)
}

// Dequantize reconstructs an approximate vector (primarily for debugging).
func (q *IPQuantizer) Dequantize(qx IPQuantized) []float32 {
	panicOnInvalid("(*IPQuantizer).Dequantize", ValidateIPQuantized(q.dim, q.bitWidth, qx))
	mseRecon := q.mse.Dequantize(qx.MSE)
	// Add QJL reconstruction: mse_recon + sqrt(pi/2)/d * resNorm * S^T * sign_vector
	scale := float32(math.Sqrt(math.Pi/2.0)) / float32(q.dim) * qx.ResNorm
	for i := 0; i < q.dim; i++ {
		var sum float32
		for j := 0; j < q.dim; j++ {
			signBit := qx.Signs[j/8] & (1 << uint(j%8))
			s := float32(-1.0)
			if signBit != 0 {
				s = 1.0
			}
			sum += q.proj[j*q.dim+i] * s // S^T[i][j] = S[j][i]
		}
		mseRecon[i] += scale * sum
	}
	return mseRecon
}

// InnerProduct estimates <x, y> from quantized x and raw y.
// Asymmetric: does NOT reconstruct the full vector.
// Unbiased: E[InnerProduct(Quantize(x), y)] = <x, y>
func (q *IPQuantizer) InnerProduct(qx IPQuantized, y []float32) float32 {
	panicOnInvalid("(*IPQuantizer).InnerProduct", ValidateIPQuantized(q.dim, q.bitWidth, qx))
	panicOnInvalid("(*IPQuantizer).InnerProduct", ValidateVector(q.dim, y))
	mseDot := q.mse.InnerProduct(qx.MSE, 1, y)
	// QJL part
	qjlDot := qjlInnerProductBlocked(qx.Signs, qx.ResNorm, y, q.proj, q.proj8, q.dim)
	return mseDot + qjlDot
}

// AllocPreparedQuery allocates reusable storage for a prepared query of the
// given dimension.
func AllocPreparedQuery(dim int) PreparedQuery {
	panicOnInvalid("turboquant.AllocPreparedQuery", validateDim(dim))
	return PreparedQuery{
		signLUT: make([]float32, preparedQuerySignLUTLen(dim)),
		rotY:    make([]float32, dim),
	}
}

func preparedQuerySignLUTLen(dim int) int {
	return ((dim + 7) / 8) * 256
}

func buildSignedNibbleTable(dst *[16]float32, weights []float32) {
	size := 1 << len(weights)
	for value := 0; value < size; value++ {
		mask := value
		var sum float32
		for i := 0; i < len(weights); i++ {
			if mask&1 != 0 {
				sum += weights[i]
			} else {
				sum -= weights[i]
			}
			mask >>= 1
		}
		dst[value] = sum
	}
	for offset := size; offset < len(dst); offset += size {
		copy(dst[offset:offset+size], dst[:size])
	}
}

func buildSignedNibbleTable4(dst *[16]float32, w0, w1, w2, w3 float32) {
	var p0 [2]float32
	var p1 [2]float32
	var p2 [2]float32
	var p3 [2]float32
	p0[0], p0[1] = -w0, w0
	p1[0], p1[1] = -w1, w1
	p2[0], p2[1] = -w2, w2
	p3[0], p3[1] = -w3, w3
	for b3 := 0; b3 < 2; b3++ {
		v3 := p3[b3]
		base3 := b3 << 3
		for b2 := 0; b2 < 2; b2++ {
			v23 := v3 + p2[b2]
			base23 := base3 | (b2 << 2)
			for b1 := 0; b1 < 2; b1++ {
				v231 := v23 + p1[b1]
				base231 := base23 | (b1 << 1)
				dst[base231] = v231 + p0[0]
				dst[base231|1] = v231 + p0[1]
			}
		}
	}
}

func combineNibbleTables256(dst []float32, low, high *[16]float32) {
	for hi := 0; hi < 16; hi++ {
		highVal := high[hi]
		row := hi << 4
		for lo := 0; lo < 16; lo++ {
			dst[row|lo] = low[lo] + highVal
		}
	}
}

func fillPreparedSignTable(dst []float32, weights []float32) {
	if len(weights) == 8 {
		var low [16]float32
		var high [16]float32
		buildSignedNibbleTable4(&low, weights[0], weights[1], weights[2], weights[3])
		buildSignedNibbleTable4(&high, weights[4], weights[5], weights[6], weights[7])
		combineNibbleTables256(dst, &low, &high)
		return
	}
	var low [16]float32
	var high [16]float32
	lowCount := len(weights)
	if lowCount > 4 {
		lowCount = 4
	}
	buildSignedNibbleTable(&low, weights[:lowCount])
	if len(weights) > 4 {
		buildSignedNibbleTable(&high, weights[4:])
	} else {
		buildSignedNibbleTable(&high, nil)
	}
	combineNibbleTables256(dst, &low, &high)
}

func fillPreparedSignTable8(dst []float32, dots [8]float32) {
	var low [16]float32
	var high [16]float32
	buildSignedNibbleTable4(&low, dots[0], dots[1], dots[2], dots[3])
	buildSignedNibbleTable4(&high, dots[4], dots[5], dots[6], dots[7])
	combineNibbleTables256(dst, &low, &high)
}

func preparedQueryMSELUTLen(dim, bitWidth int) int {
	switch bitWidth {
	case 1, 2, 4, 8:
		return PackedSize(dim, bitWidth) * 256
	default:
		return 0
	}
}

// AllocPreparedQuery allocates reusable storage sized for this quantizer,
// including the fast prepared MSE lookup table when supported.
func (q *IPQuantizer) AllocPreparedQuery() PreparedQuery {
	pq := AllocPreparedQuery(q.dim)
	pq.mseBitWidth = uint8(q.mse.bitWidth)
	if n := preparedQueryMSELUTLen(q.dim, q.mse.bitWidth); n > 0 {
		pq.mseLUT = make([]float32, n)
	}
	return pq
}

// PrepareQuery precomputes S^T * y for efficient repeated IP queries.
func (q *IPQuantizer) PrepareQuery(y []float32) PreparedQuery {
	pq := q.AllocPreparedQuery()
	q.PrepareQueryTo(&pq, y)
	return pq
}

// PrepareQueryTo precomputes query state into caller-owned storage.
func (q *IPQuantizer) PrepareQueryTo(dst *PreparedQuery, y []float32) {
	if dst == nil {
		panic("(*IPQuantizer).PrepareQueryTo: turboquant: nil destination")
	}
	panicOnInvalid("(*IPQuantizer).PrepareQuery", ValidateVector(q.dim, y))
	panicOnInvalid("(*IPQuantizer).PrepareQueryTo", ValidatePreparedQuery(q.dim, *dst))
	q.PrepareQueryToTrusted(dst, y)
}

// PrepareQueryToTrusted precomputes query state into caller-owned storage
// without revalidating dst. The caller must ensure dst matches this quantizer's
// shape and y has the correct dimension.
func (q *IPQuantizer) PrepareQueryToTrusted(dst *PreparedQuery, y []float32) {
	if dst == nil {
		panic("(*IPQuantizer).PrepareQueryToTrusted: turboquant: nil destination")
	}
	if dst.mseBitWidth != 0 && int(dst.mseBitWidth) != q.mse.bitWidth {
		panic("(*IPQuantizer).PrepareQueryToTrusted: turboquant: prepared query buffer bit width mismatch")
	}
	signBytes := (q.dim + 7) / 8
	fullBytes := q.dim / 8
	for byteIdx := 0; byteIdx < fullBytes; byteIdx++ {
		base := byteIdx * 8
		var dots [8]float32
		if len(q.proj8) != 0 {
			block := q.proj8[byteIdx*q.dim*8 : (byteIdx+1)*q.dim*8]
			dotFloat32Rows8Blocked(&dots, block, y)
		} else {
			dotFloat32Rows8(&dots, q.proj[base*q.dim:(base+8)*q.dim], y)
		}
		table := dst.signLUT[byteIdx*256 : (byteIdx+1)*256]
		fillPreparedSignTable8(table, dots)
	}
	for byteIdx := fullBytes; byteIdx < signBytes; byteIdx++ {
		base := byteIdx * 8
		width := q.dim - base
		if width <= 0 {
			break
		}
		var dots [8]float32
		for bit := 0; bit < width; bit++ {
			row := (base + bit) * q.dim
			dots[bit] = DotFloat32s(q.proj[row:row+q.dim], y)
		}
		table := dst.signLUT[byteIdx*256 : (byteIdx+1)*256]
		fillPreparedSignTable(table, dots[:width])
	}
	buf := q.mse.pool.Get().(*scratchBuf)
	q.mse.rotation.apply(dst.rotY, y, buf.work)
	q.mse.pool.Put(buf)
	if len(dst.mseLUT) != 0 {
		q.mse.buildPreparedDotLUT(dst.mseLUT, dst.rotY)
	}
}

// InnerProductPrepared estimates <x, y> using a precomputed query projection.
func (q *IPQuantizer) InnerProductPrepared(qx IPQuantized, pq PreparedQuery) float32 {
	panicOnInvalid("(*IPQuantizer).InnerProductPrepared", ValidateIPQuantized(q.dim, q.bitWidth, qx))
	panicOnInvalid("(*IPQuantizer).InnerProductPrepared", ValidatePreparedQuery(q.dim, pq))
	return q.innerProductPreparedTrusted(qx, pq)
}

// InnerProductPreparedTrusted estimates <x, y> using a precomputed query
// projection without revalidating qx or pq. The caller must ensure both values
// were produced for this quantizer's dimension and bit width.
func (q *IPQuantizer) InnerProductPreparedTrusted(qx IPQuantized, pq PreparedQuery) float32 {
	return q.innerProductPreparedTrusted(qx, pq)
}

// InnerProductPreparedBatchTo scores one quantized vector against multiple
// prepared queries into caller-owned storage. dst length must match len(pqs).
func (q *IPQuantizer) InnerProductPreparedBatchTo(dst []float32, qx IPQuantized, pqs []PreparedQuery) {
	if len(dst) != len(pqs) {
		panic("(*IPQuantizer).InnerProductPreparedBatchTo: turboquant: destination/query length mismatch")
	}
	panicOnInvalid("(*IPQuantizer).InnerProductPreparedBatchTo", ValidateIPQuantized(q.dim, q.bitWidth, qx))
	for i := range pqs {
		panicOnInvalid("(*IPQuantizer).InnerProductPreparedBatchTo", ValidatePreparedQuery(q.dim, pqs[i]))
	}
	q.InnerProductPreparedBatchToTrusted(dst, qx, pqs)
}

// InnerProductPreparedBatchToTrusted scores one quantized vector against
// multiple prepared queries into caller-owned storage without revalidating qx
// or pqs. The caller must ensure all values match this quantizer's shape.
func (q *IPQuantizer) InnerProductPreparedBatchToTrusted(dst []float32, qx IPQuantized, pqs []PreparedQuery) {
	if len(dst) != len(pqs) {
		panic("(*IPQuantizer).InnerProductPreparedBatchToTrusted: turboquant: destination/query length mismatch")
	}
	for i := range dst {
		dst[i] = 0
	}
	if len(pqs) == 0 {
		return
	}

	wantMSELUT := preparedQueryMSELUTLen(q.dim, q.mse.bitWidth)
	allFastMSE := wantMSELUT != 0
	for i := range pqs {
		if int(pqs[i].mseBitWidth) != q.mse.bitWidth || len(pqs[i].mseLUT) != wantMSELUT {
			allFastMSE = false
			break
		}
	}

	switch len(pqs) {
	case 1:
		q.innerProductPreparedBatch1(dst, qx, pqs[0], allFastMSE)
	case 2:
		q.innerProductPreparedBatch2(dst, qx, pqs[0], pqs[1], allFastMSE)
	case 3:
		q.innerProductPreparedBatch3(dst, qx, pqs[0], pqs[1], pqs[2], allFastMSE)
	case 4:
		q.innerProductPreparedBatch4(dst, qx, pqs[0], pqs[1], pqs[2], pqs[3], allFastMSE)
	default:
		q.innerProductPreparedBatchN(dst, qx, pqs, allFastMSE)
	}
}

func (q *IPQuantizer) innerProductPreparedTrusted(qx IPQuantized, pq PreparedQuery) float32 {
	mseDot := q.mse.innerProductPrepared(qx.MSE, 1, pq.rotY, pq.mseLUT, int(pq.mseBitWidth))
	// QJL part using precomputed projections
	scale := float32(math.Sqrt(math.Pi/2.0)) / float32(q.dim) * qx.ResNorm
	var sum float32
	for i, signByte := range qx.Signs {
		sum += pq.signLUT[i*256+int(signByte)]
	}
	return mseDot + scale*sum
}

func (q *IPQuantizer) innerProductPreparedBatch1(dst []float32, qx IPQuantized, pq0 PreparedQuery, fastMSE bool) {
	if fastMSE {
		var score0 float32
		for i, packedByte := range qx.MSE {
			score0 += pq0.mseLUT[i*256+int(packedByte)]
		}
		scale := float32(math.Sqrt(math.Pi/2.0)) / float32(q.dim) * qx.ResNorm
		for i, signByte := range qx.Signs {
			score0 += scale * pq0.signLUT[i*256+int(signByte)]
		}
		dst[0] = score0
		return
	}
	dst[0] = q.innerProductPreparedTrusted(qx, pq0)
}

func (q *IPQuantizer) innerProductPreparedBatch2(dst []float32, qx IPQuantized, pq0, pq1 PreparedQuery, fastMSE bool) {
	if fastMSE {
		var score0, score1 float32
		for i, packedByte := range qx.MSE {
			offset := i*256 + int(packedByte)
			score0 += pq0.mseLUT[offset]
			score1 += pq1.mseLUT[offset]
		}
		scale := float32(math.Sqrt(math.Pi/2.0)) / float32(q.dim) * qx.ResNorm
		for i, signByte := range qx.Signs {
			offset := i*256 + int(signByte)
			score0 += scale * pq0.signLUT[offset]
			score1 += scale * pq1.signLUT[offset]
		}
		dst[0], dst[1] = score0, score1
		return
	}
	dst[0] = q.innerProductPreparedTrusted(qx, pq0)
	dst[1] = q.innerProductPreparedTrusted(qx, pq1)
}

func (q *IPQuantizer) innerProductPreparedBatch3(dst []float32, qx IPQuantized, pq0, pq1, pq2 PreparedQuery, fastMSE bool) {
	if fastMSE {
		var score0, score1, score2 float32
		for i, packedByte := range qx.MSE {
			offset := i*256 + int(packedByte)
			score0 += pq0.mseLUT[offset]
			score1 += pq1.mseLUT[offset]
			score2 += pq2.mseLUT[offset]
		}
		scale := float32(math.Sqrt(math.Pi/2.0)) / float32(q.dim) * qx.ResNorm
		for i, signByte := range qx.Signs {
			offset := i*256 + int(signByte)
			score0 += scale * pq0.signLUT[offset]
			score1 += scale * pq1.signLUT[offset]
			score2 += scale * pq2.signLUT[offset]
		}
		dst[0], dst[1], dst[2] = score0, score1, score2
		return
	}
	dst[0] = q.innerProductPreparedTrusted(qx, pq0)
	dst[1] = q.innerProductPreparedTrusted(qx, pq1)
	dst[2] = q.innerProductPreparedTrusted(qx, pq2)
}

func (q *IPQuantizer) innerProductPreparedBatch4(dst []float32, qx IPQuantized, pq0, pq1, pq2, pq3 PreparedQuery, fastMSE bool) {
	if fastMSE {
		var score0, score1, score2, score3 float32
		for i, packedByte := range qx.MSE {
			offset := i*256 + int(packedByte)
			score0 += pq0.mseLUT[offset]
			score1 += pq1.mseLUT[offset]
			score2 += pq2.mseLUT[offset]
			score3 += pq3.mseLUT[offset]
		}
		scale := float32(math.Sqrt(math.Pi/2.0)) / float32(q.dim) * qx.ResNorm
		for i, signByte := range qx.Signs {
			offset := i*256 + int(signByte)
			score0 += scale * pq0.signLUT[offset]
			score1 += scale * pq1.signLUT[offset]
			score2 += scale * pq2.signLUT[offset]
			score3 += scale * pq3.signLUT[offset]
		}
		dst[0], dst[1], dst[2], dst[3] = score0, score1, score2, score3
		return
	}
	dst[0] = q.innerProductPreparedTrusted(qx, pq0)
	dst[1] = q.innerProductPreparedTrusted(qx, pq1)
	dst[2] = q.innerProductPreparedTrusted(qx, pq2)
	dst[3] = q.innerProductPreparedTrusted(qx, pq3)
}

func (q *IPQuantizer) innerProductPreparedBatchN(dst []float32, qx IPQuantized, pqs []PreparedQuery, fastMSE bool) {
	if fastMSE {
		for i, packedByte := range qx.MSE {
			offset := i*256 + int(packedByte)
			for j := range pqs {
				dst[j] += pqs[j].mseLUT[offset]
			}
		}
		scale := float32(math.Sqrt(math.Pi/2.0)) / float32(q.dim) * qx.ResNorm
		for i, signByte := range qx.Signs {
			offset := i*256 + int(signByte)
			for j := range pqs {
				dst[j] += scale * pqs[j].signLUT[offset]
			}
		}
		return
	}
	for j := range pqs {
		dst[j] = q.innerProductPreparedTrusted(qx, pqs[j])
	}
}
