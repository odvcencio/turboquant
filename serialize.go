package turboquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

const (
	serializeHeaderSizeV1 = 24
	serializeHeaderSize   = 25
)

const (
	portableMagic      = "TQPB"
	portableVersion    = 1
	portableTypeMSE    = 1
	portableTypeIP     = 2
	portableHeaderSize = 40
)

// MarshalQuantizer serializes a Quantizer to bytes.
// The quantizer is fully reconstructible from dim, bitWidth, seed, and
// rotation family.
func MarshalQuantizer(q *Quantizer) ([]byte, error) {
	if q == nil {
		return nil, fmt.Errorf("turboquant: nil quantizer")
	}
	if err := validateDim(q.dim); err != nil {
		return nil, err
	}
	if err := validateBitWidth(q.bitWidth); err != nil {
		return nil, err
	}
	buf := make([]byte, serializeHeaderSize)
	binary.LittleEndian.PutUint64(buf[0:8], uint64(q.dim))
	binary.LittleEndian.PutUint64(buf[8:16], uint64(q.bitWidth))
	binary.LittleEndian.PutUint64(buf[16:24], uint64(q.seed))
	buf[24] = byte(q.rotation.kind)
	return buf, nil
}

// UnmarshalQuantizer reconstructs a Quantizer from serialized bytes.
func UnmarshalQuantizer(data []byte) (*Quantizer, error) {
	if len(data) < serializeHeaderSizeV1 {
		return nil, fmt.Errorf("turboquant: serialized data too short (%d bytes)", len(data))
	}
	dim := int(binary.LittleEndian.Uint64(data[0:8]))
	bitWidth := int(binary.LittleEndian.Uint64(data[8:16]))
	seed := int64(binary.LittleEndian.Uint64(data[16:24]))
	if err := validateDim(dim); err != nil {
		return nil, err
	}
	if err := validateBitWidth(bitWidth); err != nil {
		return nil, err
	}
	if len(data) < serializeHeaderSize {
		return NewDenseWithSeed(dim, bitWidth, seed), nil
	}
	switch rotationKind(data[24]) {
	case rotationKindDense:
		return NewDenseWithSeed(dim, bitWidth, seed), nil
	case rotationKindHadamard:
		return NewHadamardWithSeed(dim, bitWidth, seed), nil
	default:
		return nil, fmt.Errorf("turboquant: unsupported rotation kind %d", data[24])
	}
}

// MarshalIPQuantizer serializes an IPQuantizer to bytes.
func MarshalIPQuantizer(q *IPQuantizer) ([]byte, error) {
	if q == nil {
		return nil, fmt.Errorf("turboquant: nil IP quantizer")
	}
	if err := validateDim(q.dim); err != nil {
		return nil, err
	}
	if err := validateIPBitWidth(q.bitWidth); err != nil {
		return nil, err
	}
	buf := make([]byte, serializeHeaderSize)
	binary.LittleEndian.PutUint64(buf[0:8], uint64(q.dim))
	binary.LittleEndian.PutUint64(buf[8:16], uint64(q.bitWidth))
	binary.LittleEndian.PutUint64(buf[16:24], uint64(q.seed))
	buf[24] = byte(q.mse.rotation.kind)
	return buf, nil
}

// UnmarshalIPQuantizer reconstructs an IPQuantizer from serialized bytes.
func UnmarshalIPQuantizer(data []byte) (*IPQuantizer, error) {
	if len(data) < serializeHeaderSizeV1 {
		return nil, fmt.Errorf("turboquant: serialized data too short (%d bytes)", len(data))
	}
	dim := int(binary.LittleEndian.Uint64(data[0:8]))
	bitWidth := int(binary.LittleEndian.Uint64(data[8:16]))
	seed := int64(binary.LittleEndian.Uint64(data[16:24]))
	if err := validateDim(dim); err != nil {
		return nil, err
	}
	if err := validateIPBitWidth(bitWidth); err != nil {
		return nil, err
	}
	if len(data) < serializeHeaderSize {
		return NewIPDenseWithSeed(dim, bitWidth, seed), nil
	}
	switch rotationKind(data[24]) {
	case rotationKindDense:
		return NewIPDenseWithSeed(dim, bitWidth, seed), nil
	case rotationKindHadamard:
		return NewIPHadamardWithSeed(dim, bitWidth, seed), nil
	default:
		return nil, fmt.Errorf("turboquant: unsupported rotation kind %d", data[24])
	}
}

// MarshalPortableQuantizer serializes a quantizer with its full rotation matrix
// and codebook for cross-language interop.
func MarshalPortableQuantizer(q *Quantizer) ([]byte, error) {
	if q == nil {
		return nil, fmt.Errorf("turboquant: nil quantizer")
	}
	return marshalPortableState(
		portableTypeMSE,
		q.dim,
		q.bitWidth,
		q.seed,
		q.portable,
		q.rotation.matrix(),
		q.cb.centroids,
		q.cb.boundaries,
		nil,
	)
}

// UnmarshalPortableQuantizer reconstructs a quantizer from portable binary
// state. Portable state always loads as a dense rotation, even if the original
// quantizer used a structured rotation backend.
func UnmarshalPortableQuantizer(data []byte) (*Quantizer, error) {
	state, err := unmarshalPortableState(data)
	if err != nil {
		return nil, err
	}
	if state.kind != portableTypeMSE {
		return nil, fmt.Errorf("turboquant: expected portable MSE quantizer, got type %d", state.kind)
	}
	cb := codebook{
		centroids:  state.centroids,
		boundaries: state.boundaries,
	}
	if err := validatePortableMSEState(state); err != nil {
		return nil, err
	}
	q := newQuantizerWithRotation(
		state.dim,
		state.bitWidth,
		state.seed,
		newDenseRotationFromMatrix(state.dim, state.rotation),
		cb,
	)
	q.portable = state.rotationKind
	return q, nil
}

// MarshalPortableIPQuantizer serializes an IP quantizer with its full MSE
// rotation matrix, codebook, and QJL projection matrix for cross-language
// interop.
func MarshalPortableIPQuantizer(q *IPQuantizer) ([]byte, error) {
	if q == nil {
		return nil, fmt.Errorf("turboquant: nil IP quantizer")
	}
	return marshalPortableState(
		portableTypeIP,
		q.dim,
		q.bitWidth,
		q.seed,
		q.mse.portable,
		q.mse.rotation.matrix(),
		q.mse.cb.centroids,
		q.mse.cb.boundaries,
		q.proj,
	)
}

// UnmarshalPortableIPQuantizer reconstructs an IP quantizer from portable
// binary state.
func UnmarshalPortableIPQuantizer(data []byte) (*IPQuantizer, error) {
	state, err := unmarshalPortableState(data)
	if err != nil {
		return nil, err
	}
	if state.kind != portableTypeIP {
		return nil, fmt.Errorf("turboquant: expected portable IP quantizer, got type %d", state.kind)
	}
	cb := codebook{
		centroids:  state.centroids,
		boundaries: state.boundaries,
	}
	if err := validatePortableIPState(state); err != nil {
		return nil, err
	}
	mseQ := newQuantizerWithRotation(
		state.dim,
		state.bitWidth-1,
		state.seed,
		newDenseRotationFromMatrix(state.dim, state.rotation),
		cb,
	)
	mseQ.portable = state.rotationKind
	proj := make([]float32, len(state.proj))
	copy(proj, state.proj)
	return newIPQuantizerWithProjection(state.dim, state.bitWidth, state.seed, mseQ, proj), nil
}

type portableState struct {
	kind         byte
	rotationKind rotationKind
	dim          int
	bitWidth     int
	seed         int64
	rotation     []float32
	centroids    []float32
	boundaries   []float32
	proj         []float32
}

func validatePortableMSEState(state portableState) error {
	if err := validateDim(state.dim); err != nil {
		return err
	}
	if err := validateBitWidth(state.bitWidth); err != nil {
		return err
	}
	switch state.rotationKind {
	case rotationKindDense, rotationKindHadamard:
	default:
		return fmt.Errorf("turboquant: unsupported portable rotation kind %d", state.rotationKind)
	}
	if len(state.rotation) != state.dim*state.dim {
		return fmt.Errorf("turboquant: portable rotation matrix has %d values want %d", len(state.rotation), state.dim*state.dim)
	}
	if want := 1 << uint(state.bitWidth); len(state.centroids) != want {
		return fmt.Errorf("turboquant: portable centroid count %d want %d", len(state.centroids), want)
	}
	if want := (1 << uint(state.bitWidth)) - 1; len(state.boundaries) != want {
		return fmt.Errorf("turboquant: portable boundary count %d want %d", len(state.boundaries), want)
	}
	return nil
}

func validatePortableIPState(state portableState) error {
	if err := validateIPBitWidth(state.bitWidth); err != nil {
		return err
	}
	if err := validatePortableMSEState(portableState{
		dim:          state.dim,
		bitWidth:     state.bitWidth - 1,
		rotationKind: state.rotationKind,
		rotation:     state.rotation,
		centroids:    state.centroids,
		boundaries:   state.boundaries,
	}); err != nil {
		return err
	}
	if len(state.proj) != state.dim*state.dim {
		return fmt.Errorf("turboquant: portable projection matrix has %d values want %d", len(state.proj), state.dim*state.dim)
	}
	return nil
}

func marshalPortableState(kind byte, dim, bitWidth int, seed int64, rotKind rotationKind, rotation, centroids, boundaries, proj []float32) ([]byte, error) {
	var buf bytes.Buffer
	buf.Grow(portableHeaderSize + 4*(len(rotation)+len(centroids)+len(boundaries)+len(proj)))
	buf.WriteString(portableMagic)
	buf.WriteByte(portableVersion)
	buf.WriteByte(kind)
	buf.WriteByte(byte(rotKind))
	buf.WriteByte(0)

	writePortableUint32(&buf, uint32(dim))
	writePortableUint32(&buf, uint32(bitWidth))
	writePortableUint64(&buf, uint64(seed))
	writePortableUint32(&buf, uint32(len(rotation)))
	writePortableUint32(&buf, uint32(len(centroids)))
	writePortableUint32(&buf, uint32(len(boundaries)))
	writePortableUint32(&buf, uint32(len(proj)))
	writePortableFloat32s(&buf, rotation)
	writePortableFloat32s(&buf, centroids)
	writePortableFloat32s(&buf, boundaries)
	writePortableFloat32s(&buf, proj)
	return buf.Bytes(), nil
}

func unmarshalPortableState(data []byte) (portableState, error) {
	if len(data) < portableHeaderSize {
		return portableState{}, fmt.Errorf("turboquant: portable data too short (%d bytes)", len(data))
	}
	if string(data[0:4]) != portableMagic {
		return portableState{}, fmt.Errorf("turboquant: invalid portable magic %q", data[0:4])
	}
	if data[4] != portableVersion {
		return portableState{}, fmt.Errorf("turboquant: unsupported portable version %d", data[4])
	}

	state := portableState{
		kind:         data[5],
		rotationKind: rotationKind(data[6]),
		dim:          int(binary.LittleEndian.Uint32(data[8:12])),
		bitWidth:     int(binary.LittleEndian.Uint32(data[12:16])),
		seed:         int64(binary.LittleEndian.Uint64(data[16:24])),
	}
	rotationLen := int(binary.LittleEndian.Uint32(data[24:28]))
	centroidLen := int(binary.LittleEndian.Uint32(data[28:32]))
	boundaryLen := int(binary.LittleEndian.Uint32(data[32:36]))
	projLen := int(binary.LittleEndian.Uint32(data[36:40]))

	offset := portableHeaderSize
	var err error
	state.rotation, offset, err = readPortableFloat32s(data, offset, rotationLen)
	if err != nil {
		return portableState{}, err
	}
	state.centroids, offset, err = readPortableFloat32s(data, offset, centroidLen)
	if err != nil {
		return portableState{}, err
	}
	state.boundaries, offset, err = readPortableFloat32s(data, offset, boundaryLen)
	if err != nil {
		return portableState{}, err
	}
	state.proj, offset, err = readPortableFloat32s(data, offset, projLen)
	if err != nil {
		return portableState{}, err
	}
	if offset != len(data) {
		return portableState{}, fmt.Errorf("turboquant: trailing portable data (%d bytes)", len(data)-offset)
	}
	return state, nil
}

func writePortableUint32(buf *bytes.Buffer, value uint32) {
	var scratch [4]byte
	binary.LittleEndian.PutUint32(scratch[:], value)
	buf.Write(scratch[:])
}

func writePortableUint64(buf *bytes.Buffer, value uint64) {
	var scratch [8]byte
	binary.LittleEndian.PutUint64(scratch[:], value)
	buf.Write(scratch[:])
}

func writePortableFloat32s(buf *bytes.Buffer, values []float32) {
	var scratch [4]byte
	for _, v := range values {
		binary.LittleEndian.PutUint32(scratch[:], math.Float32bits(v))
		buf.Write(scratch[:])
	}
}

func readPortableFloat32s(data []byte, offset, count int) ([]float32, int, error) {
	if count == 0 {
		return nil, offset, nil
	}
	bytesNeeded := count * 4
	if len(data) < offset+bytesNeeded {
		return nil, offset, fmt.Errorf("turboquant: truncated portable payload")
	}
	values := make([]float32, count)
	for i := 0; i < count; i++ {
		start := offset + i*4
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[start : start+4]))
	}
	return values, offset + bytesNeeded, nil
}
