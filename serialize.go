package turboquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"math"
)

const (
	serializeHeaderSizeV1 = 24
	serializeHeaderSizeV2 = 25
	serializeHeaderSizeV3 = 26
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
// rotation family. A dense or legacy single-round Hadamard rotation writes a
// 25-byte header. A multi-round Hadamard rotation writes a 26-byte header
// that carries the round count.
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
	return marshalRotationHeader(q.dim, q.bitWidth, q.seed, q.rotation), nil
}

// UnmarshalQuantizer reconstructs a Quantizer from serialized bytes.
func UnmarshalQuantizer(data []byte) (*Quantizer, error) {
	dim, bitWidth, seed, err := unmarshalHeaderFields(data)
	if err != nil {
		return nil, err
	}
	if err := validateBitWidth(bitWidth); err != nil {
		return nil, err
	}
	if len(data) < serializeHeaderSizeV2 {
		return NewDenseWithSeed(dim, bitWidth, seed), nil
	}
	rounds, err := decodeRotationHeader(data)
	if err != nil {
		return nil, err
	}
	switch rotationKind(data[24]) {
	case rotationKindDense:
		return NewDenseWithSeed(dim, bitWidth, seed), nil
	case rotationKindHadamard, rotationKindHadamardMulti:
		return NewHadamardRoundsWithSeed(dim, bitWidth, rounds, seed), nil
	default:
		return nil, fmt.Errorf("turboquant: unsupported rotation kind %d", data[24])
	}
}

// marshalRotationHeader writes the shared dim/bitWidth/seed/kind header used
// by both MarshalQuantizer and MarshalIPQuantizer. It emits a 26-byte header
// for a multi-round Hadamard rotation, and a 25-byte header otherwise.
func marshalRotationHeader(dim, bitWidth int, seed int64, rotation rotationState) []byte {
	size := serializeHeaderSizeV2
	if rotation.kind == rotationKindHadamardMulti {
		size = serializeHeaderSizeV3
	}
	buf := make([]byte, size)
	binary.LittleEndian.PutUint64(buf[0:8], uint64(dim))
	binary.LittleEndian.PutUint64(buf[8:16], uint64(bitWidth))
	binary.LittleEndian.PutUint64(buf[16:24], uint64(seed))
	buf[24] = byte(rotation.kind)
	if size == serializeHeaderSizeV3 {
		buf[25] = byte(len(rotation.rounds))
	}
	return buf
}

// unmarshalHeaderFields reads and validates the shared dim/bitWidth/seed
// header fields common to MarshalQuantizer and MarshalIPQuantizer.
func unmarshalHeaderFields(data []byte) (dim, bitWidth int, seed int64, err error) {
	if len(data) < serializeHeaderSizeV1 {
		return 0, 0, 0, fmt.Errorf("turboquant: serialized data too short (%d bytes)", len(data))
	}
	dim = int(binary.LittleEndian.Uint64(data[0:8]))
	bitWidth = int(binary.LittleEndian.Uint64(data[8:16]))
	seed = int64(binary.LittleEndian.Uint64(data[16:24]))
	if err := validateDim(dim); err != nil {
		return 0, 0, 0, err
	}
	return dim, bitWidth, seed, nil
}

// decodeRotationHeader reads the round count for a rotationKindHadamard or
// rotationKindHadamardMulti header. data must be at least
// serializeHeaderSizeV2 bytes; the caller has already checked that.
func decodeRotationHeader(data []byte) (rounds int, err error) {
	switch rotationKind(data[24]) {
	case rotationKindDense:
		return 0, nil
	case rotationKindHadamard:
		return 1, nil
	case rotationKindHadamardMulti:
		if len(data) < serializeHeaderSizeV3 {
			return 0, fmt.Errorf("turboquant: serialized data too short for multi-round header (%d bytes)", len(data))
		}
		rounds = int(data[25])
		if err := validateRounds(rounds); err != nil {
			return 0, err
		}
		return rounds, nil
	default:
		return 0, fmt.Errorf("turboquant: unsupported rotation kind %d", data[24])
	}
}

// MarshalIPQuantizer serializes an IPQuantizer to bytes. A dense or legacy
// single-round Hadamard rotation writes a 25-byte header. A multi-round
// Hadamard rotation writes a 26-byte header that carries the round count.
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
	rotation := rotationState{kind: rotationKindHadamard}
	if q.mse != nil {
		rotation = q.mse.rotation
	}
	return marshalRotationHeader(q.dim, q.bitWidth, q.seed, rotation), nil
}

// UnmarshalIPQuantizer reconstructs an IPQuantizer from serialized bytes.
func UnmarshalIPQuantizer(data []byte) (*IPQuantizer, error) {
	dim, bitWidth, seed, err := unmarshalHeaderFields(data)
	if err != nil {
		return nil, err
	}
	if err := validateIPBitWidth(bitWidth); err != nil {
		return nil, err
	}
	if len(data) < serializeHeaderSizeV2 {
		return NewIPDenseWithSeed(dim, bitWidth, seed), nil
	}
	rounds, err := decodeRotationHeader(data)
	if err != nil {
		return nil, err
	}
	switch rotationKind(data[24]) {
	case rotationKindDense:
		return NewIPDenseWithSeed(dim, bitWidth, seed), nil
	case rotationKindHadamard, rotationKindHadamardMulti:
		return NewIPHadamardRoundsWithSeed(dim, bitWidth, rounds, seed), nil
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
	family := rotationKindHadamard
	var rotation, centroids, boundaries []float32
	if q.mse != nil {
		family = q.mse.portable
		rotation = q.mse.rotation.matrix()
		centroids = q.mse.cb.centroids
		boundaries = q.mse.cb.boundaries
	}
	return marshalPortableState(
		portableTypeIP,
		q.dim,
		q.bitWidth,
		q.seed,
		family,
		rotation,
		centroids,
		boundaries,
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
	if err := validatePortableIPState(state); err != nil {
		return nil, err
	}
	if state.bitWidth == 1 {
		proj := make([]float32, len(state.proj))
		copy(proj, state.proj)
		return newIPQuantizerWithProjection(state.dim, 1, state.seed, nil, proj), nil
	}
	cb := codebook{
		centroids:  state.centroids,
		boundaries: state.boundaries,
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
	case rotationKindDense, rotationKindHadamard, rotationKindHadamardMulti:
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
	if state.bitWidth == 1 {
		// b=1 carries no MSE stage: only the QJL projection matrix.
		if err := validateDim(state.dim); err != nil {
			return err
		}
		if len(state.rotation) != 0 || len(state.centroids) != 0 || len(state.boundaries) != 0 {
			return fmt.Errorf("turboquant: portable b=1 IP state must not carry MSE rotation or codebook data")
		}
		if len(state.proj) != state.dim*state.dim {
			return fmt.Errorf("turboquant: portable projection matrix has %d values want %d", len(state.proj), state.dim*state.dim)
		}
		return nil
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
