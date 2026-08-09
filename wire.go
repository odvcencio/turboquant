package turboquant

import (
	"encoding/binary"
	"fmt"
	"math"
)

// The version byte (data[2]) is per record type: MSE records use wireVersion,
// IP records use wireVersionIP2 (or wireVersion for legacy IP payloads). Read
// the type byte (data[3]) alongside the version byte to classify a record.
const (
	wireMagic   = "TQ"
	wireVersion = 1
	// wireVersionIP2 extends the IP record header with the input vector norm.
	// Version-1 IP records decode with Norm = 1, preserving their original
	// unit-space semantics.
	wireVersionIP2   = 2
	wireTypeMSE      = 1
	wireTypeIP       = 2
	wireHeaderSize   = 22
	wireIPHeaderSize = 26
)

// EncodeMSE encodes an MSE-quantized vector to the TurboQuant wire format.
func EncodeMSE(dim, bitWidth int, packed []byte, norm float32) []byte {
	panicOnInvalid("turboquant.EncodeMSE", validateWireDim(dim))
	panicOnInvalid("turboquant.EncodeMSE", ValidatePacked(dim, bitWidth, packed))
	if math.IsNaN(float64(norm)) || math.IsInf(float64(norm), 0) {
		panic(fmt.Sprintf("turboquant.EncodeMSE: turboquant: invalid norm %v", norm))
	}
	buf := make([]byte, wireHeaderSize+len(packed))
	buf[0], buf[1] = 'T', 'Q'
	buf[2] = wireVersion
	buf[3] = wireTypeMSE
	binary.BigEndian.PutUint16(buf[4:6], uint16(dim))
	buf[6] = byte(bitWidth)
	buf[7] = 0 // reserved
	binary.BigEndian.PutUint32(buf[8:12], math.Float32bits(norm))
	binary.BigEndian.PutUint32(buf[12:16], uint32(len(packed)))
	binary.BigEndian.PutUint32(buf[16:20], 0) // no signs for MSE
	buf[20], buf[21] = 0, 0                   // reserved
	copy(buf[wireHeaderSize:], packed)
	return buf
}

// DecodeMSE decodes an MSE-quantized vector from the TurboQuant wire format.
func DecodeMSE(data []byte) (dim, bitWidth int, packed []byte, norm float32, err error) {
	if len(data) < wireHeaderSize {
		return 0, 0, nil, 0, fmt.Errorf("turboquant: data too short (%d bytes)", len(data))
	}
	if data[0] != 'T' || data[1] != 'Q' {
		return 0, 0, nil, 0, fmt.Errorf("turboquant: invalid magic %q", data[0:2])
	}
	if data[2] != wireVersion {
		return 0, 0, nil, 0, fmt.Errorf("turboquant: unsupported version %d", data[2])
	}
	if data[3] != wireTypeMSE {
		return 0, 0, nil, 0, fmt.Errorf("turboquant: expected MSE type, got %d", data[3])
	}
	dim = int(binary.BigEndian.Uint16(data[4:6]))
	bitWidth = int(data[6])
	if err := validateWireDim(dim); err != nil {
		return 0, 0, nil, 0, err
	}
	if err := validateBitWidth(bitWidth); err != nil {
		return 0, 0, nil, 0, err
	}
	norm = math.Float32frombits(binary.BigEndian.Uint32(data[8:12]))
	dataLen := int(binary.BigEndian.Uint32(data[12:16]))
	if len(data) != wireHeaderSize+dataLen {
		return 0, 0, nil, 0, fmt.Errorf("turboquant: truncated payload")
	}
	if want := PackedSize(dim, bitWidth); dataLen != want {
		return 0, 0, nil, 0, fmt.Errorf("turboquant: invalid MSE payload length %d want %d", dataLen, want)
	}
	packed = make([]byte, dataLen)
	copy(packed, data[wireHeaderSize:wireHeaderSize+dataLen])
	return dim, bitWidth, packed, norm, nil
}

// EncodeIP encodes an IP-quantized vector to the TurboQuant wire format
// (version 2, which carries the input vector norm).
func EncodeIP(dim, bitWidth int, qx IPQuantized) []byte {
	panicOnInvalid("turboquant.EncodeIP", validateWireDim(dim))
	panicOnInvalid("turboquant.EncodeIP", ValidateIPQuantized(dim, bitWidth, qx))
	buf := make([]byte, wireIPHeaderSize+len(qx.MSE)+len(qx.Signs))
	buf[0], buf[1] = 'T', 'Q'
	buf[2] = wireVersionIP2
	buf[3] = wireTypeIP
	binary.BigEndian.PutUint16(buf[4:6], uint16(dim))
	buf[6] = byte(bitWidth)
	buf[7] = 0
	binary.BigEndian.PutUint32(buf[8:12], math.Float32bits(qx.ResNorm))
	binary.BigEndian.PutUint32(buf[12:16], uint32(len(qx.MSE)))
	binary.BigEndian.PutUint32(buf[16:20], uint32(len(qx.Signs)))
	buf[20], buf[21] = 0, 0
	binary.BigEndian.PutUint32(buf[22:26], math.Float32bits(qx.Norm))
	copy(buf[wireIPHeaderSize:], qx.MSE)
	copy(buf[wireIPHeaderSize+len(qx.MSE):], qx.Signs)
	return buf
}

// DecodeIP decodes an IP-quantized vector from the TurboQuant wire format.
// It accepts version 1 (legacy, no input norm; Norm defaults to 1) and
// version 2 payloads.
func DecodeIP(data []byte) (dim, bitWidth int, qx IPQuantized, err error) {
	if len(data) < wireHeaderSize {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: data too short (%d bytes)", len(data))
	}
	if data[0] != 'T' || data[1] != 'Q' {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: invalid magic %q", data[0:2])
	}
	headerSize := wireHeaderSize
	switch data[2] {
	case wireVersion:
	case wireVersionIP2:
		headerSize = wireIPHeaderSize
		if len(data) < headerSize {
			return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: data too short (%d bytes)", len(data))
		}
	default:
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: unsupported version %d", data[2])
	}
	if data[3] != wireTypeIP {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: expected IP type, got %d", data[3])
	}
	dim = int(binary.BigEndian.Uint16(data[4:6]))
	bitWidth = int(data[6])
	if err := validateWireDim(dim); err != nil {
		return 0, 0, IPQuantized{}, err
	}
	if err := validateIPBitWidth(bitWidth); err != nil {
		return 0, 0, IPQuantized{}, err
	}
	qx.ResNorm = math.Float32frombits(binary.BigEndian.Uint32(data[8:12]))
	if math.IsNaN(float64(qx.ResNorm)) || math.IsInf(float64(qx.ResNorm), 0) || qx.ResNorm < 0 {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: invalid residual norm %v", qx.ResNorm)
	}
	qx.Norm = 1
	if data[2] == wireVersionIP2 {
		qx.Norm = math.Float32frombits(binary.BigEndian.Uint32(data[22:26]))
	}
	if math.IsNaN(float64(qx.Norm)) || math.IsInf(float64(qx.Norm), 0) || qx.Norm < 0 {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: invalid input norm %v", qx.Norm)
	}
	mseLen := int(binary.BigEndian.Uint32(data[12:16]))
	signsLen := int(binary.BigEndian.Uint32(data[16:20]))
	if len(data) != headerSize+mseLen+signsLen {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: truncated payload")
	}
	if want := PackedSize(dim, bitWidth-1); mseLen != want {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: invalid IP MSE payload length %d want %d", mseLen, want)
	}
	if want := (dim + 7) / 8; signsLen != want {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: invalid IP sign payload length %d want %d", signsLen, want)
	}
	qx.MSE = make([]byte, mseLen)
	copy(qx.MSE, data[headerSize:headerSize+mseLen])
	qx.Signs = make([]byte, signsLen)
	copy(qx.Signs, data[headerSize+mseLen:headerSize+mseLen+signsLen])
	return dim, bitWidth, qx, nil
}
