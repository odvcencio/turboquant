package turboquant

import (
	"encoding/binary"
	"fmt"
	"math"
)

const (
	wireMagic      = "TQ"
	wireVersion    = 1
	wireTypeMSE    = 1
	wireTypeIP     = 2
	wireHeaderSize = 22
)

// EncodeMSE encodes an MSE-quantized vector to the TurboQuant wire format.
func EncodeMSE(dim, bitWidth int, packed []byte, norm float32) []byte {
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
	norm = math.Float32frombits(binary.BigEndian.Uint32(data[8:12]))
	dataLen := int(binary.BigEndian.Uint32(data[12:16]))
	if len(data) < wireHeaderSize+dataLen {
		return 0, 0, nil, 0, fmt.Errorf("turboquant: truncated payload")
	}
	packed = make([]byte, dataLen)
	copy(packed, data[wireHeaderSize:wireHeaderSize+dataLen])
	return dim, bitWidth, packed, norm, nil
}

// EncodeIP encodes an IP-quantized vector to the TurboQuant wire format.
func EncodeIP(dim, bitWidth int, qx IPQuantized) []byte {
	buf := make([]byte, wireHeaderSize+len(qx.MSE)+len(qx.Signs))
	buf[0], buf[1] = 'T', 'Q'
	buf[2] = wireVersion
	buf[3] = wireTypeIP
	binary.BigEndian.PutUint16(buf[4:6], uint16(dim))
	buf[6] = byte(bitWidth)
	buf[7] = 0
	binary.BigEndian.PutUint32(buf[8:12], math.Float32bits(qx.ResNorm))
	binary.BigEndian.PutUint32(buf[12:16], uint32(len(qx.MSE)))
	binary.BigEndian.PutUint32(buf[16:20], uint32(len(qx.Signs)))
	buf[20], buf[21] = 0, 0
	copy(buf[wireHeaderSize:], qx.MSE)
	copy(buf[wireHeaderSize+len(qx.MSE):], qx.Signs)
	return buf
}

// DecodeIP decodes an IP-quantized vector from the TurboQuant wire format.
func DecodeIP(data []byte) (dim, bitWidth int, qx IPQuantized, err error) {
	if len(data) < wireHeaderSize {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: data too short (%d bytes)", len(data))
	}
	if data[0] != 'T' || data[1] != 'Q' {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: invalid magic %q", data[0:2])
	}
	if data[2] != wireVersion {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: unsupported version %d", data[2])
	}
	if data[3] != wireTypeIP {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: expected IP type, got %d", data[3])
	}
	dim = int(binary.BigEndian.Uint16(data[4:6]))
	bitWidth = int(data[6])
	qx.ResNorm = math.Float32frombits(binary.BigEndian.Uint32(data[8:12]))
	mseLen := int(binary.BigEndian.Uint32(data[12:16]))
	signsLen := int(binary.BigEndian.Uint32(data[16:20]))
	if len(data) < wireHeaderSize+mseLen+signsLen {
		return 0, 0, IPQuantized{}, fmt.Errorf("turboquant: truncated payload")
	}
	qx.MSE = make([]byte, mseLen)
	copy(qx.MSE, data[wireHeaderSize:wireHeaderSize+mseLen])
	qx.Signs = make([]byte, signsLen)
	copy(qx.Signs, data[wireHeaderSize+mseLen:wireHeaderSize+mseLen+signsLen])
	return dim, bitWidth, qx, nil
}
