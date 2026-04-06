package turboquant

import (
	"encoding/binary"
	"fmt"
)

const serializeHeaderSize = 24

// MarshalQuantizer serializes a Quantizer to bytes.
// The quantizer is fully reconstructible from dim, bitWidth, and seed.
func MarshalQuantizer(q *Quantizer) ([]byte, error) {
	buf := make([]byte, serializeHeaderSize)
	binary.LittleEndian.PutUint64(buf[0:8], uint64(q.dim))
	binary.LittleEndian.PutUint64(buf[8:16], uint64(q.bitWidth))
	binary.LittleEndian.PutUint64(buf[16:24], uint64(q.seed))
	return buf, nil
}

// UnmarshalQuantizer reconstructs a Quantizer from serialized bytes.
func UnmarshalQuantizer(data []byte) (*Quantizer, error) {
	if len(data) < serializeHeaderSize {
		return nil, fmt.Errorf("turboquant: serialized data too short (%d bytes)", len(data))
	}
	dim := int(binary.LittleEndian.Uint64(data[0:8]))
	bitWidth := int(binary.LittleEndian.Uint64(data[8:16]))
	seed := int64(binary.LittleEndian.Uint64(data[16:24]))
	return NewWithSeed(dim, bitWidth, seed), nil
}

// MarshalIPQuantizer serializes an IPQuantizer to bytes.
func MarshalIPQuantizer(q *IPQuantizer) ([]byte, error) {
	buf := make([]byte, serializeHeaderSize)
	binary.LittleEndian.PutUint64(buf[0:8], uint64(q.dim))
	binary.LittleEndian.PutUint64(buf[8:16], uint64(q.bitWidth))
	binary.LittleEndian.PutUint64(buf[16:24], uint64(q.seed))
	return buf, nil
}

// UnmarshalIPQuantizer reconstructs an IPQuantizer from serialized bytes.
func UnmarshalIPQuantizer(data []byte) (*IPQuantizer, error) {
	if len(data) < serializeHeaderSize {
		return nil, fmt.Errorf("turboquant: serialized data too short (%d bytes)", len(data))
	}
	dim := int(binary.LittleEndian.Uint64(data[0:8]))
	bitWidth := int(binary.LittleEndian.Uint64(data[8:16]))
	seed := int64(binary.LittleEndian.Uint64(data[16:24]))
	return NewIPWithSeed(dim, bitWidth, seed), nil
}
