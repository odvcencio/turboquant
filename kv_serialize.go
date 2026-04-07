package turboquant

import (
	"bytes"
	"encoding/binary"
	"fmt"
)

const (
	kvPageMagic   = "TQKV"
	kvPageVersion = 1
)

// StorageBytes reports the currently allocated storage footprint for this page,
// excluding optional GPU-resident mirrors.
func (p *KVCachePage) StorageBytes() uint64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.storageBytesLocked()
}

// LiveBytes reports the storage currently used by populated KV entries,
// excluding optional GPU-resident mirrors.
func (p *KVCachePage) LiveBytes() uint64 {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.liveBytesLocked()
}

// MarshalBinary serializes the quantized KV page, including quantizer config,
// allocated capacity, and currently populated entries.
func (p *KVCachePage) MarshalBinary() ([]byte, error) {
	if p == nil {
		return nil, fmt.Errorf("turboquant: nil KV cache page")
	}
	keyQ, err := MarshalIPQuantizer(p.keyQ)
	if err != nil {
		return nil, err
	}
	valueQ, err := MarshalQuantizer(p.valueQ)
	if err != nil {
		return nil, err
	}

	p.mu.RLock()
	defer p.mu.RUnlock()

	buf := new(bytes.Buffer)
	write := func(value any) error {
		return binary.Write(buf, binary.LittleEndian, value)
	}
	if _, err := buf.WriteString(kvPageMagic); err != nil {
		return nil, err
	}
	if err := write(uint32(kvPageVersion)); err != nil {
		return nil, err
	}
	if err := write(uint32(len(keyQ))); err != nil {
		return nil, err
	}
	if err := write(uint32(len(valueQ))); err != nil {
		return nil, err
	}
	if err := write(uint32(p.length)); err != nil {
		return nil, err
	}
	if err := write(uint32(cap(p.keyResNorms))); err != nil {
		return nil, err
	}
	if _, err := buf.Write(keyQ); err != nil {
		return nil, err
	}
	if _, err := buf.Write(valueQ); err != nil {
		return nil, err
	}
	if _, err := buf.Write(p.keyMSE[:p.length*p.keyMSEBytes]); err != nil {
		return nil, err
	}
	if _, err := buf.Write(p.keySigns[:p.length*p.keySignBytes]); err != nil {
		return nil, err
	}
	if err := writeFloat32s(buf, p.keyResNorms[:p.length]); err != nil {
		return nil, err
	}
	if _, err := buf.Write(p.valuePacked[:p.length*p.valueBytes]); err != nil {
		return nil, err
	}
	if err := writeFloat32s(buf, p.valueNorms[:p.length]); err != nil {
		return nil, err
	}
	return buf.Bytes(), nil
}

// UnmarshalKVCachePage reconstructs a KV cache page from MarshalBinary output.
func UnmarshalKVCachePage(data []byte) (*KVCachePage, error) {
	if len(data) < 24 {
		return nil, fmt.Errorf("turboquant: serialized KV page too short (%d bytes)", len(data))
	}
	reader := bytes.NewReader(data)
	magic := make([]byte, len(kvPageMagic))
	if _, err := reader.Read(magic); err != nil {
		return nil, err
	}
	if string(magic) != kvPageMagic {
		return nil, fmt.Errorf("turboquant: invalid KV page magic %q", string(magic))
	}
	var version, keyQLen, valueQLen, length, capacity uint32
	read := func(dst any) error {
		return binary.Read(reader, binary.LittleEndian, dst)
	}
	if err := read(&version); err != nil {
		return nil, err
	}
	if version != kvPageVersion {
		return nil, fmt.Errorf("turboquant: unsupported KV page version %d", version)
	}
	if err := read(&keyQLen); err != nil {
		return nil, err
	}
	if err := read(&valueQLen); err != nil {
		return nil, err
	}
	if err := read(&length); err != nil {
		return nil, err
	}
	if err := read(&capacity); err != nil {
		return nil, err
	}
	if capacity < length {
		return nil, fmt.Errorf("turboquant: KV page capacity %d smaller than length %d", capacity, length)
	}
	keyQBytes := make([]byte, keyQLen)
	if _, err := reader.Read(keyQBytes); err != nil {
		return nil, err
	}
	valueQBytes := make([]byte, valueQLen)
	if _, err := reader.Read(valueQBytes); err != nil {
		return nil, err
	}
	keyQ, err := UnmarshalIPQuantizer(keyQBytes)
	if err != nil {
		return nil, err
	}
	valueQ, err := UnmarshalQuantizer(valueQBytes)
	if err != nil {
		return nil, err
	}
	page := NewKVCachePageWithQuantizers(keyQ, valueQ, int(capacity))
	page.length = int(length)
	if _, err := reader.Read(page.keyMSE[:page.length*page.keyMSEBytes]); err != nil {
		return nil, err
	}
	if _, err := reader.Read(page.keySigns[:page.length*page.keySignBytes]); err != nil {
		return nil, err
	}
	if err := readFloat32s(reader, page.keyResNorms[:page.length]); err != nil {
		return nil, err
	}
	if _, err := reader.Read(page.valuePacked[:page.length*page.valueBytes]); err != nil {
		return nil, err
	}
	if err := readFloat32s(reader, page.valueNorms[:page.length]); err != nil {
		return nil, err
	}
	if reader.Len() != 0 {
		return nil, fmt.Errorf("turboquant: unexpected trailing KV page bytes (%d)", reader.Len())
	}
	return page, nil
}

func (p *KVCachePage) storageBytesLocked() uint64 {
	return uint64(len(p.keyMSE) + len(p.keySigns) + len(p.valuePacked) + len(p.keyResNorms)*4 + len(p.valueNorms)*4)
}

func (p *KVCachePage) liveBytesLocked() uint64 {
	return uint64(p.length*p.keyMSEBytes + p.length*p.keySignBytes + p.length*p.valueBytes + p.length*4 + p.length*4)
}

func writeFloat32s(buf *bytes.Buffer, values []float32) error {
	for _, value := range values {
		if err := binary.Write(buf, binary.LittleEndian, value); err != nil {
			return err
		}
	}
	return nil
}

func readFloat32s(reader *bytes.Reader, dst []float32) error {
	for i := range dst {
		if err := binary.Read(reader, binary.LittleEndian, &dst[i]); err != nil {
			return err
		}
	}
	return nil
}
