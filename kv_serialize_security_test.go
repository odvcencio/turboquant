package turboquant

import (
	"encoding/binary"
	"testing"
)

// TestUnmarshalKVCachePageRejectsHugeCapacity pins the B1 fix: a crafted
// header that declares a multi-gigabyte capacity from a tiny payload must be
// rejected before allocating, not decoded into a giant page.
func TestUnmarshalKVCachePageRejectsHugeCapacity(t *testing.T) {
	page := NewKVCachePageWithSeed(16, 3, 16, 2, 8, 42)
	page.Append(testVector(16, 1), testVector(16, 2))
	data, err := page.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	// The capacity field is the fifth uint32 after the 4-byte magic:
	// version, keyQLen, valueQLen, length, capacity.
	const capOffset = 4 + 4*4
	binary.LittleEndian.PutUint32(data[capOffset:capOffset+4], 0x0FFFFFFF)

	if _, err := UnmarshalKVCachePage(data); err == nil {
		t.Fatal("expected an error for an oversized capacity, got nil (memory-exhaustion vector open)")
	}
}

// TestUnmarshalKVCachePageEmptyRoundTrip pins the M1 fix: a zero-length page
// must survive marshal/unmarshal instead of failing with EOF on the first
// empty read.
func TestUnmarshalKVCachePageEmptyRoundTrip(t *testing.T) {
	page := NewKVCachePageWithSeed(16, 3, 16, 2, 0, 42)
	data, err := page.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := UnmarshalKVCachePage(data)
	if err != nil {
		t.Fatalf("empty page round-trip failed: %v", err)
	}
	if restored.Len() != 0 {
		t.Fatalf("restored empty page has length %d, want 0", restored.Len())
	}
}

// TestUnmarshalKVCachePageRejectsTruncatedEntries pins the short-read hardening:
// a header that claims more entries than the payload holds must error, not
// silently accept a partial read.
func TestUnmarshalKVCachePageRejectsTruncatedEntries(t *testing.T) {
	page := NewKVCachePageWithSeed(16, 3, 16, 2, 8, 42)
	for i := 0; i < 4; i++ {
		page.Append(testVector(16, float32(i+1)), testVector(16, float32(i+5)))
	}
	data, err := page.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	if _, err := UnmarshalKVCachePage(data[:len(data)-8]); err == nil {
		t.Fatal("expected an error for a truncated payload, got nil")
	}
}
