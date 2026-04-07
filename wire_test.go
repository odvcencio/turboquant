package turboquant

import (
	"bytes"
	"testing"
)

func TestWireMSERoundTrip(t *testing.T) {
	q := NewWithSeed(8, 2, 42)
	vec := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	packed, norm := q.Quantize(vec)

	encoded := EncodeMSE(8, 2, packed, norm)
	dim, bitWidth, gotPacked, gotNorm, err := DecodeMSE(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if dim != 8 || bitWidth != 2 {
		t.Fatalf("header mismatch: dim=%d bitWidth=%d", dim, bitWidth)
	}
	if gotNorm != norm {
		t.Fatalf("norm mismatch: %v != %v", gotNorm, norm)
	}
	if !bytes.Equal(gotPacked, packed) {
		t.Fatal("packed data mismatch")
	}
}

func TestWireIPRoundTrip(t *testing.T) {
	q := NewIPWithSeed(8, 3, 42)
	vec := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	qx := q.Quantize(vec)

	encoded := EncodeIP(8, 3, qx)
	dim, bitWidth, gotQx, err := DecodeIP(encoded)
	if err != nil {
		t.Fatalf("decode: %v", err)
	}
	if dim != 8 || bitWidth != 3 {
		t.Fatalf("header mismatch: dim=%d bitWidth=%d", dim, bitWidth)
	}
	if gotQx.ResNorm != qx.ResNorm {
		t.Fatalf("resNorm mismatch")
	}
	if !bytes.Equal(gotQx.MSE, qx.MSE) {
		t.Fatal("MSE data mismatch")
	}
	if !bytes.Equal(gotQx.Signs, qx.Signs) {
		t.Fatal("Signs data mismatch")
	}
}

func TestWireRejectsBadMagic(t *testing.T) {
	bad := make([]byte, wireHeaderSize+2)
	bad[0], bad[1] = 'X', 'X'
	_, _, _, _, err := DecodeMSE(bad)
	if err == nil {
		t.Fatal("expected error for bad magic")
	}
}

func TestWireRejectsTruncated(t *testing.T) {
	_, _, _, _, err := DecodeMSE([]byte("TQ\x01"))
	if err == nil {
		t.Fatal("expected error for truncated header")
	}
}

func TestWireRejectsWrongType(t *testing.T) {
	q := NewWithSeed(8, 2, 42)
	vec := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	packed, norm := q.Quantize(vec)
	encoded := EncodeMSE(8, 2, packed, norm)

	// Try to decode MSE as IP
	_, _, _, err := DecodeIP(encoded)
	if err == nil {
		t.Fatal("expected error for wrong type")
	}
}

func TestWireEncodeRejectsWrongPackedLength(t *testing.T) {
	expectPanic(t, func() {
		EncodeMSE(8, 2, []byte{1}, 1)
	})
}

func TestWireDecodeRejectsInvalidPayloadLength(t *testing.T) {
	q := NewWithSeed(8, 2, 42)
	vec := []float32{1, 2, 3, 4, 5, 6, 7, 8}
	packed, norm := q.Quantize(vec)
	encoded := EncodeMSE(8, 2, packed, norm)
	encoded = append(encoded, 0)
	_, _, _, _, err := DecodeMSE(encoded)
	if err == nil {
		t.Fatal("expected error for invalid payload length")
	}
}
