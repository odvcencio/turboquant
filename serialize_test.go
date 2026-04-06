package turboquant

import (
	"testing"
)

func TestSerializeMSERoundTrip(t *testing.T) {
	q := NewWithSeed(16, 3, 12345)
	vec := randomUnitVector(16, newTestRNG())

	packed, norm := q.Quantize(vec)

	data, err := MarshalQuantizer(q)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	q2, err := UnmarshalQuantizer(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	packed2, norm2 := q2.Quantize(vec)
	if norm != norm2 {
		t.Fatalf("norm mismatch: %v != %v", norm, norm2)
	}
	if len(packed) != len(packed2) {
		t.Fatalf("packed length mismatch")
	}
	for i := range packed {
		if packed[i] != packed2[i] {
			t.Fatalf("packed byte %d mismatch", i)
		}
	}
}

func TestSerializeIPRoundTrip(t *testing.T) {
	q := NewIPWithSeed(16, 3, 12345)
	vec := randomUnitVector(16, newTestRNG())

	qx := q.Quantize(vec)

	data, err := MarshalIPQuantizer(q)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}

	q2, err := UnmarshalIPQuantizer(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	qx2 := q2.Quantize(vec)
	if qx.ResNorm != qx2.ResNorm {
		t.Fatalf("resNorm mismatch")
	}
}

func TestSerializeRejectsTruncated(t *testing.T) {
	_, err := UnmarshalQuantizer([]byte{1, 2, 3})
	if err == nil {
		t.Fatal("expected error for truncated data")
	}
}
