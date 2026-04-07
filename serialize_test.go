package turboquant

import (
	"bytes"
	"encoding/hex"
	"testing"
)

const (
	compactMSEGoldenHex  = "040000000000000002000000000000002a0000000000000002"
	portableMSEGoldenHex = "545150420101020004000000020000002a00000000000000100000000400000003000000000000000000003f0000003f0000003f000000bf000000bf0000003f0000003f0000003f0000003f000000bf0000003f0000003f0000003f0000003f000000bf0000003f95a62cbf6bb660be6bb6603e95a62c3f30d4e4be0000a8a530d4e43e"
	compactIPGoldenHex   = "040000000000000003000000000000002a0000000000000002"
	portableIPGoldenHex  = "545150420102020004000000030000002a00000000000000100000000400000003000000100000000000003f0000003f0000003f000000bf000000bf0000003f0000003f0000003f0000003f000000bf0000003f0000003f0000003f0000003f000000bf0000003f95a62cbf6bb660be6bb6603e95a62c3f30d4e4be0000a8a530d4e43e82d0233fa8c94cbed520563f62fe093fdb846dbf7490583e4b770a40346f8c3dafe2c93f2b79a03f33e2ba3f17dec9bd10d651beb16764be7894ab3f73dc4a3f"
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

func TestSerializeIPHadamardRoundTrip(t *testing.T) {
	q := NewIPHadamardWithSeed(16, 3, 12345)
	data, err := MarshalIPQuantizer(q)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	q2, err := UnmarshalIPQuantizer(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if q2.RotationKind() != "hadamard" {
		t.Fatalf("RotationKind() = %q want hadamard", q2.RotationKind())
	}
}

func TestSerializeRejectsTruncated(t *testing.T) {
	_, err := UnmarshalQuantizer([]byte{1, 2, 3})
	if err == nil {
		t.Fatal("expected error for truncated data")
	}
}

func TestSerializeHadamardRoundTrip(t *testing.T) {
	q := NewHadamardWithSeed(16, 3, 12345)
	data, err := MarshalQuantizer(q)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	q2, err := UnmarshalQuantizer(data)
	if err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if q2.RotationKind() != "hadamard" {
		t.Fatalf("RotationKind() = %q want hadamard", q2.RotationKind())
	}
}

func TestSerializeLegacyDenseRoundTrip(t *testing.T) {
	q := NewWithSeed(16, 3, 12345)
	data, err := MarshalQuantizer(q)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	legacy := data[:serializeHeaderSizeV1]
	q2, err := UnmarshalQuantizer(legacy)
	if err != nil {
		t.Fatalf("unmarshal legacy: %v", err)
	}
	if q2.RotationKind() != "dense" {
		t.Fatalf("legacy RotationKind() = %q want dense", q2.RotationKind())
	}
}

func TestPortableSerializeMSERoundTripDense(t *testing.T) {
	q := NewWithSeed(16, 3, 12345)
	data, err := MarshalPortableQuantizer(q)
	if err != nil {
		t.Fatalf("marshal portable: %v", err)
	}
	q2, err := UnmarshalPortableQuantizer(data)
	if err != nil {
		t.Fatalf("unmarshal portable: %v", err)
	}
	data2, err := MarshalPortableQuantizer(q2)
	if err != nil {
		t.Fatalf("marshal portable round-trip: %v", err)
	}
	if !bytes.Equal(data, data2) {
		t.Fatal("portable dense round-trip changed bytes")
	}
}

func TestPortableSerializeMSERoundTripHadamard(t *testing.T) {
	q := NewHadamardWithSeed(16, 3, 12345)
	data, err := MarshalPortableQuantizer(q)
	if err != nil {
		t.Fatalf("marshal portable: %v", err)
	}
	q2, err := UnmarshalPortableQuantizer(data)
	if err != nil {
		t.Fatalf("unmarshal portable: %v", err)
	}
	if q2.RotationKind() != "dense" {
		t.Fatalf("portable quantizer loaded as %q want dense", q2.RotationKind())
	}
	data2, err := MarshalPortableQuantizer(q2)
	if err != nil {
		t.Fatalf("marshal portable round-trip: %v", err)
	}
	if !bytes.Equal(data, data2) {
		t.Fatal("portable hadamard round-trip changed bytes")
	}
}

func TestPortableSerializeIPRoundTrip(t *testing.T) {
	q := NewIPHadamardWithSeed(16, 3, 12345)
	data, err := MarshalPortableIPQuantizer(q)
	if err != nil {
		t.Fatalf("marshal portable IP: %v", err)
	}
	q2, err := UnmarshalPortableIPQuantizer(data)
	if err != nil {
		t.Fatalf("unmarshal portable IP: %v", err)
	}
	data2, err := MarshalPortableIPQuantizer(q2)
	if err != nil {
		t.Fatalf("marshal portable IP round-trip: %v", err)
	}
	if !bytes.Equal(data, data2) {
		t.Fatal("portable IP round-trip changed bytes")
	}
}

func TestPortableSerializeRejectsTruncated(t *testing.T) {
	_, err := UnmarshalPortableQuantizer([]byte("TQPB"))
	if err == nil {
		t.Fatal("expected error for truncated portable data")
	}
}

func TestSerializeGoldens(t *testing.T) {
	q := NewHadamardWithSeed(4, 2, 42)
	gotCompactMSE, err := MarshalQuantizer(q)
	if err != nil {
		t.Fatalf("marshal compact mse: %v", err)
	}
	wantCompactMSE, _ := hex.DecodeString(compactMSEGoldenHex)
	if !bytes.Equal(gotCompactMSE, wantCompactMSE) {
		t.Fatal("compact MSE golden mismatch")
	}

	gotPortableMSE, err := MarshalPortableQuantizer(q)
	if err != nil {
		t.Fatalf("marshal portable mse: %v", err)
	}
	wantPortableMSE, _ := hex.DecodeString(portableMSEGoldenHex)
	if !bytes.Equal(gotPortableMSE, wantPortableMSE) {
		t.Fatal("portable MSE golden mismatch")
	}

	ipq := NewIPHadamardWithSeed(4, 3, 42)
	gotCompactIP, err := MarshalIPQuantizer(ipq)
	if err != nil {
		t.Fatalf("marshal compact ip: %v", err)
	}
	wantCompactIP, _ := hex.DecodeString(compactIPGoldenHex)
	if !bytes.Equal(gotCompactIP, wantCompactIP) {
		t.Fatal("compact IP golden mismatch")
	}

	gotPortableIP, err := MarshalPortableIPQuantizer(ipq)
	if err != nil {
		t.Fatalf("marshal portable ip: %v", err)
	}
	wantPortableIP, _ := hex.DecodeString(portableIPGoldenHex)
	if !bytes.Equal(gotPortableIP, wantPortableIP) {
		t.Fatal("portable IP golden mismatch")
	}
}
