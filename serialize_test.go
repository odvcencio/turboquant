package turboquant

import (
	"bytes"
	"encoding/hex"
	"testing"
)

const (
	// compactMSEGoldenHex and portableMSEGoldenHex reflect the default
	// (DefaultHadamardRounds, currently 3-round) Hadamard rotation and the
	// prefix-sum-quadrature codebook solver with the Panter-Dite compander
	// seed (the near-zero boundary moved by one float32 ulp when the seed
	// changed; the converged codebook is otherwise identical).
	compactMSEGoldenHex  = "040000000000000002000000000000002a000000000000000303"
	portableMSEGoldenHex = "545150420101030004000000020000002a0000000000000010000000040000000300000000000000000000bf000000bf0000003f0000003f0000003f0000003f0000003f0000003f000000bf0000003f000000bf0000003f000000bf0000003f0000003f000000bf96a62cbf6cb660be6cb6603e96a62c3f31d4e4be004073a831d4e43e"
	compactIPGoldenHex   = "040000000000000003000000000000002a000000000000000303"
	portableIPGoldenHex  = "545150420102030004000000030000002a0000000000000010000000040000000300000010000000000000bf000000bf0000003f0000003f0000003f0000003f0000003f0000003f000000bf0000003f000000bf0000003f000000bf0000003f0000003f000000bf96a62cbf6cb660be6cb6603e96a62c3f31d4e4be004073a831d4e43e82d0233fa8c94cbed520563f62fe093fdb846dbf7490583e4b770a40346f8c3dafe2c93f2b79a03f33e2ba3f17dec9bd10d651beb16764be7894ab3f73dc4a3f"

	// legacyCompactMSEGoldenHex is a real pre-Phase-2 25-byte payload: a
	// single-round Hadamard rotation, kind byte 2, no round-count byte. It
	// guards decode compatibility for data written before multi-round
	// rotations existed.
	legacyCompactMSEGoldenHex = "040000000000000002000000000000002a0000000000000002"
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
	if q2.RotationKind() != "hadamard-multi" {
		t.Fatalf("RotationKind() = %q want hadamard-multi", q2.RotationKind())
	}
	if q2.mse.Rounds() != DefaultHadamardRounds {
		t.Fatalf("Rounds() = %d want %d", q2.mse.Rounds(), DefaultHadamardRounds)
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
	if q2.RotationKind() != "hadamard-multi" {
		t.Fatalf("RotationKind() = %q want hadamard-multi", q2.RotationKind())
	}
	if q2.Rounds() != DefaultHadamardRounds {
		t.Fatalf("Rounds() = %d want %d", q2.Rounds(), DefaultHadamardRounds)
	}
}

func TestSerializeHadamardRoundsRoundTrip(t *testing.T) {
	for _, rounds := range []int{1, 2, 3, 4} {
		q := NewHadamardRoundsWithSeed(16, 3, rounds, 12345)
		vec := randomUnitVector(16, newTestRNG())
		packed, norm := q.Quantize(vec)

		data, err := MarshalQuantizer(q)
		if err != nil {
			t.Fatalf("rounds=%d: marshal: %v", rounds, err)
		}
		wantLen := serializeHeaderSizeV2
		if rounds != 1 {
			wantLen = serializeHeaderSizeV3
		}
		if len(data) != wantLen {
			t.Fatalf("rounds=%d: header length = %d want %d", rounds, len(data), wantLen)
		}

		q2, err := UnmarshalQuantizer(data)
		if err != nil {
			t.Fatalf("rounds=%d: unmarshal: %v", rounds, err)
		}
		if q2.Rounds() != rounds {
			t.Fatalf("rounds=%d: Rounds() = %d", rounds, q2.Rounds())
		}
		packed2, norm2 := q2.Quantize(vec)
		if norm != norm2 {
			t.Fatalf("rounds=%d: norm mismatch: %v != %v", rounds, norm, norm2)
		}
		for i := range packed {
			if packed[i] != packed2[i] {
				t.Fatalf("rounds=%d: packed byte %d mismatch", rounds, i)
			}
		}
	}
}

// TestLegacyRotationDecodesUnchanged proves that a real pre-Phase-2 25-byte
// payload (rotationKindHadamard, one implicit round, no round-count byte)
// still decodes to the exact numerics it was written with, and that those
// numerics match NewHadamardRoundsWithSeed(dim, bits, 1, seed), the
// documented way to reproduce the legacy transform going forward.
func TestLegacyRotationDecodesUnchanged(t *testing.T) {
	legacy, err := hex.DecodeString(legacyCompactMSEGoldenHex)
	if err != nil {
		t.Fatalf("decode legacy hex: %v", err)
	}
	if len(legacy) != serializeHeaderSizeV2 {
		t.Fatalf("legacy golden length = %d want %d", len(legacy), serializeHeaderSizeV2)
	}

	q1, err := UnmarshalQuantizer(legacy)
	if err != nil {
		t.Fatalf("unmarshal legacy: %v", err)
	}
	if q1.RotationKind() != "hadamard" {
		t.Fatalf("legacy RotationKind() = %q want hadamard", q1.RotationKind())
	}
	if q1.Rounds() != 1 {
		t.Fatalf("legacy Rounds() = %d want 1", q1.Rounds())
	}

	q2 := NewHadamardRoundsWithSeed(4, 2, 1, 42)

	vec := randomUnitVector(4, newTestRNG())
	p1, n1 := q1.Quantize(vec)
	p2, n2 := q2.Quantize(vec)
	if n1 != n2 {
		t.Fatalf("norm mismatch: %v != %v", n1, n2)
	}
	for i := range p1 {
		if p1[i] != p2[i] {
			t.Fatalf("packed byte %d mismatch: %d != %d", i, p1[i], p2[i])
		}
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
