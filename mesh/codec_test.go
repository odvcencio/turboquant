package mesh

import (
	"encoding/json"
	"math"
	"testing"
)

func TestEncodeDecodeRoundTrip(t *testing.T) {
	// N = 3 vertices.
	pos := []float32{
		0, 0, 0,
		1, 2, 3,
		-5, 0.5, 10,
	}
	c := float32(0.57735027)
	nrm := []float32{
		1, 0, 0,
		0, 1, 0,
		c, c, c,
	}
	normalize3(nrm)
	joints := []uint32{
		0, 1, 2, 3,
		4, 250, 12, 9,
		5, 6, 7, 8,
	}
	weights := []float32{
		0.1, 0.2, 0.3, 0.4,
		0.7, 0.3, 0, 0,
		0.25, 0.25, 0.25, 0.25,
	}

	q := Encode(pos, nrm, joints, weights, Options{})
	gotPos, gotNrm, gotJoints, gotWeights := Decode(q)

	// Positions within Task-1 half-step tolerance.
	for i := range pos {
		axis := i % 3
		tol := float64(q.Spec.Position.Scale[axis])/2 + 1e-6
		if d := math.Abs(float64(gotPos[i] - pos[i])); d > tol {
			t.Errorf("pos %d: got %v want %v, diff %v > tol %v", i, gotPos[i], pos[i], d, tol)
		}
	}

	// Normals within 1 degree angular error.
	for i := 0; i+2 < len(nrm); i += 3 {
		dot := float64(nrm[i])*float64(gotNrm[i]) +
			float64(nrm[i+1])*float64(gotNrm[i+1]) +
			float64(nrm[i+2])*float64(gotNrm[i+2])
		if dot > 1 {
			dot = 1
		}
		if dot < -1 {
			dot = -1
		}
		if deg := math.Acos(dot) * 180 / math.Pi; deg >= 1.0 {
			t.Errorf("normal %d: angular error %v deg >= 1.0", i/3, deg)
		}
	}

	// Joints exact.
	for i := range joints {
		if gotJoints[i] != joints[i] {
			t.Errorf("joint %d: got %d want %d", i, gotJoints[i], joints[i])
		}
	}

	// Weights within tolerance and sum to 1.
	const wtol = 1.0/255 + 1e-4
	for i := range weights {
		if d := math.Abs(float64(gotWeights[i] - weights[i])); d > wtol {
			t.Errorf("weight %d: got %v want %v, diff %v > tol %v", i, gotWeights[i], weights[i], d, wtol)
		}
	}
	for v := 0; v < 3; v++ {
		var sum float64
		for k := 0; k < 4; k++ {
			sum += float64(gotWeights[v*4+k])
		}
		if math.Abs(sum-1) > 1e-5 {
			t.Errorf("vertex %d weights sum to %v, want 1", v, sum)
		}
	}

	// Spec sanity.
	if q.Spec.VertexCount != 3 {
		t.Errorf("VertexCount=%d, want 3", q.Spec.VertexCount)
	}
	if q.Spec.WeightBits != 8 {
		t.Errorf("WeightBits=%d, want 8", q.Spec.WeightBits)
	}
}

func TestEncodeNoNormals(t *testing.T) {
	pos := []float32{0, 0, 0, 1, 1, 1}
	q := Encode(pos, nil, nil, nil, Options{})
	if q.Spec.NormalBits != 0 {
		t.Errorf("NormalBits=%d, want 0 for nil normals", q.Spec.NormalBits)
	}
	if q.Normals != nil {
		t.Errorf("Normals=%v, want nil", q.Normals)
	}
	_, gotNrm, _, _ := Decode(q)
	if gotNrm != nil {
		t.Errorf("decoded normals=%v, want nil", gotNrm)
	}
}

func TestDefaultsApplied(t *testing.T) {
	pos := []float32{0, 0, 0, 1, 1, 1}
	nrm := []float32{1, 0, 0, 0, 1, 0}
	joints := []uint32{0, 1, 2, 3, 4, 5, 6, 7}
	weights := []float32{1, 0, 0, 0, 0.5, 0.5, 0, 0}
	q := Encode(pos, nrm, joints, weights, Options{})
	if q.Spec.Position.Bits != 16 {
		t.Errorf("default PositionBits=%d, want 16", q.Spec.Position.Bits)
	}
	if q.Spec.NormalBits != 16 {
		t.Errorf("default NormalBits=%d, want 16", q.Spec.NormalBits)
	}
	if q.Spec.JointBits != 8 {
		t.Errorf("default JointBits=%d, want 8", q.Spec.JointBits)
	}
}

func TestDequantSpecJSONPortable(t *testing.T) {
	spec := DequantSpec{
		VertexCount: 7,
		Position: PositionParams{
			Bits:  16,
			Min:   [3]float32{-1, 0.5, 100},
			Scale: [3]float32{0.001, 0.25, 3.5},
		},
		NormalBits: 16,
		JointBits:  8,
		WeightBits: 8,
	}
	b, err := json.Marshal(spec)
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	var rt DequantSpec
	if err := json.Unmarshal(b, &rt); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	// DequantSpec must be a comparable value struct (no slices) so == works.
	if rt != spec {
		t.Errorf("round-trip mismatch:\n got %+v\nwant %+v", rt, spec)
	}
}
