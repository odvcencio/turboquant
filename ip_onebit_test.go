package turboquant

import (
	"math"
	"math/rand"
	"testing"
)

// TestIPBitWidth1PureQJL pins the paper's b=1 inner-product quantizer: no MSE
// stage, the residual is the unit-normalized input itself (ResNorm = 1), and
// the estimate is the pure QJL sign estimator rescaled by the input norm.
func TestIPBitWidth1PureQJL(t *testing.T) {
	const dim = 128
	q := NewIPHadamardWithSeed(dim, 1, 7)
	if got := q.BitWidth(); got != 1 {
		t.Fatalf("BitWidth() = %d want 1", got)
	}
	rng := rand.New(rand.NewSource(2))
	x := scaledVector(randomUnitVector(dim, rng), 2)
	y := randomUnitVector(dim, rng)

	qx := q.Quantize(x)
	if len(qx.MSE) != 0 {
		t.Fatalf("b=1 quantization must have no MSE payload, got %d bytes", len(qx.MSE))
	}
	if math.Abs(float64(qx.ResNorm-1)) > 1e-5 {
		t.Fatalf("b=1 residual norm = %v want 1 (residual is the unit input)", qx.ResNorm)
	}
	if math.Abs(float64(qx.Norm-2)) > 1e-4 {
		t.Fatalf("b=1 stored norm = %v want 2", qx.Norm)
	}

	est := q.InnerProduct(qx, y)
	pq := q.PrepareQuery(y)
	estP := q.InnerProductPrepared(qx, pq)
	if math.Abs(float64(estP-est)) > 1e-3*(1+math.Abs(float64(est))) {
		t.Fatalf("prepared estimate %v differs from direct estimate %v", estP, est)
	}
	batch := make([]float32, 1)
	q.InnerProductPreparedBatchTo(batch, qx, []PreparedQuery{pq})
	if math.Abs(float64(batch[0]-est)) > 1e-3*(1+math.Abs(float64(est))) {
		t.Fatalf("batch estimate %v differs from direct estimate %v", batch[0], est)
	}
}

// TestIPBitWidth1UnbiasedWithinPaperBound checks unbiasedness and the paper's
// b=1 distortion table entry D_prod ≈ 1.57/d (per unit input and query norm).
func TestIPBitWidth1UnbiasedWithinPaperBound(t *testing.T) {
	const dim, trials = 128, 400
	rng := rand.New(rand.NewSource(3))
	xUnit := randomUnitVector(dim, rng)
	perp := orthogonalUnit(rng, dim, xUnit)
	x := scaledVector(xUnit, 2)
	y := combineUnits(0.6, xUnit, 0.8, perp)

	var want float64
	for i := range x {
		want += float64(x[i]) * float64(y[i])
	}

	var sumErr, sumSq float64
	for trial := range trials {
		q := NewIPHadamardWithSeed(dim, 1, int64(5000+trial))
		est := float64(q.InnerProduct(q.Quantize(x), y))
		e := est - want
		sumErr += e
		sumSq += e * e
	}
	mean := sumErr / trials
	variance := sumSq/trials - mean*mean
	ci := 4 * math.Sqrt(variance/trials)
	if math.Abs(mean) > ci+1e-3 {
		t.Fatalf("b=1 estimator biased: mean err %.5f (CI ±%.5f), true %.4f", mean, ci, want)
	}
	// ||x||^2 = 4, ||y||^2 = 1: paper bound 1.57/d scales to 4*1.57/d.
	bound := 4 * 1.5708 / float64(dim)
	if variance > bound*1.5 {
		t.Fatalf("b=1 distortion %.6f exceeds 1.5x paper bound %.6f", variance, bound)
	}
}

// TestIPBitWidth1WireAndSerializeRoundtrip covers the b=1 wire record and the
// compact quantizer serialization.
func TestIPBitWidth1WireAndSerializeRoundtrip(t *testing.T) {
	const dim = 48
	q := NewIPHadamardWithSeed(dim, 1, 11)
	rng := rand.New(rand.NewSource(4))
	x := scaledVector(randomUnitVector(dim, rng), 3)
	y := randomUnitVector(dim, rng)

	qx := q.Quantize(x)
	want := q.InnerProduct(qx, y)

	wireDim, wireBits, decoded, err := DecodeIP(EncodeIP(dim, 1, qx))
	if err != nil {
		t.Fatal(err)
	}
	if wireDim != dim || wireBits != 1 {
		t.Fatalf("wire roundtrip shape (%d,%d) want (%d,1)", wireDim, wireBits, dim)
	}
	if got := q.InnerProduct(decoded, y); math.Abs(float64(got-want)) > 1e-6*(1+math.Abs(float64(want))) {
		t.Fatalf("wire roundtrip estimate %v want %v", got, want)
	}

	blob, err := MarshalIPQuantizer(q)
	if err != nil {
		t.Fatal(err)
	}
	restored, err := UnmarshalIPQuantizer(blob)
	if err != nil {
		t.Fatal(err)
	}
	qx2 := restored.Quantize(x)
	if got := restored.InnerProduct(qx2, y); math.Abs(float64(got-want)) > 1e-6*(1+math.Abs(float64(want))) {
		t.Fatalf("serialize roundtrip estimate %v want %v", got, want)
	}
}

// TestIPBitWidth1ScoreUpperBoundPrunable pins the m3 fix: at bit width 1 the
// MSE upper component is exactly 0, so the pure-QJL bound is valid and
// prunable, and it must never undercut the true prepared score for any norm.
func TestIPBitWidth1ScoreUpperBoundPrunable(t *testing.T) {
	const dim = 96
	q := NewIPHadamardWithSeed(dim, 1, 5)
	rng := rand.New(rand.NewSource(8))
	y := randomUnitVector(dim, rng)
	pq := q.PrepareQuery(y)

	if _, prunable := pq.ScoreUpperBound(1, 1); !prunable {
		t.Fatalf("b=1 ScoreUpperBound reports prunable=false; the pure-QJL bound is valid")
	}
	for trial := 0; trial < 200; trial++ {
		scale := float32(0.01 + rng.Float64()*9.99)
		qx := q.Quantize(scaledVector(randomUnitVector(dim, rng), scale))
		score := q.InnerProductPrepared(qx, pq)
		bound, prunable := pq.ScoreUpperBound(qx.Norm, qx.ResNorm)
		if !prunable {
			t.Fatalf("trial %d: prunable=false", trial)
		}
		slack := float32(1e-4) * (absF32(bound) + 1)
		if score > bound+slack {
			t.Fatalf("trial %d: score %.6f exceeds bound %.6f (norm=%.4f)", trial, score, bound, qx.Norm)
		}
	}
}

// TestKVCachePageKeyBits1 exercises a KV page whose keys use the pure QJL
// quantizer: page scores must match the direct quantizer scores.
func TestKVCachePageKeyBits1(t *testing.T) {
	const dim = 64
	rng := rand.New(rand.NewSource(6))
	page := NewKVCachePageWithSeed(dim, 1, dim, 8, 4, 15)
	keys := make([][]float32, 3)
	for i := range keys {
		keys[i] = scaledVector(randomUnitVector(dim, rng), float32(1+i))
		page.Append(keys[i], randomUnitVector(dim, rng))
	}
	y := randomUnitVector(dim, rng)
	pq := page.PrepareQuery(y)

	idx := make([]uint32, 3)
	scores := make([]float32, 3)
	page.TopKPreparedTo(idx, scores, pq)

	kq := page.KeyQuantizer()
	for rank, slot := range idx {
		want := kq.InnerProductPrepared(kq.Quantize(keys[slot]), pq)
		if math.Abs(float64(scores[rank]-want)) > 1e-5*(1+math.Abs(float64(want))) {
			t.Fatalf("rank %d (slot %d): page score %v want %v", rank, slot, scores[rank], want)
		}
	}
}
