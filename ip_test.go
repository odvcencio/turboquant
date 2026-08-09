package turboquant

import (
	"math"
	"math/rand"
	"testing"
)

func TestIPQuantizerUnbiased(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	dim := 384
	rng := rand.New(rand.NewSource(99))
	x := randomUnitVector(dim, rng)
	y := randomUnitVector(dim, rng)

	var trueIP float64
	for i := range x {
		trueIP += float64(x[i]) * float64(y[i])
	}

	// Average over many independently-seeded quantizers
	var sumEstimate float64
	trials := 200
	for trial := 0; trial < trials; trial++ {
		q := NewIPDenseWithSeed(dim, 3, int64(trial+100))
		qx := q.Quantize(x)
		estimate := q.InnerProduct(qx, y)
		sumEstimate += float64(estimate)
	}
	avgEstimate := sumEstimate / float64(trials)
	if math.Abs(avgEstimate-trueIP) > 0.05 {
		t.Errorf("IP avg estimate %.4f vs true IP %.4f", avgEstimate, trueIP)
	}
}

func TestIPQuantizerDistortion(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	dim := 384
	rng := rand.New(rand.NewSource(99))
	x := randomUnitVector(dim, rng)
	y := randomUnitVector(dim, rng)

	var trueIP float64
	for i := range x {
		trueIP += float64(x[i]) * float64(y[i])
	}

	var sumSqErr float64
	trials := 200
	for trial := 0; trial < trials; trial++ {
		q := NewIPDenseWithSeed(dim, 2, int64(trial+200))
		qx := q.Quantize(x)
		estimate := q.InnerProduct(qx, y)
		err := float64(estimate) - trueIP
		sumSqErr += err * err
	}
	avgDistortion := sumSqErr / float64(trials)
	bound := 0.56 / float64(dim) // paper value for b=2
	if avgDistortion > bound*2.0 {
		t.Errorf("IP distortion %.6f exceeds 2x bound %.6f", avgDistortion, bound)
	}
}

func TestIPQuantizerDimAndBitWidth(t *testing.T) {
	q := NewIPWithSeed(256, 3, 42)
	if q.Dim() != 256 {
		t.Errorf("Dim() = %d want 256", q.Dim())
	}
	if q.BitWidth() != 3 {
		t.Errorf("BitWidth() = %d want 3", q.BitWidth())
	}
	if q.Seed() != 42 {
		t.Errorf("Seed() = %d want 42", q.Seed())
	}
}

func TestIPQuantizerPreparedQuery(t *testing.T) {
	dim := 128
	q := NewIPWithSeed(dim, 3, 42)
	rng := rand.New(rand.NewSource(99))
	x := randomUnitVector(dim, rng)
	y := randomUnitVector(dim, rng)

	qx := q.Quantize(x)
	direct := q.InnerProduct(qx, y)
	pq := q.PrepareQuery(y)
	prepared := q.InnerProductPrepared(qx, pq)

	if math.Abs(float64(direct-prepared)) > 1e-5 {
		t.Errorf("prepared %.6f != direct %.6f", prepared, direct)
	}
}

func TestNewIPHadamardRoundsWithSeed(t *testing.T) {
	for _, rounds := range []int{1, 2, 3, 4} {
		q := NewIPHadamardRoundsWithSeed(64, 3, rounds, 42)
		if q.mse.Rounds() != rounds {
			t.Errorf("rounds=%d: mse.Rounds() = %d", rounds, q.mse.Rounds())
		}
		rng := rand.New(rand.NewSource(11))
		x := randomUnitVector(64, rng)
		y := randomUnitVector(64, rng)
		qx := q.Quantize(x)
		direct := q.InnerProduct(qx, y)
		pq := q.PrepareQuery(y)
		prepared := q.InnerProductPrepared(qx, pq)
		if math.Abs(float64(direct-prepared)) > 1e-5 {
			t.Errorf("rounds=%d: prepared %.6f != direct %.6f", rounds, prepared, direct)
		}
	}
}

func TestIPHadamardPreparedQuery(t *testing.T) {
	dim := 128
	q := NewIPHadamardWithSeed(dim, 3, 42)
	if q.RotationKind() != "hadamard-multi" {
		t.Fatalf("RotationKind() = %q want hadamard-multi", q.RotationKind())
	}
	rng := rand.New(rand.NewSource(77))
	x := randomUnitVector(dim, rng)
	y := randomUnitVector(dim, rng)
	qx := q.Quantize(x)
	direct := q.InnerProduct(qx, y)
	pq := q.PrepareQuery(y)
	prepared := q.InnerProductPrepared(qx, pq)
	if math.Abs(float64(direct-prepared)) > 1e-5 {
		t.Fatalf("prepared %.6f != direct %.6f", prepared, direct)
	}
}

func TestIPPreparedQueryNonByteAlignedDim(t *testing.T) {
	dim := 130
	q := NewIPHadamardWithSeed(dim, 3, 42)
	rng := rand.New(rand.NewSource(78))
	x := randomUnitVector(dim, rng)
	y := randomUnitVector(dim, rng)
	qx := q.Quantize(x)
	direct := q.InnerProduct(qx, y)
	pq := q.PrepareQuery(y)
	prepared := q.InnerProductPrepared(qx, pq)
	if math.Abs(float64(direct-prepared)) > 1e-5 {
		t.Fatalf("prepared %.6f != direct %.6f", prepared, direct)
	}
}

func TestIPPreparedQueryZeroAllocs(t *testing.T) {
	skipAllocsUnderRace(t)
	q := NewIPHadamardWithSeed(128, 3, 42)
	rng := rand.New(rand.NewSource(55))
	x := randomUnitVector(128, rng)
	y := randomUnitVector(128, rng)
	qx := q.Quantize(x)
	pq := q.PrepareQuery(y)

	// Warm sync.Pool-backed scratch buffers before measuring.
	_ = q.InnerProductPrepared(qx, pq)

	allocs := testing.AllocsPerRun(100, func() {
		_ = q.InnerProductPrepared(qx, pq)
	})
	if allocs != 0 {
		t.Fatalf("InnerProductPrepared allocs = %.2f want 0", allocs)
	}
}

func TestIPPreparedQueryTrustedMatchesValidated(t *testing.T) {
	q := NewIPHadamardWithSeed(128, 3, 42)
	rng := rand.New(rand.NewSource(66))
	x := randomUnitVector(128, rng)
	y := randomUnitVector(128, rng)
	qx := q.Quantize(x)
	pq := q.PrepareQuery(y)
	got := q.InnerProductPreparedTrusted(qx, pq)
	want := q.InnerProductPrepared(qx, pq)
	if math.Abs(float64(got-want)) > 1e-6 {
		t.Fatalf("trusted %.6f != validated %.6f", got, want)
	}
}

func TestPrepareQueryLegacyBufferStillWorks(t *testing.T) {
	q := NewIPHadamardWithSeed(128, 3, 42)
	rng := rand.New(rand.NewSource(67))
	x := randomUnitVector(128, rng)
	y := randomUnitVector(128, rng)
	qx := q.Quantize(x)
	pq := AllocPreparedQuery(q.Dim())
	q.PrepareQueryTo(&pq, y)
	got := q.InnerProductPreparedTrusted(qx, pq)
	want := q.InnerProduct(qx, y)
	if math.Abs(float64(got-want)) > 1e-5 {
		t.Fatalf("legacy prepared %.6f != direct %.6f", got, want)
	}
}

func TestPrepareQueryToZeroAllocs(t *testing.T) {
	skipAllocsUnderRace(t)
	q := NewIPHadamardWithSeed(128, 3, 42)
	rng := rand.New(rand.NewSource(88))
	y := randomUnitVector(128, rng)
	pq := q.AllocPreparedQuery()

	q.PrepareQueryTo(&pq, y)

	allocs := testing.AllocsPerRun(100, func() {
		q.PrepareQueryTo(&pq, y)
	})
	if allocs != 0 {
		t.Fatalf("PrepareQueryTo allocs = %.2f want 0", allocs)
	}
}

func TestPrepareQueryToTrustedMatchesValidated(t *testing.T) {
	q := NewIPHadamardWithSeed(128, 3, 42)
	rng := rand.New(rand.NewSource(91))
	y := randomUnitVector(128, rng)
	got := q.AllocPreparedQuery()
	want := q.AllocPreparedQuery()
	q.PrepareQueryToTrusted(&got, y)
	q.PrepareQueryTo(&want, y)
	if len(got.signLUT) != len(want.signLUT) || len(got.rotY) != len(want.rotY) || len(got.mseLUT) != len(want.mseLUT) {
		t.Fatal("prepared query shapes differ")
	}
	for i := range want.signLUT {
		if math.Abs(float64(got.signLUT[i]-want.signLUT[i])) > 1e-6 {
			t.Fatalf("signLUT[%d] = %.6f want %.6f", i, got.signLUT[i], want.signLUT[i])
		}
	}
	for i := range want.rotY {
		if math.Abs(float64(got.rotY[i]-want.rotY[i])) > 1e-6 {
			t.Fatalf("rotY[%d] = %.6f want %.6f", i, got.rotY[i], want.rotY[i])
		}
	}
	for i := range want.mseLUT {
		if math.Abs(float64(got.mseLUT[i]-want.mseLUT[i])) > 1e-6 {
			t.Fatalf("mseLUT[%d] = %.6f want %.6f", i, got.mseLUT[i], want.mseLUT[i])
		}
	}
}

func TestIPPreparedBatchTrustedMatchesSingleScores(t *testing.T) {
	q := NewIPHadamardWithSeed(128, 3, 42)
	rng := rand.New(rand.NewSource(89))
	x := randomUnitVector(128, rng)
	queries := [][]float32{
		randomUnitVector(128, rng),
		randomUnitVector(128, rng),
		randomUnitVector(128, rng),
		randomUnitVector(128, rng),
	}
	pqs := make([]PreparedQuery, len(queries))
	for i := range queries {
		pqs[i] = q.PrepareQuery(queries[i])
	}
	qx := q.Quantize(x)
	got := make([]float32, len(pqs))
	q.InnerProductPreparedBatchToTrusted(got, qx, pqs)
	for i := range pqs {
		want := q.InnerProductPreparedTrusted(qx, pqs[i])
		if math.Abs(float64(got[i]-want)) > 1e-6 {
			t.Fatalf("batch score %d = %.6f want %.6f", i, got[i], want)
		}
	}
}

func TestIPPreparedBatchTrustedZeroAllocs(t *testing.T) {
	skipAllocsUnderRace(t)
	q := NewIPHadamardWithSeed(128, 3, 42)
	rng := rand.New(rand.NewSource(90))
	x := randomUnitVector(128, rng)
	queries := [][]float32{
		randomUnitVector(128, rng),
		randomUnitVector(128, rng),
		randomUnitVector(128, rng),
		randomUnitVector(128, rng),
	}
	pqs := make([]PreparedQuery, len(queries))
	for i := range queries {
		pqs[i] = q.PrepareQuery(queries[i])
	}
	qx := q.Quantize(x)
	dst := make([]float32, len(pqs))
	q.InnerProductPreparedBatchToTrusted(dst, qx, pqs)
	allocs := testing.AllocsPerRun(100, func() {
		q.InnerProductPreparedBatchToTrusted(dst, qx, pqs)
	})
	if allocs != 0 {
		t.Fatalf("InnerProductPreparedBatchToTrusted allocs = %.2f want 0", allocs)
	}
}

func TestIPQuantizeToZeroAllocs(t *testing.T) {
	skipAllocsUnderRace(t)
	q := NewIPHadamardWithSeed(128, 3, 42)
	rng := rand.New(rand.NewSource(55))
	x := randomUnitVector(128, rng)
	qx := AllocIPQuantized(q.Dim(), q.BitWidth())

	q.QuantizeTo(&qx, x)

	allocs := testing.AllocsPerRun(100, func() {
		q.QuantizeTo(&qx, x)
	})
	if allocs != 0 {
		t.Fatalf("QuantizeTo allocs = %.2f want 0", allocs)
	}
}

func TestIPQuantizeToRejectsWrongShape(t *testing.T) {
	q := NewIPHadamardWithSeed(8, 3, 42)
	qx := IPQuantized{
		MSE:   make([]byte, 1),
		Signs: make([]byte, 1),
	}
	expectPanic(t, func() {
		q.QuantizeTo(&qx, make([]float32, 8))
	})
}

func TestIPHadamardParityToDense(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	dense := NewIPDenseWithSeed(384, 3, 42)
	hadamard := NewIPHadamardWithSeed(384, 3, 42)
	rng := rand.New(rand.NewSource(333))

	var denseErr, hadamardErr float64
	trials := 250
	for trial := 0; trial < trials; trial++ {
		x := randomUnitVector(384, rng)
		y := randomUnitVector(384, rng)

		var trueIP float64
		for i := range x {
			trueIP += float64(x[i]) * float64(y[i])
		}

		qDense := dense.Quantize(x)
		qHad := hadamard.Quantize(x)

		dDense := float64(dense.InnerProduct(qDense, y)) - trueIP
		dHad := float64(hadamard.InnerProduct(qHad, y)) - trueIP
		denseErr += dDense * dDense
		hadamardErr += dHad * dHad
	}

	ratio := hadamardErr / denseErr
	if ratio > 1.50 {
		t.Fatalf("hadamard/dense IP distortion ratio %.3f exceeds 1.50", ratio)
	}
}

func TestScoreUpperBoundIsProvenUpperBound(t *testing.T) {
	dim := 128
	rng := rand.New(rand.NewSource(7))
	// IP bit widths whose MSE stage (bitWidth-1) lands on an MSE-LUT width
	// (1/2/4/8): 2->1, 3->2, 5->4. MSE stage 8 would need IP bit width 9, which
	// validateIPBitWidth rejects (IP bit width is 2-8), so it is unreachable via
	// IPQuantizer. For all reachable LUT widths prunable must be true and the
	// bound must never undercut the true prepared score.
	for _, ipBitWidth := range []int{2, 3, 5} {
		mseStage := ipBitWidth - 1
		q := NewIPHadamardWithSeed(dim, ipBitWidth, int64(1000+ipBitWidth))
		y := randomUnitVector(dim, rng)
		pq := q.PrepareQuery(y)

		// Vary residual norms by scaling corpus vectors over a wide range.
		for trial := 0; trial < 64; trial++ {
			scale := float32(0.01 + rng.Float64()*9.99)
			x := randomUnitVector(dim, rng)
			for i := range x {
				x[i] *= scale
			}
			qx := q.Quantize(x)

			score := q.InnerProductPrepared(qx, pq)
			bound, prunable := pq.ScoreUpperBound(qx.Norm, qx.ResNorm)
			if !prunable {
				t.Fatalf("ipBitWidth=%d (MSE stage %d): prunable=false, want true", ipBitWidth, mseStage)
			}
			// Allow a tiny float32 slack: bound and score accumulate the same
			// terms in different order, so equality at the aligned extreme can
			// differ by ~ULP. A violation beyond that falsifies the proof.
			slack := float32(1e-4) * (absF32(bound) + 1)
			if score > bound+slack {
				t.Fatalf("ipBitWidth=%d trial=%d: score %.6f exceeds bound %.6f (resNorm=%.4f)",
					ipBitWidth, trial, score, bound, qx.ResNorm)
			}
		}
	}
}

func TestScoreUpperBoundNonLUTWidthNotPrunable(t *testing.T) {
	dim := 128
	rng := rand.New(rand.NewSource(11))
	// IP bit widths whose MSE stage (bitWidth-1) is 3/5/6/7 have no MSE LUT:
	// 4->3, 6->5, 7->6, 8->7. ScoreUpperBound must report prunable=false.
	for _, ipBitWidth := range []int{4, 6, 7, 8} {
		mseStage := ipBitWidth - 1
		q := NewIPHadamardWithSeed(dim, ipBitWidth, int64(2000+ipBitWidth))
		y := randomUnitVector(dim, rng)
		pq := q.PrepareQuery(y)
		_, prunable := pq.ScoreUpperBound(1.0, 1.0)
		if prunable {
			t.Fatalf("ipBitWidth=%d (MSE stage %d): prunable=true, want false (no MSE LUT)", ipBitWidth, mseStage)
		}
	}
}

func TestScoreUpperBoundMemoizedStable(t *testing.T) {
	dim := 96
	rng := rand.New(rand.NewSource(13))
	q := NewIPHadamardWithSeed(dim, 3, 42)
	y := randomUnitVector(dim, rng)
	pq := q.PrepareQuery(y)

	for _, resNorm := range []float32{0, 0.5, 1.0, 7.25} {
		b0, p0 := pq.ScoreUpperBound(1.0, resNorm)
		b1, p1 := pq.ScoreUpperBound(1.0, resNorm)
		if b0 != b1 || p0 != p1 {
			t.Fatalf("resNorm=%.4f: repeated ScoreUpperBound differ: (%.6f,%v) vs (%.6f,%v)",
				resNorm, b0, p0, b1, p1)
		}
	}
	// Memoization must produce the same A/B as a freshly prepared, never-queried
	// copy of the same query.
	pq2 := q.PrepareQuery(y)
	want, _ := pq2.ScoreUpperBound(1.0, 3.0)
	got, _ := pq.ScoreUpperBound(1.0, 3.0)
	if got != want {
		t.Fatalf("memoized bound %.6f != fresh bound %.6f", got, want)
	}
}

func absF32(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
