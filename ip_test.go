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

func TestIPHadamardPreparedQuery(t *testing.T) {
	dim := 128
	q := NewIPHadamardWithSeed(dim, 3, 42)
	if q.RotationKind() != "hadamard" {
		t.Fatalf("RotationKind() = %q want hadamard", q.RotationKind())
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
