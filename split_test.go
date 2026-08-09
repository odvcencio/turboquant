package turboquant

import (
	"math"
	"math/rand"
	"testing"
)

func seqInts(n int) []int {
	out := make([]int, n)
	for i := range out {
		out[i] = i
	}
	return out
}

func gatherChannels(v []float32, idx []int) []float32 {
	out := make([]float32, len(idx))
	for i, ch := range idx {
		out[i] = v[ch]
	}
	return out
}

func complementChannels(dim int, idx []int) []int {
	member := make([]bool, dim)
	for _, ch := range idx {
		member[ch] = true
	}
	out := make([]int, 0, dim-len(idx))
	for ch := range dim {
		if !member[ch] {
			out = append(out, ch)
		}
	}
	return out
}

// TestSplitSpecEffectiveBits pins the paper's fractional configurations:
// 32 outlier channels at 4 bits + 96 regular channels at 2 bits over
// dimension 128 is the 2.5-bit setup. (The paper prints "32x3+96x2 = 2.5",
// which mis-multiplies; the honest arithmetic for 2.5 needs 4-bit outliers.)
func TestSplitSpecEffectiveBits(t *testing.T) {
	spec := SplitSpec{Outliers: seqInts(32), OutlierBits: 4, RegularBits: 2}
	if got := spec.EffectiveBits(128); math.Abs(got-2.5) > 1e-9 {
		t.Fatalf("EffectiveBits = %v want 2.5", got)
	}
	spec35 := SplitSpec{Outliers: seqInts(64), OutlierBits: 4, RegularBits: 3}
	if got := spec35.EffectiveBits(128); math.Abs(got-3.5) > 1e-9 {
		t.Fatalf("EffectiveBits = %v want 3.5", got)
	}
	uniform := SplitSpec{RegularBits: 3}
	if got := uniform.EffectiveBits(128); math.Abs(got-3.0) > 1e-9 {
		t.Fatalf("uniform EffectiveBits = %v want 3", got)
	}
}

// TestSplitIPMatchesManualComposition pins the split estimator to the sum of
// the two independent sub-quantizer estimates over the gathered channel sets.
func TestSplitIPMatchesManualComposition(t *testing.T) {
	const dim = 32
	spec := SplitSpec{Outliers: []int{1, 5, 7, 20, 21, 30}, OutlierBits: 4, RegularBits: 2}
	q := NewSplitIPWithSeed(dim, spec, 9)
	rng := rand.New(rand.NewSource(3))
	x := scaledVector(randomUnitVector(dim, rng), 1.8)
	y := randomUnitVector(dim, rng)

	qx := q.Quantize(x)
	est := q.InnerProduct(qx, y)

	regIdx := complementChannels(dim, spec.Outliers)
	outQ, regQ := q.OutlierQuantizer(), q.RegularQuantizer()
	wantOut := outQ.InnerProduct(outQ.Quantize(gatherChannels(x, spec.Outliers)), gatherChannels(y, spec.Outliers))
	wantReg := regQ.InnerProduct(regQ.Quantize(gatherChannels(x, regIdx)), gatherChannels(y, regIdx))
	want := wantOut + wantReg
	if math.Abs(float64(est-want)) > 1e-5*(1+math.Abs(float64(want))) {
		t.Fatalf("split estimate %v differs from manual composition %v", est, want)
	}

	spq := q.PrepareQuery(y)
	if got := q.InnerProductPrepared(qx, spq); math.Abs(float64(got-est)) > 1e-3*(1+math.Abs(float64(est))) {
		t.Fatalf("prepared split estimate %v differs from direct %v", got, est)
	}

	// Dequantize must scatter sub-reconstructions back to their channels.
	recon := q.Dequantize(qx)
	outRecon := outQ.Dequantize(qx.Out)
	for i, ch := range spec.Outliers {
		if math.Abs(float64(recon[ch]-outRecon[i])) > 1e-6 {
			t.Fatalf("outlier channel %d reconstruction %v differs from sub-quantizer %v", ch, recon[ch], outRecon[i])
		}
	}
}

// TestSplitIPUnbiased checks the composed estimator stays unbiased for
// non-unit inputs: each part is unbiased and they are independent.
func TestSplitIPUnbiased(t *testing.T) {
	const dim, trials = 128, 300
	spec := SplitSpec{Outliers: seqInts(32), OutlierBits: 4, RegularBits: 2}
	rng := rand.New(rand.NewSource(8))
	xUnit := randomUnitVector(dim, rng)
	perp := orthogonalUnit(rng, dim, xUnit)
	x := scaledVector(xUnit, 2.5)
	y := combineUnits(0.7, xUnit, 0.5, perp)

	var want float64
	for i := range x {
		want += float64(x[i]) * float64(y[i])
	}
	var sumErr, sumSq float64
	for trial := range trials {
		q := NewSplitIPWithSeed(dim, spec, int64(9000+trial))
		e := float64(q.InnerProduct(q.Quantize(x), y)) - want
		sumErr += e
		sumSq += e * e
	}
	mean := sumErr / trials
	variance := sumSq/trials - mean*mean
	ci := 4 * math.Sqrt(variance/trials)
	if math.Abs(mean) > ci+1e-3 {
		t.Fatalf("split IP estimator biased: mean err %.5f (CI ±%.5f), true %.4f", mean, ci, want)
	}
}

// TestSelectOutlierChannels ranks channels by mean absolute magnitude.
func TestSelectOutlierChannels(t *testing.T) {
	const dim = 24
	rng := rand.New(rand.NewSource(5))
	samples := make([][]float32, 40)
	for i := range samples {
		v := randomUnitVector(dim, rng)
		v[3] *= 12
		v[17] *= 9
		samples[i] = v
	}
	got := SelectOutlierChannels(samples, 2)
	if len(got) != 2 || got[0] != 3 || got[1] != 17 {
		t.Fatalf("SelectOutlierChannels = %v want [3 17]", got)
	}
}

// TestSplitMSEKeepsChannelSeparation: energy on outlier channels must stay on
// outlier channels through quantize/dequantize (the two instances are
// independent per the paper).
func TestSplitMSEKeepsChannelSeparation(t *testing.T) {
	const dim = 32
	spec := SplitSpec{Outliers: []int{0, 1, 2, 3, 4, 5, 6, 7}, OutlierBits: 4, RegularBits: 2}
	q := NewSplitMSEWithSeed(dim, spec, 4)
	x := make([]float32, dim)
	rng := rand.New(rand.NewSource(2))
	for _, ch := range spec.Outliers {
		x[ch] = float32(rng.NormFloat64())
	}
	recon := q.Dequantize(q.Quantize(x))
	for ch := 8; ch < dim; ch++ {
		if recon[ch] != 0 {
			t.Fatalf("regular channel %d leaked energy %v from outlier-only input", ch, recon[ch])
		}
	}
	var errSq, normSq float64
	for _, ch := range spec.Outliers {
		d := float64(recon[ch] - x[ch])
		errSq += d * d
		normSq += float64(x[ch]) * float64(x[ch])
	}
	// The 8 outlier channels form their own 4-bit sub-quantizer; relative
	// error must be far below 1 (loose sanity, small-dim codebook).
	if errSq/normSq > 0.35 {
		t.Fatalf("outlier-side relative reconstruction error %.4f too large", errSq/normSq)
	}
}

// TestSplitKVPageMatchesManualScores pins split KV page key scoring to the
// split quantizer, with non-unit keys.
func TestSplitKVPageMatchesManualScores(t *testing.T) {
	const dim = 32
	keySpec := SplitSpec{Outliers: seqInts(8), OutlierBits: 4, RegularBits: 2}
	valueSpec := SplitSpec{RegularBits: 8}
	page := NewSplitKVCachePageWithSeed(dim, keySpec, dim, valueSpec, 4, 21)
	rng := rand.New(rand.NewSource(11))

	keys := make([][]float32, 4)
	for i := range keys {
		keys[i] = scaledVector(randomUnitVector(dim, rng), float32(1+i))
		page.Append(keys[i], randomUnitVector(dim, rng))
	}
	if page.Len() != 4 {
		t.Fatalf("Len = %d want 4", page.Len())
	}
	query := randomUnitVector(dim, rng)
	spq := page.PrepareQuery(query)

	idx := make([]uint32, 4)
	scores := make([]float32, 4)
	page.TopKPreparedTo(idx, scores, spq)

	kq := page.KeyQuantizer()
	for rank, slot := range idx {
		want := kq.InnerProduct(kq.Quantize(keys[slot]), query)
		if math.Abs(float64(scores[rank]-want)) > 1e-3*(1+math.Abs(float64(want))) {
			t.Fatalf("rank %d slot %d: page score %v want %v", rank, slot, scores[rank], want)
		}
	}
	for i := 1; i < len(scores); i++ {
		if scores[i] > scores[i-1] {
			t.Fatalf("top-k scores not sorted: %v", scores)
		}
	}

	out := make([]float32, dim)
	positions, weights := page.AttentionOutputPreparedTo(out, spq, 2)
	if len(positions) != 2 || len(weights) != 2 {
		t.Fatalf("attention output shape: positions=%v weights=%v", positions, weights)
	}
	var weightSum float32
	for _, w := range weights {
		weightSum += w
	}
	if math.Abs(float64(weightSum-1)) > 1e-5 {
		t.Fatalf("attention weights sum to %v want 1", weightSum)
	}
	for i, v := range out {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Fatalf("attention output[%d] = %v", i, v)
		}
	}
}

// TestSplitKVPageEffectiveBits reports the fractional code rates.
func TestSplitKVPageEffectiveBits(t *testing.T) {
	keySpec := SplitSpec{Outliers: seqInts(32), OutlierBits: 4, RegularBits: 2}
	valueSpec := SplitSpec{Outliers: seqInts(64), OutlierBits: 4, RegularBits: 3}
	page := NewSplitKVCachePageWithSeed(128, keySpec, 128, valueSpec, 8, 3)
	if got := page.EffectiveKeyBits(); math.Abs(got-2.5) > 1e-9 {
		t.Fatalf("EffectiveKeyBits = %v want 2.5", got)
	}
	if got := page.EffectiveValueBits(); math.Abs(got-3.5) > 1e-9 {
		t.Fatalf("EffectiveValueBits = %v want 3.5", got)
	}
}
