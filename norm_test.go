package turboquant

import (
	"math"
	"math/rand"
	"testing"
)

func scaledVector(v []float32, alpha float32) []float32 {
	out := make([]float32, len(v))
	for i := range v {
		out[i] = v[i] * alpha
	}
	return out
}

// combineUnits returns a*u + b*w for unit vectors u, w.
func combineUnits(a float32, u []float32, b float32, w []float32) []float32 {
	out := make([]float32, len(u))
	for i := range u {
		out[i] = a*u[i] + b*w[i]
	}
	return out
}

// orthogonalUnit returns a random unit vector orthogonal to every vector in
// basis (each of which must already be unit-norm and mutually orthogonal).
func orthogonalUnit(rng *rand.Rand, dim int, basis ...[]float32) []float32 {
	v := randomUnitVector(dim, rng)
	for _, b := range basis {
		var dot float32
		for i := range v {
			dot += v[i] * b[i]
		}
		for i := range v {
			v[i] -= dot * b[i]
		}
	}
	norm := vecNorm(v)
	for i := range v {
		v[i] /= norm
	}
	return v
}

// TestIPInnerProductScalesWithInputNorm pins the paper's non-unit-vector
// contract: quantizing alpha*x must produce an estimate of <alpha*x, y>, not
// of <x/||x||, y>. Doubling the input is exact in binary float, so the codes
// match and the estimates must differ by exactly the norm ratio.
func TestIPInnerProductScalesWithInputNorm(t *testing.T) {
	const dim, bits = 96, 3
	q := NewIPHadamardWithSeed(dim, bits, 41)
	rng := rand.New(rand.NewSource(5))
	x := randomUnitVector(dim, rng)
	y := scaledVector(randomUnitVector(dim, rng), 1.7)

	qx := q.Quantize(x)
	qx2 := q.Quantize(scaledVector(x, 2))
	est := q.InnerProduct(qx, y)
	est2 := q.InnerProduct(qx2, y)
	if math.Abs(float64(est)) < 1e-4 {
		t.Fatalf("degenerate base estimate %v", est)
	}
	if rel := math.Abs(float64(est2-2*est)) / math.Abs(float64(2*est)); rel > 1e-4 {
		t.Fatalf("InnerProduct does not scale with input norm: est(x)=%v est(2x)=%v want ratio 2", est, est2)
	}

	pq := q.PrepareQuery(y)
	estP := q.InnerProductPrepared(qx2, pq)
	if rel := math.Abs(float64(estP-2*est)) / math.Abs(float64(2*est)); rel > 1e-3 {
		t.Fatalf("InnerProductPrepared does not scale with input norm: got %v want %v", estP, 2*est)
	}

	batch := make([]float32, 2)
	q.InnerProductPreparedBatchTo(batch, qx2, []PreparedQuery{pq, pq})
	for i, got := range batch {
		if rel := math.Abs(float64(got-2*est)) / math.Abs(float64(2*est)); rel > 1e-3 {
			t.Fatalf("prepared batch[%d] does not scale with input norm: got %v want %v", i, got, 2*est)
		}
	}
}

// TestIPDequantizeScalesWithInputNorm requires DeQuant_prod to rescale by the
// stored input norm, per the paper's norm-storage note.
func TestIPDequantizeScalesWithInputNorm(t *testing.T) {
	const dim, bits = 64, 4
	q := NewIPHadamardWithSeed(dim, bits, 17)
	rng := rand.New(rand.NewSource(6))
	x := randomUnitVector(dim, rng)
	base := q.Dequantize(q.Quantize(x))
	scaled := q.Dequantize(q.Quantize(scaledVector(x, 2)))
	for i := range base {
		if diff := math.Abs(float64(scaled[i] - 2*base[i])); diff > 1e-4*(1+math.Abs(float64(2*base[i]))) {
			t.Fatalf("Dequantize does not scale with input norm at %d: base=%v scaled=%v", i, base[i], scaled[i])
		}
	}
}

// TestIPUnbiasedForNonUnitInputs checks E[<y, deq(quant(x))>] = <x, y> for an
// input with norm 3 against a correlated query with norm 2. The current
// unit-space estimator has a systematic multiplicative bias of 1/3 here,
// far outside the confidence interval.
func TestIPUnbiasedForNonUnitInputs(t *testing.T) {
	const dim, bits, trials = 128, 3, 400
	rng := rand.New(rand.NewSource(9))
	xUnit := randomUnitVector(dim, rng)
	perp := orthogonalUnit(rng, dim, xUnit)
	x := scaledVector(xUnit, 3)
	y := scaledVector(combineUnits(0.8, xUnit, 0.6, perp), 2)

	var want float64
	for i := range x {
		want += float64(x[i]) * float64(y[i])
	}

	var sumErr, sumSq float64
	for trial := 0; trial < trials; trial++ {
		q := NewIPHadamardWithSeed(dim, bits, int64(1000+trial))
		est := float64(q.InnerProduct(q.Quantize(x), y))
		e := est - want
		sumErr += e
		sumSq += e * e
	}
	mean := sumErr / trials
	variance := sumSq/trials - mean*mean
	ci := 4 * math.Sqrt(variance/trials)
	if math.Abs(mean) > ci+1e-3 {
		t.Fatalf("IP estimator biased for non-unit input: mean err %.5f (CI ±%.5f), true IP %.4f", mean, ci, want)
	}
}

// TestKVAttentionRespectsKeyNorms builds two keys with identical query
// alignment but norms 1 and 10. True attention logits are 0.5 vs 5.0, so
// softmax must put essentially all weight on the high-norm key. A cache that
// drops key norms scores both keys equally.
func TestKVAttentionRespectsKeyNorms(t *testing.T) {
	const dim = 64
	rng := rand.New(rand.NewSource(21))
	query := randomUnitVector(dim, rng)
	p1 := orthogonalUnit(rng, dim, query)
	p2 := orthogonalUnit(rng, dim, query, p1)
	root75 := float32(math.Sqrt(0.75))
	kSmall := combineUnits(0.5, query, root75, p1)
	kBig := scaledVector(combineUnits(0.5, query, root75, p2), 10)

	vSmall := make([]float32, dim)
	vSmall[0] = 1
	vBig := make([]float32, dim)
	vBig[1] = 1

	page := NewKVCachePageWithSeed(dim, 4, dim, 8, 4, 77)
	page.Append(kSmall, vSmall)
	page.Append(kBig, vBig)

	pq := page.PrepareQuery(query)
	out := make([]float32, dim)
	positions, weights := page.AttentionOutputPreparedTo(out, pq, 2)

	var wBig float32
	for i, pos := range positions {
		if pos == 1 {
			wBig = weights[i]
		}
	}
	if wBig < 0.9 {
		t.Fatalf("attention weight on high-norm key = %v, want > 0.9 (positions=%v weights=%v)", wBig, positions, weights)
	}
	if out[1] < 0.8 {
		t.Fatalf("attention output ignores key norms: out[0]=%v out[1]=%v", out[0], out[1])
	}
}

// TestWireIPRoundtripPreservesNorm requires the wire format to carry the
// input norm: an encoded/decoded record must reproduce the scaled estimate.
func TestWireIPRoundtripPreservesNorm(t *testing.T) {
	const dim, bits = 48, 3
	q := NewIPHadamardWithSeed(dim, bits, 3)
	rng := rand.New(rand.NewSource(4))
	xUnit := randomUnitVector(dim, rng)
	y := randomUnitVector(dim, rng)

	estUnit := q.InnerProduct(q.Quantize(xUnit), y)
	if math.Abs(float64(estUnit)) < 1e-4 {
		t.Fatalf("degenerate base estimate %v", estUnit)
	}

	qx := q.Quantize(scaledVector(xUnit, 2))
	_, _, decoded, err := DecodeIP(EncodeIP(dim, bits, qx))
	if err != nil {
		t.Fatal(err)
	}
	roundtrip := q.InnerProduct(decoded, y)
	if rel := math.Abs(float64(roundtrip-2*estUnit)) / math.Abs(float64(2*estUnit)); rel > 1e-3 {
		t.Fatalf("wire roundtrip loses input norm: got %v want %v", roundtrip, 2*estUnit)
	}
}

// TestKVPageScoreMatchesDirectQuantization pins the page-level key score to
// the direct IPQuantizer score for the same key and query. The page stores
// packed MSE codes and signs in place, but must also persist the residual
// norm (and input norm) that QuantizeTo reports; dropping either silently
// removes the QJL correction term from every stored key.
func TestKVPageScoreMatchesDirectQuantization(t *testing.T) {
	const dim = 64
	rng := rand.New(rand.NewSource(51))
	page := NewKVCachePageWithSeed(dim, 4, dim, 8, 4, 99)
	k := randomUnitVector(dim, rng)
	v := randomUnitVector(dim, rng)
	page.Append(k, v)
	y := randomUnitVector(dim, rng)
	pq := page.PrepareQuery(y)

	want := page.KeyQuantizer().InnerProductPrepared(page.KeyQuantizer().Quantize(k), pq)
	idx := make([]uint32, 1)
	scores := make([]float32, 1)
	page.TopKPreparedTo(idx, scores, pq)
	if math.Abs(float64(scores[0]-want)) > 1e-5*(1+math.Abs(float64(want))) {
		t.Fatalf("page key score %v differs from direct quantizer score %v (residual norm dropped on Append?)", scores[0], want)
	}
}

// TestKVPageSerializeRoundtripKeyNorms requires KV page serialization to carry
// per-token key norms: attention output must survive marshal/unmarshal for
// non-unit keys.
func TestKVPageSerializeRoundtripKeyNorms(t *testing.T) {
	const dim = 32
	rng := rand.New(rand.NewSource(31))
	page := NewKVCachePageWithSeed(dim, 4, dim, 8, 4, 13)
	for i := 0; i < 3; i++ {
		k := scaledVector(randomUnitVector(dim, rng), float32(1+3*i))
		v := randomUnitVector(dim, rng)
		page.Append(k, v)
	}
	query := randomUnitVector(dim, rng)
	pq := page.PrepareQuery(query)
	wantIdx := make([]uint32, 2)
	wantScores := make([]float32, 2)
	page.TopKPreparedTo(wantIdx, wantScores, pq)

	data, err := page.MarshalBinary()
	if err != nil {
		t.Fatal(err)
	}
	restored, err := UnmarshalKVCachePage(data)
	if err != nil {
		t.Fatal(err)
	}
	gotIdx := make([]uint32, 2)
	gotScores := make([]float32, 2)
	restored.TopKPreparedTo(gotIdx, gotScores, restored.PrepareQuery(query))
	for i := range wantScores {
		if gotIdx[i] != wantIdx[i] {
			t.Fatalf("restored top-k order differs: got %v want %v", gotIdx, wantIdx)
		}
		if math.Abs(float64(gotScores[i]-wantScores[i])) > 1e-5*(1+math.Abs(float64(wantScores[i]))) {
			t.Fatalf("restored top-k scores differ at %d: got %v want %v", i, gotScores[i], wantScores[i])
		}
	}
}
