package turboquant

import (
	"bytes"
	"math"
	"math/rand"
	"testing"
)

func qjlProjectScalar(signs []byte, x, projMatrix []float32, dim int) float32 {
	norm := vecNorm(x)
	for i := range signs {
		signs[i] = 0
	}
	for i := 0; i < dim; i++ {
		row := i * dim
		if dotFloat32sScalar(projMatrix[row:row+dim], x) >= 0 {
			signs[i/8] |= 1 << uint(i%8)
		}
	}
	return norm
}

func qjlInnerProductScalar(signs []byte, resNorm float32, y, projMatrix []float32, dim int) float32 {
	scale := float32(math.Sqrt(math.Pi/2.0)) / float32(dim) * resNorm
	var sum float32
	for i := 0; i < dim; i++ {
		row := i * dim
		dot := dotFloat32sScalar(projMatrix[row:row+dim], y)
		if signs[i/8]&(1<<uint(i%8)) != 0 {
			sum += dot
		} else {
			sum -= dot
		}
	}
	return scale * sum
}

func TestQJLUnbiased(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	dim := 384
	rng := rand.New(rand.NewSource(42))
	x := randomUnitVector(dim, rng)
	y := randomUnitVector(dim, rng)

	var trueIP float64
	for i := range x {
		trueIP += float64(x[i]) * float64(y[i])
	}

	var sumEstimate float64
	trials := 5000
	for trial := 0; trial < trials; trial++ {
		proj := generateGaussianMatrix(dim, rand.New(rand.NewSource(int64(trial+100))))
		signs := make([]byte, (dim+7)/8)
		resNorm := qjlProject(signs, x, proj, dim)
		estimate := qjlInnerProduct(signs, resNorm, y, proj, dim)
		sumEstimate += float64(estimate)
	}
	avgEstimate := sumEstimate / float64(trials)
	if math.Abs(avgEstimate-trueIP) > 0.05 {
		t.Errorf("QJL average estimate %.4f vs true IP %.4f", avgEstimate, trueIP)
	}
}

func TestQJLVarianceBound(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping statistical test under race detector")
	}
	dim := 384
	rng := rand.New(rand.NewSource(42))
	x := randomUnitVector(dim, rng)
	y := randomUnitVector(dim, rng)

	var trueIP float64
	var yNormSq float64
	for i := range x {
		trueIP += float64(x[i]) * float64(y[i])
		yNormSq += float64(y[i]) * float64(y[i])
	}

	var sumSqErr float64
	trials := 5000
	for trial := 0; trial < trials; trial++ {
		proj := generateGaussianMatrix(dim, rand.New(rand.NewSource(int64(trial+200))))
		signs := make([]byte, (dim+7)/8)
		resNorm := qjlProject(signs, x, proj, dim)
		estimate := qjlInnerProduct(signs, resNorm, y, proj, dim)
		err := float64(estimate) - trueIP
		sumSqErr += err * err
	}
	variance := sumSqErr / float64(trials)
	bound := math.Pi / (2.0 * float64(dim)) * yNormSq
	if variance > bound*2.0 {
		t.Errorf("QJL variance %.6f exceeds 2x bound %.6f", variance, bound)
	}
}

func TestQJLGroupedProjectionMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(123))
	for _, dim := range []int{8, 16, 31, 128, 130, 384} {
		x := randomUnitVector(dim, rng)
		y := randomUnitVector(dim, rng)
		proj := generateGaussianMatrix(dim, rand.New(rand.NewSource(int64(dim+500))))
		gotSigns := make([]byte, (dim+7)/8)
		wantSigns := make([]byte, (dim+7)/8)
		gotNorm := qjlProject(gotSigns, x, proj, dim)
		wantNorm := qjlProjectScalar(wantSigns, x, proj, dim)
		if math.Abs(float64(gotNorm-wantNorm)) > 1e-6 {
			t.Fatalf("dim=%d: got norm %.6f want %.6f", dim, gotNorm, wantNorm)
		}
		if !bytes.Equal(gotSigns, wantSigns) {
			t.Fatalf("dim=%d: projected signs mismatch", dim)
		}
		gotIP := qjlInnerProduct(gotSigns, gotNorm, y, proj, dim)
		wantIP := qjlInnerProductScalar(wantSigns, wantNorm, y, proj, dim)
		if math.Abs(float64(gotIP-wantIP)) > 1e-5 {
			t.Fatalf("dim=%d: got IP %.6f want %.6f", dim, gotIP, wantIP)
		}
	}
}

func TestQJLBlockedProjectionMatchesScalar(t *testing.T) {
	rng := rand.New(rand.NewSource(321))
	for _, dim := range []int{8, 16, 128, 384} {
		x := randomUnitVector(dim, rng)
		y := randomUnitVector(dim, rng)
		proj := generateGaussianMatrix(dim, rand.New(rand.NewSource(int64(dim+700))))
		proj8 := blockProjectionRows8(proj, dim)
		gotSigns := make([]byte, (dim+7)/8)
		wantSigns := make([]byte, (dim+7)/8)
		gotNorm := qjlProjectBlocked(gotSigns, x, proj, proj8, dim)
		wantNorm := qjlProjectScalar(wantSigns, x, proj, dim)
		if math.Abs(float64(gotNorm-wantNorm)) > 1e-6 {
			t.Fatalf("dim=%d: got norm %.6f want %.6f", dim, gotNorm, wantNorm)
		}
		if !bytes.Equal(gotSigns, wantSigns) {
			t.Fatalf("dim=%d: blocked projected signs mismatch", dim)
		}
		gotIP := qjlInnerProductBlocked(gotSigns, gotNorm, y, proj, proj8, dim)
		wantIP := qjlInnerProductScalar(wantSigns, wantNorm, y, proj, dim)
		if math.Abs(float64(gotIP-wantIP)) > 1e-5 {
			t.Fatalf("dim=%d: got IP %.6f want %.6f", dim, gotIP, wantIP)
		}
	}
}
