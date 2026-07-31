package turboquant

import (
	"math"
	"sync"
)

// codebookGridPoints is the number of intervals in the uniform quadrature
// grid over [-1, 1]. The grid has codebookGridPoints+1 points, spaced about
// 7.6e-6 apart. The Beta density this grid samples vanishes at +/-1, so a
// grid this fine keeps composite-Simpson error far below the 1e-6 comparison
// tolerance in TestCodebookSolverMatchesReference.
const codebookGridPoints = 1 << 18

// codebookGrid holds cumulative composite-Simpson quadrature arrays for the
// projected-coordinate density at one dimension. cdf, m1, and m2 approximate
// the integral of f, x*f, and x*x*f respectively from -1 to each grid point.
// Every bit width for a dimension shares one grid: computeCodebook reads any
// interval's integral by subtracting two cumulative values and, when a
// boundary falls between grid points, linearly interpolating.
type codebookGrid struct {
	h   float64 // grid spacing, 2 / codebookGridPoints
	cdf []float64
	m1  []float64
	m2  []float64
}

var (
	codebookGridCache   = map[int]*codebookGrid{}
	codebookGridCacheMu sync.RWMutex
)

// cachedCodebookGrid returns a cached quadrature grid for dim, building it on
// first use. Safe for concurrent use.
func cachedCodebookGrid(dim int) *codebookGrid {
	codebookGridCacheMu.RLock()
	if g, ok := codebookGridCache[dim]; ok {
		codebookGridCacheMu.RUnlock()
		return g
	}
	codebookGridCacheMu.RUnlock()

	codebookGridCacheMu.Lock()
	defer codebookGridCacheMu.Unlock()
	// Double-check after acquiring the write lock.
	if g, ok := codebookGridCache[dim]; ok {
		return g
	}
	g := buildCodebookGrid(dim)
	codebookGridCache[dim] = g
	return g
}

// buildCodebookGrid samples the projected-coordinate density once across a
// uniform grid and builds cumulative integral arrays for f, x*f, and x*x*f
// in a single pass.
func buildCodebookGrid(dim int) *codebookGrid {
	n := codebookGridPoints
	h := 2.0 / float64(n)
	logNorm := betaLogNorm(dim)

	f := make([]float64, n+1)
	xf := make([]float64, n+1)
	x2f := make([]float64, n+1)
	for i := 0; i <= n; i++ {
		x := -1.0 + float64(i)*h
		v := betaPDFWithLogNorm(x, dim, logNorm)
		f[i] = v
		xf[i] = x * v
		x2f[i] = x * x * v
	}

	return &codebookGrid{
		h:   h,
		cdf: cumulativeSimpson(f, h),
		m1:  cumulativeSimpson(xf, h),
		m2:  cumulativeSimpson(x2f, h),
	}
}

// cumulativeSimpson returns s where s[i] approximates the integral of the
// sampled function from grid point 0 to grid point i, using composite
// Simpson's rule on consecutive pairs of intervals. Each pair (values[k],
// values[k+1], values[k+2]) gives both the half-interval integral (to
// s[k+1]) and the full-pair integral (to s[k+2]) from the same quadratic fit,
// so every grid point gets a Simpson-accurate cumulative value in one pass.
// len(values) must be odd (an even number of intervals). s[0] is always 0.
func cumulativeSimpson(values []float64, h float64) []float64 {
	n := len(values) - 1
	s := make([]float64, n+1)
	for k := 0; k+2 <= n; k += 2 {
		f0, f1, f2 := values[k], values[k+1], values[k+2]
		s[k+1] = s[k] + h/12*(5*f0+8*f1-f2)
		s[k+2] = s[k] + h/3*(f0+4*f1+f2)
	}
	return s
}

// at returns the interpolated cumulative value at x, linearly interpolating
// between the two nearest grid points when x does not land exactly on one.
func (g *codebookGrid) at(s []float64, x float64) float64 {
	n := len(s) - 1
	if x <= -1 {
		return 0
	}
	if x >= 1 {
		return s[n]
	}
	t := (x + 1) / g.h
	i0 := int(t)
	if i0 >= n {
		return s[n]
	}
	frac := t - float64(i0)
	return s[i0] + frac*(s[i0+1]-s[i0])
}

// integral returns the integral of the sampled function over [lo, hi] using
// the cumulative array s (one of g.cdf, g.m1, or g.m2).
func (g *codebookGrid) integral(s []float64, lo, hi float64) float64 {
	return g.at(s, hi) - g.at(s, lo)
}

// betaLogNorm computes the log normalization constant for the projected
// coordinate density on the unit hypersphere S^{d-1}:
//
//	log Γ(d/2) - log(√π) - log Γ((d-1)/2)
//
// Hoisting this out of the per-point density evaluation means the quadrature
// grid pays for the two Lgamma calls once per dimension instead of twice per
// grid point.
func betaLogNorm(dim int) float64 {
	d := float64(dim)
	lgD2, _ := math.Lgamma(d / 2)
	lgDm12, _ := math.Lgamma((d - 1) / 2)
	return lgD2 - 0.5*math.Log(math.Pi) - lgDm12
}

// betaPDFWithLogNorm evaluates the projected-coordinate density at x given a
// precomputed log normalization constant from betaLogNorm.
func betaPDFWithLogNorm(x float64, dim int, logNorm float64) float64 {
	if x <= -1 || x >= 1 {
		return 0
	}
	d := float64(dim)
	exponent := (d - 3) / 2
	onemx2 := 1 - x*x
	logBody := exponent * math.Log(onemx2)
	return math.Exp(logNorm + logBody)
}
