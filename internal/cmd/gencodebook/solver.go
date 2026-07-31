package main

import (
	"math"
	"sort"
)

// This file duplicates the prefix-sum quadrature grid and Lloyd-Max solver
// from codebook_grid.go and codebook.go in the turboquant package.
// computeCodebook and cachedCodebookGrid stay unexported there ("Keep
// codebook, computeCodebook, and cachedCodebook unexported. Do not widen the
// public surface beyond [PrecomputeCodebook, CodebookSource, TabulatedDims]"
// per the hardening spec), so a separate main package cannot import them.
// Keep this copy numerically identical to codebook.go and codebook_grid.go:
// same grid size, same cumulative-Simpson formulas, same Lloyd-Max iteration
// count and convergence threshold. TestCodebookTableMatchesSolver in the
// turboquant package gates that identity.

const codebookGridPoints = 1 << 18

type codebookGrid struct {
	h   float64
	cdf []float64
	m1  []float64
	m2  []float64
}

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

func (g *codebookGrid) integral(s []float64, lo, hi float64) float64 {
	return g.at(s, hi) - g.at(s, lo)
}

func betaLogNorm(dim int) float64 {
	d := float64(dim)
	lgD2, _ := math.Lgamma(d / 2)
	lgDm12, _ := math.Lgamma((d - 1) / 2)
	return lgD2 - 0.5*math.Log(math.Pi) - lgDm12
}

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

// simpsonIntegrate evaluates the integral of f over [a, b] using composite
// Simpson's rule with n subintervals (n must be even).
func simpsonIntegrate(f func(float64) float64, a, b float64, n int) float64 {
	if n%2 != 0 {
		n++
	}
	h := (b - a) / float64(n)
	sum := f(a) + f(b)
	for i := 1; i < n; i++ {
		x := a + float64(i)*h
		if i%2 == 0 {
			sum += 2 * f(x)
		} else {
			sum += 4 * f(x)
		}
	}
	return sum * h / 3
}

// codebookSmallMassThreshold and fallbackPoints match the same names in
// codebook.go: below this mass, the cumulative-array subtraction may have
// lost significant digits to cancellation, so fall back to a direct
// interval-local Simpson integration, sized down at high bit widths where
// far more bins take this path. Keep both in lockstep with codebook.go;
// TestCodebookTableMatchesSolver gates that identity.
const codebookSmallMassThreshold = 1e-6

func fallbackPoints(bitWidth int) int {
	if bitWidth <= 6 {
		return 2000
	}
	return 24
}

func intervalMoments(grid *codebookGrid, dim, bitWidth int, logNorm, lo, hi float64) (num, den float64) {
	num = grid.integral(grid.m1, lo, hi)
	den = grid.integral(grid.cdf, lo, hi)
	if den >= codebookSmallMassThreshold {
		return num, den
	}
	pdf := func(x float64) float64 { return betaPDFWithLogNorm(x, dim, logNorm) }
	n := fallbackPoints(bitWidth)
	den = simpsonIntegrate(pdf, lo, hi, n)
	num = simpsonIntegrate(func(x float64) float64 { return x * pdf(x) }, lo, hi, n)
	return num, den
}

// computeCodebook runs the Lloyd-Max algorithm using grid, returning
// centroids and boundaries as float32 in the same layout codebook.go uses:
// centroids (2^bitWidth values) then boundaries (2^bitWidth - 1 values).
func computeCodebook(dim int, grid *codebookGrid, bitWidth int) (centroids, boundaries []float32) {
	k := 1 << uint(bitWidth)
	logNorm := betaLogNorm(dim)

	c := make([]float64, k)
	for i := 0; i < k; i++ {
		c[i] = -1 + (2*float64(i)+1)/float64(k)
	}
	sort.Float64s(c)

	b := make([]float64, k-1)

	for iter := 0; iter < 200; iter++ {
		for i := 0; i < k-1; i++ {
			b[i] = (c[i] + c[i+1]) / 2
		}

		maxDelta := 0.0
		for i := 0; i < k; i++ {
			lo := -1.0
			if i > 0 {
				lo = b[i-1]
			}
			hi := 1.0
			if i < k-1 {
				hi = b[i]
			}

			num, den := intervalMoments(grid, dim, bitWidth, logNorm, lo, hi)

			var newC float64
			if den < 1e-30 {
				newC = (lo + hi) / 2
			} else {
				newC = num / den
			}

			delta := math.Abs(newC - c[i])
			if delta > maxDelta {
				maxDelta = delta
			}
			c[i] = newC
		}

		if maxDelta < 1e-10 {
			break
		}
	}

	centroids = make([]float32, k)
	boundaries = make([]float32, k-1)
	for i, v := range c {
		centroids[i] = float32(v)
	}
	for i, v := range b {
		boundaries[i] = float32(v)
	}
	return centroids, boundaries
}
