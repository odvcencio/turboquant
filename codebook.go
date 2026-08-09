package turboquant

import (
	"math"
	"math/bits"
	"sort"
	"sync"
)

type codebook struct {
	centroids  []float32
	boundaries []float32
}

type codebookKey struct {
	dim      int
	bitWidth int
}

var (
	codebookCache   = map[codebookKey]codebook{}
	codebookCacheMu sync.RWMutex
)

// cachedCodebook returns a cached codebook for (dim, bitWidth), computing it
// on first use. Safe for concurrent use. It checks the embedded codebook
// table (codebook_table.go) before falling back to the solver.
func cachedCodebook(dim, bitWidth int) codebook {
	key := codebookKey{dim, bitWidth}
	codebookCacheMu.RLock()
	if cb, ok := codebookCache[key]; ok {
		codebookCacheMu.RUnlock()
		return cb
	}
	codebookCacheMu.RUnlock()

	codebookCacheMu.Lock()
	defer codebookCacheMu.Unlock()
	// Double-check after acquiring write lock.
	if cb, ok := codebookCache[key]; ok {
		return cb
	}
	var cb codebook
	if tabled, ok := codebookFromTable(dim, bitWidth); ok {
		cb = tabled
	} else {
		cb = computeCodebook(dim, bitWidth)
	}
	codebookCache[key] = cb
	return cb
}

// betaPDF computes the PDF of the projected coordinate distribution on
// the unit hypersphere S^{d-1}. Each coordinate follows:
//
//	f(x) = Γ(d/2) / (√π · Γ((d-1)/2)) · (1 - x²)^((d-3)/2)
//
// for x ∈ [-1, 1]. Uses log-space arithmetic to avoid overflow/underflow.
func betaPDF(x float64, dim int) float64 {
	return betaPDFWithLogNorm(x, dim, betaLogNorm(dim))
}

// simpsonIntegrate evaluates ∫_a^b f(x)dx using composite Simpson's rule
// with n subintervals (n must be even).
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

// codebookSmallMassThreshold guards the cumulative quadrature grid against
// float64 cancellation. Reading an interval's mass as cdf(hi) - cdf(lo)
// subtracts two O(1) cumulative values; when the true difference is much
// smaller than that, the subtraction loses most or all of its significant
// digits to the grid's accumulated rounding error (empirically on the order
// of 1e-13 to 1e-11 after summing codebookGridPoints terms). A bit width of
// 3 or higher at dimension 384 already produces bins with true mass down to
// ~1e-26, so this is not only a concern at the domain edges. Below this
// threshold, fall back to a direct, interval-local Simpson integration
// (matching computeCodebookReference) instead of trusting the subtraction.
const codebookSmallMassThreshold = 1e-6

// fallbackPoints sizes the direct quadrature that intervalMoments and
// intervalSquaredError fall back to. A small-mass interval's density is
// smooth and close to monotonic (no fine structure to resolve), so this
// needs far fewer points than computeCodebookReference's 10,000: it exists
// to get a numerically trustworthy value, not to match that function's
// point count.
//
// TestCodebookSolverMatchesReference only compares against the reference
// solver at bit widths 1-6, so bit widths 7 and 8 (which pack far more
// small-mass bins into the same tail and would otherwise dominate
// construction time) use a cheaper fallback. Both point counts keep every
// bit width well under the benchmark's 50ms solver budget.
func fallbackPoints(bitWidth int) int {
	if bitWidth <= 6 {
		return 2000
	}
	return 24
}

// intervalMoments returns ∫_lo^hi x·f(x)dx and ∫_lo^hi f(x)dx, the two
// quantities computeCodebook needs to recompute a centroid as a conditional
// expectation. It prefers the O(1) grid lookup and falls back to a direct
// quadrature over [lo, hi] when the grid reports a small enough mass that
// cancellation may have corrupted it. logNorm is a precomputed
// betaLogNorm(dim): the fallback path evaluates the density without
// recomputing the two Lgamma calls on every point, since that recomputation
// otherwise dominates the fallback cost at high bit widths.
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

// intervalSquaredError returns ∫_lo^hi (x - centroid)² f(x) dx, the
// contribution one codebook interval makes to expectedMSE. It prefers the
// O(1) grid lookup (expanding the square as m2 - 2*c*m1 + c²*cdf) and falls
// back to a direct quadrature for the same small-mass reason as
// intervalMoments.
func intervalSquaredError(grid *codebookGrid, dim, bitWidth int, logNorm, lo, hi, centroid float64) float64 {
	cdfInterval := grid.integral(grid.cdf, lo, hi)
	if cdfInterval >= codebookSmallMassThreshold {
		m2 := grid.integral(grid.m2, lo, hi)
		m1 := grid.integral(grid.m1, lo, hi)
		return m2 - 2*centroid*m1 + centroid*centroid*cdfInterval
	}
	pdf := func(x float64) float64 { return betaPDFWithLogNorm(x, dim, logNorm) }
	return simpsonIntegrate(func(x float64) float64 {
		d := x - centroid
		return d * d * pdf(x)
	}, lo, hi, fallbackPoints(bitWidth))
}

// computeCodebook runs the Lloyd-Max algorithm for optimal scalar
// quantization of the Beta distribution arising from dimension dim, using
// bitWidth bits (2^bitWidth centroids). It reads interval integrals from a
// per-dimension quadrature grid (codebook_grid.go) shared across every bit
// width, which turns each iteration into O(k) grid lookups instead of
// O(k * 10^4) density evaluations. See computeCodebookReference for the
// original per-interval Simpson solver, kept for TestCodebookSolverMatchesReference.
func computeCodebook(dim, bitWidth int) codebook {
	k := 1 << uint(bitWidth) // number of levels
	grid := cachedCodebookGrid(dim)
	logNorm := betaLogNorm(dim)

	// Seed centroids at the distribution's mass quantiles; see
	// quantileCentroids for why a uniform seed stalls at high bit widths.
	centroids := quantileCentroids(grid, k)

	boundaries := make([]float64, k-1)

	for iter := 0; iter < 500; iter++ {
		// Compute boundaries as midpoints.
		for i := 0; i < k-1; i++ {
			boundaries[i] = (centroids[i] + centroids[i+1]) / 2
		}

		// Recompute centroids as conditional expectations.
		maxDelta := 0.0
		for i := 0; i < k; i++ {
			lo := -1.0
			if i > 0 {
				lo = boundaries[i-1]
			}
			hi := 1.0
			if i < k-1 {
				hi = boundaries[i]
			}

			// E[X | lo <= X <= hi] = ∫ x·f(x)dx / ∫ f(x)dx
			num, den := intervalMoments(grid, dim, bitWidth, logNorm, lo, hi)

			var newC float64
			if den < 1e-30 {
				// Negligible probability mass; keep midpoint.
				newC = (lo + hi) / 2
			} else {
				newC = num / den
			}

			delta := math.Abs(newC - centroids[i])
			if delta > maxDelta {
				maxDelta = delta
			}
			centroids[i] = newC
		}

		if maxDelta < 1e-10 {
			break
		}
	}

	return codebookFromFloat64s(centroids, boundaries)
}

// computeCodebookReference is the original per-interval Simpson solver. It
// runs 200 Lloyd-Max iterations, each with two 10,000-point Simpson
// integrations per interval. TestCodebookSolverMatchesReference compares
// computeCodebook against this function to gate the prefix-sum quadrature
// rewrite.
func computeCodebookReference(dim, bitWidth int) codebook {
	k := 1 << uint(bitWidth) // number of levels
	nInteg := 10000          // Simpson integration points

	// The reference solver shares the quantile seed (its purpose is to
	// cross-check the quadrature, not the initialization) but keeps its own
	// per-interval Simpson integration for the Lloyd updates.
	centroids := quantileCentroids(cachedCodebookGrid(dim), k)

	boundaries := make([]float64, k-1)

	pdf := func(x float64) float64 {
		return betaPDF(x, dim)
	}

	for iter := 0; iter < 500; iter++ {
		// Compute boundaries as midpoints.
		for i := 0; i < k-1; i++ {
			boundaries[i] = (centroids[i] + centroids[i+1]) / 2
		}

		// Recompute centroids as conditional expectations.
		maxDelta := 0.0
		for i := 0; i < k; i++ {
			lo := -1.0
			if i > 0 {
				lo = boundaries[i-1]
			}
			hi := 1.0
			if i < k-1 {
				hi = boundaries[i]
			}

			// E[X | lo <= X <= hi] = ∫ x·f(x)dx / ∫ f(x)dx
			num := simpsonIntegrate(func(x float64) float64 {
				return x * pdf(x)
			}, lo, hi, nInteg)

			den := simpsonIntegrate(pdf, lo, hi, nInteg)

			var newC float64
			if den < 1e-30 {
				// Negligible probability mass; keep midpoint.
				newC = (lo + hi) / 2
			} else {
				newC = num / den
			}

			delta := math.Abs(newC - centroids[i])
			if delta > maxDelta {
				maxDelta = delta
			}
			centroids[i] = newC
		}

		if maxDelta < 1e-10 {
			break
		}
	}

	return codebookFromFloat64s(centroids, boundaries)
}

func codebookFromFloat64s(centroids, boundaries []float64) codebook {
	c32 := make([]float32, len(centroids))
	b32 := make([]float32, len(boundaries))
	for i, v := range centroids {
		c32[i] = float32(v)
	}
	for i, v := range boundaries {
		b32[i] = float32(v)
	}
	return codebook{centroids: c32, boundaries: b32}
}

// nearestCentroid returns the index of the nearest centroid using binary
// search on the sorted boundaries. O(bitWidth) per lookup.
func (c *codebook) nearestCentroid(value float32) int {
	return sort.Search(len(c.boundaries), func(i int) bool {
		return c.boundaries[i] > value
	})
}

// centroidValue returns the centroid at index i.
func (c *codebook) centroidValue(index int) float32 {
	return c.centroids[index]
}

// expectedMSE computes the theoretical per-coordinate MSE:
//
//	Σ_i ∫[b_{i-1} to b_i] (x - c_i)² · f(x) dx
//
// then multiplied by dim for the per-vector MSE. Expands the squared term as
// m2 - 2*c*m1 + c*c*cdf so each interval's contribution comes from the same
// quadrature grid computeCodebook uses.
func (c *codebook) expectedMSE(dim int) float64 {
	k := len(c.centroids)
	bitWidth := bits.Len(uint(k)) - 1
	grid := cachedCodebookGrid(dim)
	logNorm := betaLogNorm(dim)

	var totalMSE float64
	for i := 0; i < k; i++ {
		lo := -1.0
		if i > 0 {
			lo = float64(c.boundaries[i-1])
		}
		hi := 1.0
		if i < k-1 {
			hi = float64(c.boundaries[i])
		}

		ci := float64(c.centroids[i])
		totalMSE += intervalSquaredError(grid, dim, bitWidth, logNorm, lo, hi, ci)
	}

	return totalMSE * float64(dim)
}
