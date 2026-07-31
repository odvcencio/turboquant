package recall

import (
	"math"
	"math/rand"
)

// PQParams configures a product quantization (PQ) baseline.
type PQParams struct {
	Subvectors int   // number of subvector splits; must divide dim
	Bits       int   // bits per subquantizer, 1-8
	Iterations int   // Lloyd k-means iterations
	Seed       int64
}

// DefaultPQParams picks a subvector count that divides dim, 8-bit
// subquantizers, and a fixed iteration count.
func DefaultPQParams(dim int, seed int64) PQParams {
	return PQParams{
		Subvectors: defaultSubvectors(dim),
		Bits:       8,
		Iterations: 15,
		Seed:       seed,
	}
}

func defaultSubvectors(dim int) int {
	for _, m := range []int{32, 16, 8, 4, 2} {
		if dim%m == 0 {
			return m
		}
	}
	return 1
}

// PQCodebook holds trained centroids for one product quantizer.
type PQCodebook struct {
	Dim        int
	Subvectors int
	SubDim     int
	Levels     int // number of centroids per subvector, at most 256
	Centroids  []float32 // Subvectors * Levels * SubDim, row-major per subvector
}

// TrainPQ runs Lloyd k-means per subvector on corpus (row-major, count rows
// of length dim) and returns a trained codebook.
func TrainPQ(corpus []float32, dim, count int, params PQParams) *PQCodebook {
	sub := params.Subvectors
	if sub <= 0 || dim%sub != 0 {
		sub = defaultSubvectors(dim)
	}
	subDim := dim / sub
	levels := 1 << uint(params.Bits)
	if levels > 256 {
		levels = 256
	}
	if levels > count {
		levels = count
	}
	if levels < 1 {
		levels = 1
	}
	iterations := params.Iterations
	if iterations <= 0 {
		iterations = 15
	}

	rng := rand.New(rand.NewSource(params.Seed))
	centroids := make([]float32, sub*levels*subDim)
	assign := make([]int, count)
	samples := make([]float32, count*subDim)

	for s := 0; s < sub; s++ {
		for i := 0; i < count; i++ {
			copy(samples[i*subDim:(i+1)*subDim], corpus[i*dim+s*subDim:i*dim+(s+1)*subDim])
		}
		subCentroids := centroids[s*levels*subDim : (s+1)*levels*subDim]
		kmeans(samples, count, subDim, levels, iterations, rng, subCentroids, assign)
	}

	return &PQCodebook{
		Dim:        dim,
		Subvectors: sub,
		SubDim:     subDim,
		Levels:     levels,
		Centroids:  centroids,
	}
}

func kmeans(samples []float32, count, dim, levels, iterations int, rng *rand.Rand, centroids []float32, assign []int) {
	perm := rng.Perm(count)
	for i := 0; i < levels; i++ {
		src := perm[i%count]
		copy(centroids[i*dim:(i+1)*dim], samples[src*dim:(src+1)*dim])
	}

	counts := make([]int, levels)
	sums := make([]float64, levels*dim)

	for iter := 0; iter < iterations; iter++ {
		for i := range counts {
			counts[i] = 0
		}
		for i := range sums {
			sums[i] = 0
		}
		for i := 0; i < count; i++ {
			row := samples[i*dim : (i+1)*dim]
			best := nearestCentroidL2(row, centroids, levels, dim)
			assign[i] = best
			counts[best]++
			for j := 0; j < dim; j++ {
				sums[best*dim+j] += float64(row[j])
			}
		}
		for c := 0; c < levels; c++ {
			if counts[c] == 0 {
				// Re-seed a dead centroid from a random sample.
				src := rng.Intn(count)
				copy(centroids[c*dim:(c+1)*dim], samples[src*dim:(src+1)*dim])
				continue
			}
			for j := 0; j < dim; j++ {
				centroids[c*dim+j] = float32(sums[c*dim+j] / float64(counts[c]))
			}
		}
	}
}

func nearestCentroidL2(row, centroids []float32, levels, dim int) int {
	best := 0
	bestDist := math.MaxFloat64
	for c := 0; c < levels; c++ {
		cRow := centroids[c*dim : (c+1)*dim]
		var dist float64
		for j := 0; j < dim; j++ {
			d := float64(row[j]) - float64(cRow[j])
			dist += d * d
		}
		if dist < bestDist {
			bestDist = dist
			best = c
		}
	}
	return best
}

// BytesPerVector returns the encoded size of one vector: one byte per
// subvector, because Levels is capped at 256.
func (c *PQCodebook) BytesPerVector() int {
	return c.Subvectors
}

// Encode assigns vec to its nearest centroid per subvector, writing one byte
// index per subvector into dst. dst must have length BytesPerVector().
func (c *PQCodebook) Encode(dst []byte, vec []float32) {
	for s := 0; s < c.Subvectors; s++ {
		row := vec[s*c.SubDim : (s+1)*c.SubDim]
		subCentroids := c.Centroids[s*c.Levels*c.SubDim : (s+1)*c.Levels*c.SubDim]
		dst[s] = byte(nearestCentroidL2(row, subCentroids, c.Levels, c.SubDim))
	}
}

// BuildQueryLUT builds a per-subvector, per-level inner-product lookup table
// for query into dst. dst must have length Subvectors*Levels.
func (c *PQCodebook) BuildQueryLUT(dst []float32, query []float32) {
	for s := 0; s < c.Subvectors; s++ {
		row := query[s*c.SubDim : (s+1)*c.SubDim]
		subCentroids := c.Centroids[s*c.Levels*c.SubDim : (s+1)*c.Levels*c.SubDim]
		table := dst[s*c.Levels : (s+1)*c.Levels]
		for l := 0; l < c.Levels; l++ {
			cRow := subCentroids[l*c.SubDim : (l+1)*c.SubDim]
			var dot float32
			for j := 0; j < c.SubDim; j++ {
				dot += row[j] * cRow[j]
			}
			table[l] = dot
		}
	}
}

// Score estimates <x, query> for an encoded vector using a precomputed LUT
// from BuildQueryLUT.
func (c *PQCodebook) Score(code []byte, lut []float32) float32 {
	var sum float32
	for s := 0; s < c.Subvectors; s++ {
		sum += lut[s*c.Levels+int(code[s])]
	}
	return sum
}
