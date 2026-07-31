package recall

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"hash"
	"hash/fnv"
	"math"
	"os"
)

// GroundTruth holds exact top-k neighbour identifiers per query.
type GroundTruth struct {
	K         int
	Neighbors []uint32 // len == QueryCount * K
	Checksum  uint64
}

// ComputeGroundTruth runs exact search with float64 accumulation.
func ComputeGroundTruth(d *Dataset, k int) *GroundTruth {
	corpusCount := d.CorpusCount()
	queryCount := d.QueryCount()
	if k > corpusCount {
		k = corpusCount
	}
	neighbors := make([]uint32, queryCount*k)
	if corpusCount == 0 || queryCount == 0 || k == 0 {
		return &GroundTruth{K: k, Neighbors: neighbors, Checksum: checksumDataset(d)}
	}

	var corpusNorms []float64
	if d.Metric == MetricCosine {
		corpusNorms = make([]float64, corpusCount)
		for c := 0; c < corpusCount; c++ {
			corpusNorms[c] = rowNorm64(d.Corpus[c*d.Dim : (c+1)*d.Dim])
		}
	}

	scores := make([]float32, corpusCount)
	dstIdx := make([]uint32, k)
	dstScores := make([]float32, k)
	for q := 0; q < queryCount; q++ {
		query := d.Queries[q*d.Dim : (q+1)*d.Dim]
		var queryNorm float64
		if d.Metric == MetricCosine {
			queryNorm = rowNorm64(query)
		}
		for c := 0; c < corpusCount; c++ {
			row := d.Corpus[c*d.Dim : (c+1)*d.Dim]
			dot := dot64(row, query)
			if d.Metric == MetricCosine {
				denom := corpusNorms[c] * queryNorm
				if denom > 1e-30 {
					dot /= denom
				} else {
					dot = 0
				}
			}
			scores[c] = float32(dot)
		}
		TopK(dstIdx, dstScores, scores)
		copy(neighbors[q*k:(q+1)*k], dstIdx)
	}

	return &GroundTruth{
		K:         k,
		Neighbors: neighbors,
		Checksum:  checksumDataset(d),
	}
}

// groundTruthFile is the on-disk cache layout. It stores a checksum of the
// corpus and query set alongside the neighbour list so a cache reader can
// detect a stale entry.
type groundTruthFile struct {
	K         int      `json:"k"`
	Checksum  uint64   `json:"checksum"`
	Neighbors []uint32 `json:"neighbors"`
}

// LoadOrComputeGroundTruth reads a cached file or recomputes and writes it.
// The cache is invalidated whenever the dataset checksum or k changes.
func LoadOrComputeGroundTruth(path string, d *Dataset, k int) (*GroundTruth, error) {
	checksum := checksumDataset(d)
	if data, err := os.ReadFile(path); err == nil {
		var file groundTruthFile
		if jsonErr := json.Unmarshal(data, &file); jsonErr == nil {
			if file.Checksum == checksum && file.K == k {
				return &GroundTruth{K: file.K, Neighbors: file.Neighbors, Checksum: file.Checksum}, nil
			}
		}
	}

	gt := ComputeGroundTruth(d, k)
	file := groundTruthFile{K: gt.K, Checksum: gt.Checksum, Neighbors: gt.Neighbors}
	data, err := json.Marshal(file)
	if err != nil {
		return nil, fmt.Errorf("recall: marshaling ground truth cache: %w", err)
	}
	if err := os.WriteFile(path, data, 0o644); err != nil {
		return nil, fmt.Errorf("recall: writing ground truth cache: %w", err)
	}
	return gt, nil
}

// Recall1AtK returns the fraction of queries whose true nearest neighbour
// appears in the approximate list of length k. approx must have length
// QueryCount * k, where QueryCount is inferred from gt.
func Recall1AtK(gt *GroundTruth, approx []uint32, k int) float64 {
	if gt.K == 0 {
		return 0
	}
	queryCount := len(gt.Neighbors) / gt.K
	if queryCount == 0 || k == 0 {
		return 0
	}
	hits := 0
	for q := 0; q < queryCount; q++ {
		trueNN := gt.Neighbors[q*gt.K]
		row := approx[q*k : (q+1)*k]
		for _, id := range row {
			if id == trueNN {
				hits++
				break
			}
		}
	}
	return float64(hits) / float64(queryCount)
}

func checksumDataset(d *Dataset) uint64 {
	h := fnv.New64a()
	writeFloatsToHash(h, d.Corpus)
	writeFloatsToHash(h, d.Queries)
	return h.Sum64()
}

func writeFloatsToHash(h hash.Hash64, values []float32) {
	var buf [4]byte
	for _, v := range values {
		binary.LittleEndian.PutUint32(buf[:], math.Float32bits(v))
		h.Write(buf[:])
	}
}

func dot64(a, b []float32) float64 {
	var sum float64
	for i := range a {
		sum += float64(a[i]) * float64(b[i])
	}
	return sum
}

func rowNorm64(v []float32) float64 {
	return math.Sqrt(dot64(v, v))
}
