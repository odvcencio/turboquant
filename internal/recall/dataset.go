// Package recall measures TurboQuant search quality against exact search.
// It is an internal measurement tool, not part of the public API.
package recall

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"path/filepath"
	"strings"
)

// Metric names the similarity measure a Dataset ranks neighbours with.
type Metric int

const (
	// MetricInnerProduct ranks by raw dot product.
	MetricInnerProduct Metric = iota
	// MetricCosine ranks by cosine similarity.
	MetricCosine
)

// Dataset holds a corpus and a query set in row-major float32 layout.
type Dataset struct {
	Name    string
	Dim     int
	Corpus  []float32 // len(Corpus) == CorpusCount() * Dim
	Queries []float32 // len(Queries) == QueryCount() * Dim
	Metric  Metric
}

// CorpusCount returns the number of corpus vectors.
func (d *Dataset) CorpusCount() int {
	if d.Dim == 0 {
		return 0
	}
	return len(d.Corpus) / d.Dim
}

// QueryCount returns the number of query vectors.
func (d *Dataset) QueryCount() int {
	if d.Dim == 0 {
		return 0
	}
	return len(d.Queries) / d.Dim
}

// LoadDataset reads a dataset from a directory in the fvecs layout. It reads
// "<name>_base.fvecs" for the corpus and "<name>_query.fvecs" for the query
// set. An optional "<name>.metric" file selects the metric: the literal text
// "cosine" selects MetricCosine, and anything else, including a missing
// file, selects MetricInnerProduct.
func LoadDataset(dir, name string) (*Dataset, error) {
	corpus, corpusDim, err := readFvecs(filepath.Join(dir, name+"_base.fvecs"))
	if err != nil {
		return nil, fmt.Errorf("recall: loading corpus for %q: %w", name, err)
	}
	queries, queryDim, err := readFvecs(filepath.Join(dir, name+"_query.fvecs"))
	if err != nil {
		return nil, fmt.Errorf("recall: loading queries for %q: %w", name, err)
	}
	if corpusDim != queryDim {
		return nil, fmt.Errorf("recall: corpus dim %d does not match query dim %d for %q", corpusDim, queryDim, name)
	}
	return &Dataset{
		Name:    name,
		Dim:     corpusDim,
		Corpus:  corpus,
		Queries: queries,
		Metric:  readMetric(filepath.Join(dir, name+".metric")),
	}, nil
}

func readMetric(path string) Metric {
	data, err := os.ReadFile(path)
	if err != nil {
		return MetricInnerProduct
	}
	if strings.TrimSpace(string(data)) == "cosine" {
		return MetricCosine
	}
	return MetricInnerProduct
}

// readFvecs reads a .fvecs file: a sequence of records, each a little-endian
// int32 dimension followed by that many little-endian float32 values.
func readFvecs(path string) ([]float32, int, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, 0, err
	}
	defer f.Close()
	r := bufio.NewReaderSize(f, 1<<20)

	var out []float32
	dim := -1
	var dimBuf [4]byte
	for {
		if _, err := io.ReadFull(r, dimBuf[:]); err != nil {
			if err == io.EOF {
				break
			}
			return nil, 0, err
		}
		rowDim := int(binary.LittleEndian.Uint32(dimBuf[:]))
		if dim == -1 {
			dim = rowDim
		} else if rowDim != dim {
			return nil, 0, fmt.Errorf("recall: inconsistent row dimension %d, want %d", rowDim, dim)
		}
		row := make([]byte, rowDim*4)
		if _, err := io.ReadFull(r, row); err != nil {
			return nil, 0, err
		}
		for i := 0; i < rowDim; i++ {
			bits := binary.LittleEndian.Uint32(row[i*4 : i*4+4])
			out = append(out, math.Float32frombits(bits))
		}
	}
	if dim <= 0 {
		return nil, 0, fmt.Errorf("recall: empty fvecs file %s", path)
	}
	return out, dim, nil
}

// Synthetic builds a deterministic clustered dataset for CI. It mixes
// isotropic Gaussian clusters on the unit sphere so the recall number is
// stable and non-trivial, unlike a uniform random corpus.
func Synthetic(dim, corpusCount, queryCount, clusters int, seed int64) *Dataset {
	if clusters < 1 {
		clusters = 1
	}
	rng := rand.New(rand.NewSource(seed))
	centers := make([][]float32, clusters)
	for c := range centers {
		centers[c] = randomUnitVector(dim, rng)
	}
	const noiseScale = 0.35

	gen := func(count int) []float32 {
		out := make([]float32, count*dim)
		for i := 0; i < count; i++ {
			center := centers[rng.Intn(clusters)]
			row := out[i*dim : (i+1)*dim]
			var normSq float64
			for j := 0; j < dim; j++ {
				v := center[j] + float32(rng.NormFloat64())*noiseScale
				row[j] = v
				normSq += float64(v) * float64(v)
			}
			if normSq > 1e-30 {
				scale := float32(1.0 / math.Sqrt(normSq))
				for j := 0; j < dim; j++ {
					row[j] *= scale
				}
			}
		}
		return out
	}

	return &Dataset{
		Name:    "synthetic",
		Dim:     dim,
		Corpus:  gen(corpusCount),
		Queries: gen(queryCount),
		Metric:  MetricInnerProduct,
	}
}

func randomUnitVector(dim int, rng *rand.Rand) []float32 {
	v := make([]float32, dim)
	var normSq float64
	for i := range v {
		g := float32(rng.NormFloat64())
		v[i] = g
		normSq += float64(g) * float64(g)
	}
	scale := float32(1.0 / math.Sqrt(normSq))
	for i := range v {
		v[i] *= scale
	}
	return v
}
