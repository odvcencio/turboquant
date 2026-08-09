package recall

import (
	"fmt"
	"math"
	"time"

	"m31labs.dev/turboquant"
)

// Config names one measured configuration.
type Config struct {
	Method   string // "turboquant-ip", "turboquant-mse", "pq"
	BitWidth int
	Seed     int64
	K        []int // report recall at each k
}

// Report holds one configuration's measurement.
type Report struct {
	Config         Config
	Dataset        string
	Dim            int
	CorpusCount    int
	QueryCount     int
	RecallAtK      map[int]float64
	BytesPerVector int
	EncodeNanos    int64
	QueryNanos     int64
}

// Run measures one configuration against a dataset.
func Run(d *Dataset, gt *GroundTruth, cfg Config) (*Report, error) {
	switch cfg.Method {
	case "turboquant-mse":
		return runTurboQuantMSE(d, gt, cfg)
	case "turboquant-ip":
		return runTurboQuantIP(d, gt, cfg)
	case "pq":
		return runPQ(d, gt, cfg)
	default:
		return nil, fmt.Errorf("recall: unknown method %q", cfg.Method)
	}
}

func runTurboQuantMSE(d *Dataset, gt *GroundTruth, cfg Config) (*Report, error) {
	dim := d.Dim
	corpusCount := d.CorpusCount()
	queryCount := d.QueryCount()

	q := turboquant.NewWithSeed(dim, cfg.BitWidth, cfg.Seed)
	rowBytes := turboquant.PackedSize(dim, cfg.BitWidth)

	packed := make([]byte, corpusCount*rowBytes)
	norms := make([]float32, corpusCount)

	encodeStart := time.Now()
	for c := 0; c < corpusCount; c++ {
		row := vectorForMetric(d.Corpus[c*dim:(c+1)*dim], d.Metric)
		norms[c] = q.QuantizeTo(packed[c*rowBytes:(c+1)*rowBytes], row)
	}
	encodeNanos := time.Since(encodeStart).Nanoseconds()

	maxK := maxInt(cfg.K)
	approx := make([]uint32, queryCount*maxK)
	scores := make([]float32, corpusCount)
	dstScores := make([]float32, maxK)

	queryStart := time.Now()
	for i := 0; i < queryCount; i++ {
		query := vectorForMetric(d.Queries[i*dim:(i+1)*dim], d.Metric)
		for c := 0; c < corpusCount; c++ {
			scores[c] = q.InnerProduct(packed[c*rowBytes:(c+1)*rowBytes], norms[c], query)
		}
		TopK(approx[i*maxK:(i+1)*maxK], dstScores, scores)
	}
	queryNanos := time.Since(queryStart).Nanoseconds()

	return buildReport(d, gt, cfg, approx, maxK, rowBytes, encodeNanos, queryNanos), nil
}

func runTurboQuantIP(d *Dataset, gt *GroundTruth, cfg Config) (*Report, error) {
	dim := d.Dim
	corpusCount := d.CorpusCount()
	queryCount := d.QueryCount()

	q := turboquant.NewIPWithSeed(dim, cfg.BitWidth, cfg.Seed)
	mseBytes, signBytes := turboquant.IPQuantizedSizes(dim, cfg.BitWidth)
	rowBytes := mseBytes + signBytes

	qxs := make([]turboquant.IPQuantized, corpusCount)
	encodeStart := time.Now()
	for c := 0; c < corpusCount; c++ {
		row := vectorForMetric(d.Corpus[c*dim:(c+1)*dim], d.Metric)
		qxs[c] = turboquant.AllocIPQuantized(dim, cfg.BitWidth)
		q.QuantizeTo(&qxs[c], row)
	}
	encodeNanos := time.Since(encodeStart).Nanoseconds()

	maxK := maxInt(cfg.K)
	approx := make([]uint32, queryCount*maxK)
	scores := make([]float32, corpusCount)
	dstScores := make([]float32, maxK)
	pq := q.AllocPreparedQuery()

	queryStart := time.Now()
	for i := 0; i < queryCount; i++ {
		query := vectorForMetric(d.Queries[i*dim:(i+1)*dim], d.Metric)
		q.PrepareQueryTo(&pq, query)
		for c := 0; c < corpusCount; c++ {
			scores[c] = q.InnerProductPrepared(qxs[c], pq)
		}
		TopK(approx[i*maxK:(i+1)*maxK], dstScores, scores)
	}
	queryNanos := time.Since(queryStart).Nanoseconds()

	return buildReport(d, gt, cfg, approx, maxK, rowBytes, encodeNanos, queryNanos), nil
}

func runPQ(d *Dataset, gt *GroundTruth, cfg Config) (*Report, error) {
	dim := d.Dim
	corpusCount := d.CorpusCount()
	queryCount := d.QueryCount()

	params := DefaultPQParams(dim, cfg.Seed)
	cb := TrainPQ(d.Corpus, dim, corpusCount, params)
	rowBytes := cb.BytesPerVector()

	codes := make([]byte, corpusCount*rowBytes)
	encodeStart := time.Now()
	for c := 0; c < corpusCount; c++ {
		row := vectorForMetric(d.Corpus[c*dim:(c+1)*dim], d.Metric)
		cb.Encode(codes[c*rowBytes:(c+1)*rowBytes], row)
	}
	encodeNanos := time.Since(encodeStart).Nanoseconds()

	maxK := maxInt(cfg.K)
	approx := make([]uint32, queryCount*maxK)
	scores := make([]float32, corpusCount)
	dstScores := make([]float32, maxK)
	lut := make([]float32, cb.Subvectors*cb.Levels)

	queryStart := time.Now()
	for i := 0; i < queryCount; i++ {
		query := vectorForMetric(d.Queries[i*dim:(i+1)*dim], d.Metric)
		cb.BuildQueryLUT(lut, query)
		for c := 0; c < corpusCount; c++ {
			scores[c] = cb.Score(codes[c*rowBytes:(c+1)*rowBytes], lut)
		}
		TopK(approx[i*maxK:(i+1)*maxK], dstScores, scores)
	}
	queryNanos := time.Since(queryStart).Nanoseconds()

	return buildReport(d, gt, cfg, approx, maxK, rowBytes, encodeNanos, queryNanos), nil
}

func buildReport(d *Dataset, gt *GroundTruth, cfg Config, approx []uint32, maxK, bytesPerVector int, encodeNanos, queryNanos int64) *Report {
	queryCount := d.QueryCount()
	report := &Report{
		Config:         cfg,
		Dataset:        d.Name,
		Dim:            d.Dim,
		CorpusCount:    d.CorpusCount(),
		QueryCount:     queryCount,
		RecallAtK:      make(map[int]float64, len(cfg.K)),
		BytesPerVector: bytesPerVector,
		EncodeNanos:    encodeNanos,
		QueryNanos:     queryNanos,
	}
	for _, k := range cfg.K {
		if k <= 0 || k > maxK {
			continue
		}
		sub := make([]uint32, queryCount*k)
		for q := 0; q < queryCount; q++ {
			copy(sub[q*k:(q+1)*k], approx[q*maxK:q*maxK+k])
		}
		report.RecallAtK[k] = Recall1AtK(gt, sub, k)
	}
	return report
}

func maxInt(vs []int) int {
	m := 1
	for _, v := range vs {
		if v > m {
			m = v
		}
	}
	return m
}

// vectorForMetric returns the vector TurboQuant should encode or query with
// for the given metric. Cosine similarity normalizes both the corpus and the
// query to unit norm so that the raw inner-product estimator TurboQuant
// exposes approximates cosine similarity directly.
func vectorForMetric(v []float32, metric Metric) []float32 {
	if metric != MetricCosine {
		return v
	}
	var sumSq float64
	for _, x := range v {
		sumSq += float64(x) * float64(x)
	}
	norm := math.Sqrt(sumSq)
	out := make([]float32, len(v))
	if norm < 1e-12 {
		copy(out, v)
		return out
	}
	scale := float32(1.0 / norm)
	for i, x := range v {
		out[i] = x * scale
	}
	return out
}
