//go:build (!js || !wasm) && !(linux && amd64 && cgo && cuda)

package turboquant

type GPUPreparedScorer struct {
	dim         int
	bitWidth    int
	mseBitWidth int
	count       int
}

type GPUPreparedQueryBatch struct{}

func newGPUPreparedScorer(q *IPQuantizer, data GPUPreparedData) (*GPUPreparedScorer, error) {
	count, err := ValidateGPUPreparedData(q.dim, q.bitWidth, data)
	if err != nil {
		return nil, err
	}
	_ = count
	return nil, ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) Len() int {
	if s == nil {
		return 0
	}
	return s.count
}

func (s *GPUPreparedScorer) ScorePreparedQuery(pq PreparedQuery) ([]float32, error) {
	return nil, ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) ScorePreparedQueryTo(dst []float32, pq PreparedQuery) error {
	return ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) ScorePreparedQueryToTrusted(dst []float32, pq PreparedQuery) error {
	return ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) ScorePreparedQueryTopK(pq PreparedQuery, k int) ([]uint32, []float32, error) {
	return nil, nil, ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) ScorePreparedQueryTopKTo(indices []uint32, scores []float32, pq PreparedQuery) error {
	return ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) ScorePreparedQueryTopKToTrusted(indices []uint32, scores []float32, pq PreparedQuery) error {
	return ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) ScorePreparedQueriesTopK(pqs []PreparedQuery, k int) ([]uint32, []float32, error) {
	return nil, nil, ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) ScorePreparedQueriesTopKTo(indices []uint32, scores []float32, pqs []PreparedQuery, k int) error {
	return ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) ScorePreparedQueriesTopKToTrusted(indices []uint32, scores []float32, pqs []PreparedQuery, k int) error {
	return ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) UploadPreparedQueries(pqs []PreparedQuery) (*GPUPreparedQueryBatch, error) {
	return nil, ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) UploadPreparedQueriesTrusted(pqs []PreparedQuery) (*GPUPreparedQueryBatch, error) {
	return nil, ErrGPUBackendUnavailable
}

func (s *GPUPreparedScorer) Close() error {
	return nil
}

func (b *GPUPreparedQueryBatch) Len() int {
	return 0
}

func (b *GPUPreparedQueryBatch) ScoreTopK(k int) ([]uint32, []float32, error) {
	return nil, nil, ErrGPUBackendUnavailable
}

func (b *GPUPreparedQueryBatch) ScoreTopKTo(indices []uint32, scores []float32, k int) error {
	return ErrGPUBackendUnavailable
}

func (b *GPUPreparedQueryBatch) Close() error {
	return nil
}
