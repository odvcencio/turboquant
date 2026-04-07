package turboquant

type kvGPUValueBackend interface {
	accumulateRotatedTo(dst []float32, indices []uint32, weights []float32) error
	accumulateRotatedBatchTo(dst []float32, indices []uint32, weights []float32, queryCount int) error
	Close() error
}
