package recall

import (
	"testing"

	"m31labs.dev/turboquant"
)

// BenchmarkRecallEncodeSynthetic records IP quantizer encode throughput on a
// synthetic corpus. Phase 5 (batched encode) targets a 3x or better rise
// over this baseline.
func BenchmarkRecallEncodeSynthetic(b *testing.B) {
	d := Synthetic(1536, 1024, 1, 8, 42)
	q := turboquant.NewIPWithSeed(d.Dim, 4, 42)
	qx := turboquant.AllocIPQuantized(d.Dim, 4)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for c := 0; c < d.CorpusCount(); c++ {
			row := d.Corpus[c*d.Dim : (c+1)*d.Dim]
			q.QuantizeTo(&qx, row)
		}
	}
}
