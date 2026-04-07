package turboquant

import (
	"math"
	"testing"
)

func TestKVCachePageMarshalBinaryRoundTrip(t *testing.T) {
	page := NewKVCachePageWithSeed(16, 3, 16, 2, 8, 42)
	for i := 0; i < 5; i++ {
		page.Append(testVector(16, float32(i+1)), testVector(16, float32(i+11)))
	}
	beforeStorage := page.StorageBytes()
	beforeLive := page.LiveBytes()
	data, err := page.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary: %v", err)
	}
	restored, err := UnmarshalKVCachePage(data)
	if err != nil {
		t.Fatalf("UnmarshalKVCachePage: %v", err)
	}
	if restored.Len() != page.Len() || restored.Cap() != page.Cap() {
		t.Fatalf("shape = (len=%d cap=%d) want (len=%d cap=%d)", restored.Len(), restored.Cap(), page.Len(), page.Cap())
	}
	if restored.StorageBytes() != beforeStorage || restored.LiveBytes() != beforeLive {
		t.Fatalf("bytes = (%d,%d) want (%d,%d)", restored.StorageBytes(), restored.LiveBytes(), beforeStorage, beforeLive)
	}
	query := testVector(16, 3.5)
	origPQ := page.PrepareQuery(query)
	restPQ := restored.PrepareQuery(query)
	origIdx, origScores := page.TopKPrepared(origPQ, 3)
	restIdx, restScores := restored.TopKPrepared(restPQ, 3)
	for i := range origIdx {
		if origIdx[i] != restIdx[i] || math.Abs(float64(origScores[i]-restScores[i])) > 1e-5 {
			t.Fatalf("topk[%d] = (%d,%f) want (%d,%f)", i, restIdx[i], restScores[i], origIdx[i], origScores[i])
		}
	}
	origOut := make([]float32, 16)
	restOut := make([]float32, 16)
	page.AttentionOutputPreparedInto(origOut, make([]uint32, 3), make([]float32, 3), origPQ)
	restored.AttentionOutputPreparedInto(restOut, make([]uint32, 3), make([]float32, 3), restPQ)
	for i := range origOut {
		if math.Abs(float64(origOut[i]-restOut[i])) > 1e-5 {
			t.Fatalf("attention[%d] = %f want %f", i, restOut[i], origOut[i])
		}
	}
}

func TestKVCachePageStorageBytesExceedsLiveBytesWithHeadroom(t *testing.T) {
	page := NewKVCachePageWithSeed(16, 3, 16, 2, 16, 42)
	page.Append(testVector(16, 1), testVector(16, 2))
	if page.StorageBytes() <= page.LiveBytes() {
		t.Fatalf("storage bytes = %d live bytes = %d", page.StorageBytes(), page.LiveBytes())
	}
}

func testVector(dim int, base float32) []float32 {
	out := make([]float32, dim)
	for i := range out {
		out[i] = base + float32(i)/10
	}
	return out
}
