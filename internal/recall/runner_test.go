package recall

import "testing"

// raceEnabled is set to true in race_test.go when the race detector is
// active, so slow statistical sweeps can skip themselves the same way the
// root package's tests do.
var raceEnabled bool

func TestPQBaselineSanity(t *testing.T) {
	d := Synthetic(32, 2000, 100, 6, 3)
	gt := ComputeGroundTruth(d, 10)
	cfg := Config{Method: "pq", Seed: 3, K: []int{1, 10}}
	report, err := Run(d, gt, cfg)
	if err != nil {
		t.Fatalf("Run: %v", err)
	}
	for k, recallVal := range report.RecallAtK {
		if recallVal < 0 || recallVal > 1.0 {
			t.Fatalf("k=%d: recall %.4f out of range [0,1]", k, recallVal)
		}
	}
	if report.BytesPerVector <= 0 {
		t.Fatalf("BytesPerVector = %d want > 0", report.BytesPerVector)
	}
}

func TestRunUnknownMethod(t *testing.T) {
	d := Synthetic(8, 32, 4, 2, 1)
	gt := ComputeGroundTruth(d, 4)
	_, err := Run(d, gt, Config{Method: "bogus", K: []int{1}})
	if err == nil {
		t.Fatal("expected error for unknown method")
	}
}

// TestMSERankingBeatsIPRankingAtEqualBits locks in an evaluation finding:
// for approximate-nearest-neighbor ranking at a fixed storage budget, the
// plain MSE-optimal Quantizer's InnerProduct score gives higher recall than
// the dedicated IPQuantizer, across bit widths and dimensions. IPQuantizer
// spends part of its bit budget on a QJL residual to correct inner-product
// bias -- valuable when the caller needs an unbiased absolute-scale
// estimate (for example attention logits), but it costs relative-ranking
// precision at the same total bits per vector. This test does not change
// any exported default: callers still choose Quantizer or IPQuantizer
// explicitly. It exists so a future change that erodes this gap gets
// caught, and to anchor the recommendation in README.md ("Choosing an
// estimator"): prefer Quantizer.InnerProduct for top-k search, reserve
// IPQuantizer for workloads that need the unbiasedness guarantee.
func TestMSERankingBeatsIPRankingAtEqualBits(t *testing.T) {
	if raceEnabled {
		t.Skip("skipping slow statistical sweep under race detector")
	}
	if testing.Short() {
		t.Skip("skipping slow recall sweep in -short mode")
	}
	for _, dim := range []int{64, 384, 1024} {
		d := Synthetic(dim, 1200, 80, 8, int64(1000+dim))
		gt := ComputeGroundTruth(d, 10)
		for _, bw := range []int{2, 3, 4} {
			mseReport, err := Run(d, gt, Config{Method: "turboquant-mse", BitWidth: bw, Seed: 11, K: []int{1, 10}})
			if err != nil {
				t.Fatalf("dim=%d bw=%d turboquant-mse: %v", dim, bw, err)
			}
			ipReport, err := Run(d, gt, Config{Method: "turboquant-ip", BitWidth: bw, Seed: 11, K: []int{1, 10}})
			if err != nil {
				t.Fatalf("dim=%d bw=%d turboquant-ip: %v", dim, bw, err)
			}
			if mseReport.BytesPerVector != ipReport.BytesPerVector {
				t.Fatalf("dim=%d bw=%d: bytes/vector mismatch mse=%d ip=%d, comparison is not apples-to-apples",
					dim, bw, mseReport.BytesPerVector, ipReport.BytesPerVector)
			}
			for _, k := range []int{1, 10} {
				mseRecall, ipRecall := mseReport.RecallAtK[k], ipReport.RecallAtK[k]
				if mseRecall < ipRecall {
					t.Errorf("dim=%d bw=%d k=%d: turboquant-mse recall %.4f is below turboquant-ip recall %.4f at equal storage; "+
						"the documented recommendation to prefer MSE ranking no longer holds",
						dim, bw, k, mseRecall, ipRecall)
				}
			}
		}
	}
}

func TestRunTurboQuantMSEAndIP(t *testing.T) {
	d := Synthetic(32, 300, 30, 5, 11)
	gt := ComputeGroundTruth(d, 5)

	for _, method := range []string{"turboquant-mse", "turboquant-ip"} {
		bw := 4
		if method == "turboquant-mse" {
			bw = 2
		}
		cfg := Config{Method: method, BitWidth: bw, Seed: 11, K: []int{1, 5}}
		report, err := Run(d, gt, cfg)
		if err != nil {
			t.Fatalf("%s: Run: %v", method, err)
		}
		if report.CorpusCount != 300 || report.QueryCount != 30 {
			t.Fatalf("%s: unexpected counts corpus=%d query=%d", method, report.CorpusCount, report.QueryCount)
		}
		if report.RecallAtK[5] < report.RecallAtK[1] {
			t.Fatalf("%s: recall@5 (%v) < recall@1 (%v)", method, report.RecallAtK[5], report.RecallAtK[1])
		}
		if report.BytesPerVector <= 0 {
			t.Fatalf("%s: BytesPerVector = %d want > 0", method, report.BytesPerVector)
		}
	}
}
