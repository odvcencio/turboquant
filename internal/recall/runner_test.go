package recall

import "testing"

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
