// Command tqrecall measures TurboQuant recall against exact search on a
// dataset and optionally gates continuous integration on a recall
// regression against a recorded baseline.
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strconv"
	"strings"

	"m31labs.dev/turboquant/internal/recall"
)

func main() {
	os.Exit(run(os.Args[1:]))
}

func run(args []string) int {
	fs := flag.NewFlagSet("tqrecall", flag.ContinueOnError)
	dataDir := fs.String("data", "", "directory holding datasets")
	datasetName := fs.String("dataset", "synthetic", `dataset name, or "synthetic"`)
	methodList := fs.String("method", "turboquant-ip,turboquant-mse", "comma-separated method list")
	bitsList := fs.String("bits", "4", "comma-separated bit-width list")
	kList := fs.String("k", "1,10", "comma-separated k list")
	corpusN := fs.Int("corpus", 0, "optional corpus subsample size")
	queriesN := fs.Int("queries", 0, "optional query subsample size")
	seed := fs.Int64("seed", 42, "random seed")
	outPath := fs.String("out", "", "output JSON path")
	baselinePath := fs.String("baseline", "", "optional baseline JSON to compare against")
	tolerance := fs.Float64("tolerance", 0.005, "maximum allowed absolute recall drop")
	if err := fs.Parse(args); err != nil {
		return 2
	}

	ks, err := parseInts(*kList)
	if err != nil {
		fmt.Fprintln(os.Stderr, "tqrecall:", err)
		return 2
	}
	bitWidths, err := parseInts(*bitsList)
	if err != nil {
		fmt.Fprintln(os.Stderr, "tqrecall:", err)
		return 2
	}
	methods := splitTrim(*methodList)

	d, err := loadOrBuildDataset(*datasetName, *dataDir, *seed)
	if err != nil {
		fmt.Fprintln(os.Stderr, "tqrecall:", err)
		return 1
	}
	subsampleDataset(d, *corpusN, *queriesN)

	maxK := 1
	for _, k := range ks {
		if k > maxK {
			maxK = k
		}
	}
	gt := recall.ComputeGroundTruth(d, maxK)

	reports := make([]*recall.Report, 0, len(methods)*len(bitWidths))
	for _, m := range methods {
		for _, b := range bitWidths {
			cfg := recall.Config{Method: m, BitWidth: b, Seed: *seed, K: ks}
			report, err := recall.Run(d, gt, cfg)
			if err != nil {
				fmt.Fprintln(os.Stderr, "tqrecall:", err)
				return 1
			}
			reports = append(reports, report)
		}
	}

	out, err := json.MarshalIndent(reports, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "tqrecall:", err)
		return 1
	}
	if *outPath != "" {
		if err := os.WriteFile(*outPath, out, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "tqrecall:", err)
			return 1
		}
	} else {
		fmt.Println(string(out))
	}

	if *baselinePath != "" {
		baselineData, err := os.ReadFile(*baselinePath)
		if err != nil {
			fmt.Fprintln(os.Stderr, "tqrecall:", err)
			return 1
		}
		var baseline []*recall.Report
		if err := json.Unmarshal(baselineData, &baseline); err != nil {
			fmt.Fprintln(os.Stderr, "tqrecall:", err)
			return 1
		}
		if !compareToBaseline(reports, baseline, *tolerance) {
			return 1
		}
	}

	return 0
}

func loadOrBuildDataset(name, dataDir string, seed int64) (*recall.Dataset, error) {
	if name == "" || name == "synthetic" {
		return recall.Synthetic(64, 4096, 256, 8, seed), nil
	}
	return recall.LoadDataset(dataDir, name)
}

func subsampleDataset(d *recall.Dataset, corpusN, queriesN int) {
	if corpusN > 0 && corpusN < d.CorpusCount() {
		d.Corpus = d.Corpus[:corpusN*d.Dim]
	}
	if queriesN > 0 && queriesN < d.QueryCount() {
		d.Queries = d.Queries[:queriesN*d.Dim]
	}
}

func compareToBaseline(reports, baseline []*recall.Report, tolerance float64) bool {
	baseIdx := make(map[string]*recall.Report, len(baseline))
	for _, b := range baseline {
		baseIdx[reportKey(b)] = b
	}

	ok := true
	for _, r := range reports {
		b, found := baseIdx[reportKey(r)]
		if !found {
			continue
		}
		for k, got := range r.RecallAtK {
			want, found := b.RecallAtK[k]
			if !found {
				continue
			}
			if want-got > tolerance {
				fmt.Fprintf(os.Stderr, "tqrecall: recall regression method=%s bits=%d k=%d: %.4f below baseline %.4f by more than tolerance %.4f\n",
					r.Config.Method, r.Config.BitWidth, k, got, want, tolerance)
				ok = false
			}
		}
	}
	return ok
}

func reportKey(r *recall.Report) string {
	return fmt.Sprintf("%s|%d|%s", r.Config.Method, r.Config.BitWidth, r.Dataset)
}

func parseInts(s string) ([]int, error) {
	parts := splitTrim(s)
	out := make([]int, 0, len(parts))
	for _, p := range parts {
		v, err := strconv.Atoi(p)
		if err != nil {
			return nil, fmt.Errorf("invalid integer %q: %w", p, err)
		}
		out = append(out, v)
	}
	return out, nil
}

func splitTrim(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}
