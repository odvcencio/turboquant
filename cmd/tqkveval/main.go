package main

import (
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"os"
	"time"

	turboquant "github.com/odvcencio/turboquant"
)

type report struct {
	GeneratedAt string                                  `json:"generated_at"`
	Results     []turboquant.TransformerLayerEvalResult `json:"results"`
}

func main() {
	if err := runCLI(os.Args[1:], os.Stdout, os.Stderr); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func runCLI(args []string, stdout, stderr io.Writer) error {
	fs := flag.NewFlagSet("tqkveval", flag.ContinueOnError)
	fs.SetOutput(stderr)

	inputPath := fs.String("input", "", "path to a JSON TransformerLayerCapture or TransformerLayerCaptureFile")
	outputPath := fs.String("out", "", "optional path to write the JSON report")
	method := fs.String("method", turboquant.TransformerEvalMethodTurboQuant, "evaluation method: turboquant or uniform")
	keyBits := fs.Int("key-bits", 3, "key quantization bit width")
	valueBits := fs.Int("value-bits", 2, "value quantization bit width")
	topK := fs.Int("top-k", 32, "top-k keys per head for approximate attention")
	capacity := fs.Int("capacity", 0, "optional token capacity override for the quantized cache")
	seed := fs.Int64("seed", 42, "seed for deterministic quantizer construction")
	queryScale := fs.Float64("query-scale", 0, "multiplier applied to query dot products before softmax; default 0 uses the capture's query_scale or 1")
	tryGPU := fs.Bool("gpu", false, "try the experimental GPU path when available")

	if err := fs.Parse(args); err != nil {
		return err
	}
	if *inputPath == "" {
		return errors.New("tqkveval: --input is required")
	}

	results := make([]turboquant.TransformerLayerEvalResult, 0)
	err := turboquant.WalkTransformerLayerCapturesFile(*inputPath, func(sample turboquant.TransformerLayerCapture) error {
		result, err := turboquant.EvaluateTransformerLayerCapture(sample, turboquant.TransformerLayerEvalConfig{
			Method:     *method,
			KeyBits:    *keyBits,
			ValueBits:  *valueBits,
			TopK:       *topK,
			Capacity:   *capacity,
			Seed:       *seed,
			QueryScale: float32(*queryScale),
			TryGPU:     *tryGPU,
		})
		if err != nil {
			return err
		}
		results = append(results, result)
		return nil
	})
	if err != nil {
		return fmt.Errorf("tqkveval: decode input: %w", err)
	}

	payload := report{
		GeneratedAt: time.Now().UTC().Format(time.RFC3339Nano),
		Results:     results,
	}
	if *outputPath == "" {
		enc := json.NewEncoder(stdout)
		enc.SetIndent("", "  ")
		return enc.Encode(payload)
	}
	f, err := os.Create(*outputPath)
	if err != nil {
		return fmt.Errorf("tqkveval: write report: %w", err)
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	enc.SetIndent("", "  ")
	if err := enc.Encode(payload); err != nil {
		return fmt.Errorf("tqkveval: write report: %w", err)
	}
	_, err = fmt.Fprintf(stdout, "wrote %s\n", *outputPath)
	return err
}
