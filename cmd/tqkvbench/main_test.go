package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
)

func TestParsePerplexity(t *testing.T) {
	text := `
llama_kv_cache_init:       CUDA0 KV buffer size =   408.00 MiB
llama_kv_cache_init: size =  768.00 MiB ( 65536 cells,  32 layers,  1/1 seqs), K (q8_0):  384.00 MiB, V (q4_0):  384.00 MiB
[1]15.2701,[2]5.4007
Final estimate: PPL = 5.4007 +/- 0.67339
`
	ppl, err := parsePerplexity(text)
	if err != nil {
		t.Fatalf("parsePerplexity: %v", err)
	}
	if ppl.Mean != 5.4007 || ppl.Stddev != 0.67339 {
		t.Fatalf("unexpected ppl %+v", ppl)
	}

	kv := parseKVStats(text)
	if kv == nil {
		t.Fatal("expected kv stats")
	}
	if kv.TotalMiB != 768 || kv.KeyMiB != 384 || kv.ValueMiB != 384 || kv.BufferMiB != 408 {
		t.Fatalf("unexpected kv stats %+v", kv)
	}
}

func TestParseBenchSamples(t *testing.T) {
	text := `[
  {
    "build_commit": "8cf427ff",
    "build_number": 5163,
    "backends": "CUDA",
    "model_filename": "model.gguf",
    "type_k": "f16",
    "type_v": "q4_0",
    "n_prompt": 512,
    "n_gen": 0,
    "n_depth": 8192,
    "avg_ts": 7100.002165,
    "stddev_ts": 140.341520,
    "samples_ts": [6863.1, 7147.55]
  },
  {
    "build_commit": "8cf427ff",
    "build_number": 5163,
    "backends": "CUDA",
    "model_filename": "model.gguf",
    "type_k": "f16",
    "type_v": "q4_0",
    "n_prompt": 0,
    "n_gen": 128,
    "n_depth": 8192,
    "avg_ts": 118.881588,
    "stddev_ts": 1.041811
  }
]`
	samples, err := parseBenchSamples(text)
	if err != nil {
		t.Fatalf("parseBenchSamples: %v", err)
	}
	if len(samples) != 2 {
		t.Fatalf("len(samples) = %d want 2", len(samples))
	}
	if samples[0].Kind != "pp" || samples[1].Kind != "tg" {
		t.Fatalf("unexpected kinds %+v", samples)
	}
}

func TestRunCLI(t *testing.T) {
	dir := t.TempDir()
	perplexityPath := filepath.Join(dir, "fake-perplexity.sh")
	benchPath := filepath.Join(dir, "fake-bench.sh")
	configPath := filepath.Join(dir, "tqkvbench.json")
	outputPath := filepath.Join(dir, "report.json")
	datasetPath := filepath.Join(dir, "wiki.test.raw")
	modelPath := filepath.Join(dir, "model.gguf")

	writeExecutable(t, perplexityPath, `#!/bin/sh
echo "llama_kv_cache_init:       CUDA0 KV buffer size =   408.00 MiB" >&2
echo "llama_kv_cache_init: size =  768.00 MiB ( 65536 cells,  32 layers,  1/1 seqs), K (q8_0):  384.00 MiB, V (q4_0):  384.00 MiB" >&2
echo "Final estimate: PPL = 5.4007 +/- 0.67339"
`)
	writeExecutable(t, benchPath, `#!/bin/sh
cat <<'JSON'
[
  {
    "build_commit": "8cf427ff",
    "build_number": 5163,
    "backends": "CUDA",
    "model_filename": "model.gguf",
    "type_k": "q8_0",
    "type_v": "q4_0",
    "n_prompt": 512,
    "n_gen": 0,
    "n_depth": 8192,
    "avg_ts": 7100.002165,
    "stddev_ts": 140.341520
  },
  {
    "build_commit": "8cf427ff",
    "build_number": 5163,
    "backends": "CUDA",
    "model_filename": "model.gguf",
    "type_k": "q8_0",
    "type_v": "q4_0",
    "n_prompt": 0,
    "n_gen": 128,
    "n_depth": 8192,
    "avg_ts": 118.881588,
    "stddev_ts": 1.041811
  }
]
JSON
`)
	if err := os.WriteFile(datasetPath, []byte("test corpus"), 0o644); err != nil {
		t.Fatalf("write dataset: %v", err)
	}
	if err := os.WriteFile(modelPath, []byte("fake model"), 0o644); err != nil {
		t.Fatalf("write model: %v", err)
	}

	config := `{
  "runs": [
    {
      "name": "upstream",
      "model": "` + modelPath + `",
      "perplexity_bin": "` + perplexityPath + `",
      "bench_bin": "` + benchPath + `",
      "baseline": "q8_0/q4_0",
      "kv_types": [
        { "type_k": "q8_0", "type_v": "q4_0" }
      ],
      "perplexity": {
        "dataset": "` + datasetPath + `",
        "ctx_sizes": [8192],
        "batch_size": 512
      },
      "bench": {
        "n_prompt": [512],
        "n_gen": [128],
        "n_depth": [8192],
        "repetitions": 1
      }
    }
  ]
}`
	if err := os.WriteFile(configPath, []byte(config), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	var stdout strings.Builder
	var stderr strings.Builder
	if err := runCLI([]string{"--config", configPath, "--out", outputPath}, &stdout, &stderr); err != nil {
		t.Fatalf("runCLI: %v\nstderr=%s", err, stderr.String())
	}
	if !strings.Contains(stdout.String(), outputPath) {
		t.Fatalf("stdout = %q want output path", stdout.String())
	}

	payload, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	var rep report
	if err := json.Unmarshal(payload, &rep); err != nil {
		t.Fatalf("decode report: %v", err)
	}
	if len(rep.Runs) != 1 || len(rep.Runs[0].KVVariants) != 1 {
		t.Fatalf("unexpected report shape %+v", rep)
	}
	kv := rep.Runs[0].KVVariants[0]
	if len(kv.Perplexity) != 1 || kv.Perplexity[0].PPL == nil {
		t.Fatalf("missing perplexity results %+v", kv)
	}
	if kv.Bench == nil || len(kv.Bench.Samples) != 2 {
		t.Fatalf("missing bench results %+v", kv)
	}
}

func writeExecutable(t *testing.T, path, body string) {
	t.Helper()
	if runtime.GOOS == "windows" {
		t.Skip("shell-script test is not supported on windows")
	}
	if err := os.WriteFile(path, []byte(body), 0o755); err != nil {
		t.Fatalf("write executable %s: %v", path, err)
	}
}
