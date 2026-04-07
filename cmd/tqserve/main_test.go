package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestParseModelMap(t *testing.T) {
	got := parseModelMap("public-a=backend/a, public-b")
	if got["public-a"] != "backend/a" || got["public-b"] != "public-b" || len(got) != 2 {
		t.Fatalf("parseModelMap = %#v", got)
	}
}

func TestBuildServerConfigFromFile(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tqserve.json")
	if err := os.WriteFile(path, []byte(`{
		"listen": ":9090",
		"api_keys": ["sk-local"],
		"default_owner": "turboquant",
		"session_header": "X-Test-Session",
		"session_idle_ttl": "45m",
		"backends": {
			"ollama": {"type":"ollama","base_url":"http://127.0.0.1:11434"}
		},
		"models": [
			{"name":"local-chat","backend":"ollama","target":"qwen2.5:7b"}
		]
	}`), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	cfg, listen, closers, err := buildServerConfigFromFile(path)
	if err != nil {
		t.Fatalf("buildServerConfigFromFile: %v", err)
	}
	if len(closers) != 0 {
		t.Fatalf("closers = %d want 0", len(closers))
	}
	if listen != ":9090" {
		t.Fatalf("listen = %q want %q", listen, ":9090")
	}
	if len(cfg.Backends) != 1 || len(cfg.Routes) != 1 {
		t.Fatalf("cfg = %#v", cfg)
	}
	if cfg.SessionHeader != "X-Test-Session" || cfg.SessionIdleTTL.String() != "45m0s" {
		t.Fatalf("session config = (%q, %s)", cfg.SessionHeader, cfg.SessionIdleTTL)
	}
	if cfg.Routes[0].PublicModel != "local-chat" || cfg.Routes[0].BackendName != "ollama" || cfg.Routes[0].BackendModel != "qwen2.5:7b" {
		t.Fatalf("routes = %#v", cfg.Routes)
	}
}

func TestFileBackendCapacityDecode(t *testing.T) {
	var spec fileBackend
	if err := json.Unmarshal([]byte(`{
		"type":"managed_upstream",
		"base_url":"http://127.0.0.1:8082/v1",
		"command":"./local-runtime",
		"status_url":"http://127.0.0.1:8082/v1/tq/status",
		"checkpoint_url":"http://127.0.0.1:8082/v1/tq/checkpoints",
		"restore_url":"http://127.0.0.1:8082/v1/tq/checkpoints/restore",
		"capacity":{"accelerator":"cuda","device":"RTX 4090","max_sessions":8}
	}`), &spec); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if spec.Capacity.Accelerator != "cuda" || spec.Capacity.Device != "RTX 4090" || spec.Capacity.MaxSessions != 8 {
		t.Fatalf("capacity = %+v", spec.Capacity)
	}
	if spec.StatusURL == "" || spec.CheckpointURL == "" || spec.RestoreURL == "" {
		t.Fatalf("control URLs = (%q, %q, %q)", spec.StatusURL, spec.CheckpointURL, spec.RestoreURL)
	}
}

func TestFileBackendNativeExecutorDecode(t *testing.T) {
	var spec fileBackend
	if err := json.Unmarshal([]byte(`{
		"type":"native",
		"model_ids":["TurboQuant-Local-Executor"],
		"executor_backend":"ollama",
		"executor_type":"ollama",
		"executor_base_url":"http://127.0.0.1:11434",
		"executor_model":"qwen2.5:7b",
		"executor_system_prompt":"Use memory."
	}`), &spec); err != nil {
		t.Fatalf("Unmarshal: %v", err)
	}
	if spec.ExecutorBackend != "ollama" {
		t.Fatalf("executor backend = %q want %q", spec.ExecutorBackend, "ollama")
	}
	if spec.ExecutorType != "ollama" || spec.ExecutorBaseURL != "http://127.0.0.1:11434" || spec.ExecutorModel != "qwen2.5:7b" {
		t.Fatalf("executor spec = %+v", spec)
	}
	if spec.ExecutorPrompt != "Use memory." {
		t.Fatalf("executor prompt = %q", spec.ExecutorPrompt)
	}
}

func TestBuildServerConfigFromFileNativeExecutorBackendReference(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tqserve.json")
	if err := os.WriteFile(path, []byte(`{
		"listen": ":9090",
		"api_keys": ["sk-local"],
		"backends": {
			"ollama": {"type":"ollama","base_url":"http://127.0.0.1:11434"},
			"native-local": {
				"type":"native",
				"model_ids":["TurboQuant-Local-Executor"],
				"executor_backend":"ollama",
				"executor_model":"qwen2.5:7b"
			}
		},
		"models": [
			{"name":"local-native","backend":"native-local","target":"TurboQuant-Local-Executor"}
		]
	}`), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	cfg, _, closers, err := buildServerConfigFromFile(path)
	if err != nil {
		t.Fatalf("buildServerConfigFromFile: %v", err)
	}
	if len(closers) != 0 {
		t.Fatalf("closers = %d want 0", len(closers))
	}
	if len(cfg.Backends) != 2 {
		t.Fatalf("backends = %d want 2", len(cfg.Backends))
	}
	if len(cfg.Routes) != 1 || cfg.Routes[0].BackendName != "native-local" {
		t.Fatalf("routes = %#v", cfg.Routes)
	}
}

func TestBuildServerConfigFromFileRejectsMissingNativeExecutorBackend(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "tqserve.json")
	if err := os.WriteFile(path, []byte(`{
		"listen": ":9090",
		"api_keys": ["sk-local"],
		"backends": {
			"native-local": {
				"type":"native",
				"model_ids":["TurboQuant-Local-Executor"],
				"executor_backend":"missing-backend",
				"executor_model":"qwen2.5:7b"
			}
		},
		"models": [
			{"name":"local-native","backend":"native-local","target":"TurboQuant-Local-Executor"}
		]
	}`), 0o600); err != nil {
		t.Fatalf("WriteFile: %v", err)
	}

	_, _, _, err := buildServerConfigFromFile(path)
	if err == nil || !strings.Contains(err.Error(), `unknown backend "missing-backend"`) {
		t.Fatalf("err = %v", err)
	}
}
