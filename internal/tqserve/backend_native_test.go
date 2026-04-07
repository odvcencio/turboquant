package tqserve

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestNativeBackendTracksCapacityAndCheckpointState(t *testing.T) {
	backend, err := NewNativeBackend(NativeBackendConfig{
		Name:             "native",
		ModelIDs:         []string{"TurboQuant-Local-Executor"},
		Accelerator:      "cuda",
		Device:           "RTX 4090",
		TotalMemoryBytes: 24 << 30,
		WeightsBytes:     10 << 30,
		MaxSessions:      4,
		KeyDim:           16,
		KeyBits:          3,
		ValueDim:         16,
		ValueBits:        2,
		PageCapacity:     64,
		Seed:             42,
	})
	if err != nil {
		t.Fatalf("NewNativeBackend: %v", err)
	}
	resp, err := backend.ChatCompletions(context.Background(), RequestEnvelope{Model: "TurboQuant-Local-Executor", SessionID: "sess-native"}, []byte(`{"model":"TurboQuant-Local-Executor","messages":[{"role":"user","content":"hello native runtime"}]}`))
	if err != nil {
		t.Fatalf("ChatCompletions: %v", err)
	}
	defer resp.Body.Close()
	status := backend.Status(context.Background())
	if !status.Ready || status.Capacity == nil {
		t.Fatalf("status = %+v", status)
	}
	if status.Capacity.ActiveSessions != 1 || status.Capacity.KVCacheBytes == 0 {
		t.Fatalf("capacity = %+v", *status.Capacity)
	}
	state, err := backend.CaptureSessionState(context.Background(), "sess-native")
	if err != nil {
		t.Fatalf("CaptureSessionState: %v", err)
	}
	var payload map[string]any
	if err := json.Unmarshal(state, &payload); err != nil {
		t.Fatalf("Unmarshal state: %v", err)
	}
	if payload["model"] != "TurboQuant-Local-Executor" {
		t.Fatalf("state = %+v", payload)
	}
	if _, ok := payload["page"].(string); !ok {
		t.Fatalf("state missing page payload = %+v", payload)
	}
	restoreBackend, err := NewNativeBackend(NativeBackendConfig{
		Name:             "native",
		ModelIDs:         []string{"TurboQuant-Local-Executor"},
		Accelerator:      "cuda",
		Device:           "RTX 4090",
		TotalMemoryBytes: 24 << 30,
		WeightsBytes:     10 << 30,
		MaxSessions:      4,
		KeyDim:           16,
		KeyBits:          3,
		ValueDim:         16,
		ValueBits:        2,
		PageCapacity:     64,
		Seed:             42,
	})
	if err != nil {
		t.Fatalf("NewNativeBackend restore: %v", err)
	}
	if err := restoreBackend.RestoreSessionState(context.Background(), "sess-restored", state); err != nil {
		t.Fatalf("RestoreSessionState: %v", err)
	}
	restored := restoreBackend.Capacity(context.Background())
	if restored.ActiveSessions != 1 || restored.KVCacheBytes == 0 {
		t.Fatalf("restored capacity = %+v", restored)
	}
	restoredResp, err := restoreBackend.ChatCompletions(context.Background(), RequestEnvelope{Model: "TurboQuant-Local-Executor", SessionID: "sess-restored"}, []byte(`{"model":"TurboQuant-Local-Executor","messages":[{"role":"user","content":"continued"}]}`))
	if err != nil {
		t.Fatalf("continued ChatCompletions: %v", err)
	}
	defer restoredResp.Body.Close()
	bodyBytes, _ := io.ReadAll(restoredResp.Body)
	if !strings.Contains(string(bodyBytes), "sess-restored") {
		t.Fatalf("response = %s", bodyBytes)
	}
}

func TestNativeBackendRetrievesPriorTurns(t *testing.T) {
	backend, err := NewNativeBackend(NativeBackendConfig{
		Name:             "native",
		ModelIDs:         []string{"TurboQuant-Local-Executor"},
		Accelerator:      "cuda",
		Device:           "RTX 4090",
		TotalMemoryBytes: 24 << 30,
		WeightsBytes:     10 << 30,
		MaxSessions:      4,
		KeyDim:           32,
		KeyBits:          3,
		ValueDim:         32,
		ValueBits:        2,
		PageCapacity:     64,
		Seed:             42,
	})
	if err != nil {
		t.Fatalf("NewNativeBackend: %v", err)
	}
	first, err := backend.ChatCompletions(context.Background(), RequestEnvelope{Model: "TurboQuant-Local-Executor", SessionID: "sess-memory"}, []byte(`{"model":"TurboQuant-Local-Executor","messages":[{"role":"user","content":"remember that the deployment region is us-west-2"}]}`))
	if err != nil {
		t.Fatalf("first ChatCompletions: %v", err)
	}
	first.Body.Close()
	second, err := backend.ChatCompletions(context.Background(), RequestEnvelope{Model: "TurboQuant-Local-Executor", SessionID: "sess-memory"}, []byte(`{"model":"TurboQuant-Local-Executor","messages":[{"role":"user","content":"what deployment region did I mention earlier?"}]}`))
	if err != nil {
		t.Fatalf("second ChatCompletions: %v", err)
	}
	defer second.Body.Close()
	bodyBytes, _ := io.ReadAll(second.Body)
	body := string(bodyBytes)
	if !strings.Contains(body, "Relevant memory:") {
		t.Fatalf("response missing retrieval section: %s", body)
	}
	if !strings.Contains(body, "deployment region is us-west-2") {
		t.Fatalf("response missing recalled content: %s", body)
	}
}

func TestNativeBackendDelegatesGroundedGeneration(t *testing.T) {
	var request map[string]any
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("Decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"The deployment region you mentioned earlier was us-west-2."},"finish_reason":"stop"}]}`)
	}))
	defer upstream.Close()

	executor, err := NewUpstreamNativeExecutor(upstream.URL+"/v1", "", "executor-model", "", nil)
	if err != nil {
		t.Fatalf("NewUpstreamNativeExecutor: %v", err)
	}
	backend, err := NewNativeBackend(NativeBackendConfig{
		Name:             "native",
		ModelIDs:         []string{"TurboQuant-Local-Executor"},
		Accelerator:      "cuda",
		Device:           "RTX 4090",
		TotalMemoryBytes: 24 << 30,
		WeightsBytes:     10 << 30,
		MaxSessions:      4,
		KeyDim:           64,
		KeyBits:          3,
		ValueDim:         64,
		ValueBits:        2,
		PageCapacity:     64,
		Seed:             42,
		Executor:         executor,
	})
	if err != nil {
		t.Fatalf("NewNativeBackend: %v", err)
	}

	first, err := backend.ChatCompletions(context.Background(), RequestEnvelope{Model: "TurboQuant-Local-Executor", SessionID: "sess-delegate"}, []byte(`{"model":"TurboQuant-Local-Executor","messages":[{"role":"user","content":"remember that the deployment region is us-west-2"}]}`))
	if err != nil {
		t.Fatalf("first ChatCompletions: %v", err)
	}
	first.Body.Close()
	second, err := backend.ChatCompletions(context.Background(), RequestEnvelope{Model: "TurboQuant-Local-Executor", SessionID: "sess-delegate"}, []byte(`{"model":"TurboQuant-Local-Executor","messages":[{"role":"user","content":"what deployment region did I mention earlier?"}]}`))
	if err != nil {
		t.Fatalf("second ChatCompletions: %v", err)
	}
	defer second.Body.Close()

	bodyBytes, _ := io.ReadAll(second.Body)
	body := string(bodyBytes)
	if !strings.Contains(body, "us-west-2") {
		t.Fatalf("response missing delegated text: %s", body)
	}
	if request["model"] != "executor-model" {
		t.Fatalf("executor model = %v want executor-model", request["model"])
	}
	messages, _ := request["messages"].([]any)
	if len(messages) < 3 {
		t.Fatalf("executor messages = %+v", request["messages"])
	}
	foundMemory := false
	for _, item := range messages {
		message, _ := item.(map[string]any)
		content, _ := message["content"].(string)
		if strings.Contains(content, "Retrieved session memory:") && strings.Contains(content, "deployment region is us-west-2") {
			foundMemory = true
			break
		}
	}
	if !foundMemory {
		t.Fatalf("executor request missing retrieved memory: %+v", request["messages"])
	}
}
