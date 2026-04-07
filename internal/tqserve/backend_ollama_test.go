package tqserve

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestOllamaModels(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/tags" {
			t.Fatalf("path = %q want /api/tags", r.URL.Path)
		}
		_, _ = io.WriteString(w, `{"models":[{"name":"qwen2.5:7b"},{"name":"llama3.2"}]}`)
	}))
	defer srv.Close()

	backend, err := NewOllamaBackend(srv.URL, "", nil)
	if err != nil {
		t.Fatalf("NewOllamaBackend: %v", err)
	}
	models, err := backend.Models(t.Context())
	if err != nil {
		t.Fatalf("Models: %v", err)
	}
	if len(models.Data) != 2 || models.Data[0].ID != "qwen2.5:7b" || models.Data[1].ID != "llama3.2" {
		t.Fatalf("models = %+v", models.Data)
	}
}

func TestOllamaChatCompletion(t *testing.T) {
	var request map[string]any
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/api/chat" {
			t.Fatalf("path = %q want /api/chat", r.URL.Path)
		}
		defer r.Body.Close()
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		_, _ = io.WriteString(w, `{"model":"qwen2.5:7b","message":{"role":"assistant","content":"hello"},"done":true,"done_reason":"stop"}`)
	}))
	defer srv.Close()

	backend, err := NewOllamaBackend(srv.URL, "", nil)
	if err != nil {
		t.Fatalf("NewOllamaBackend: %v", err)
	}
	resp, err := backend.ChatCompletions(t.Context(), RequestEnvelope{Model: "qwen2.5:7b"}, []byte(`{"model":"qwen2.5:7b","messages":[{"role":"user","content":"hi"}],"max_tokens":12}`))
	if err != nil {
		t.Fatalf("ChatCompletions: %v", err)
	}
	defer resp.Body.Close()

	options, _ := request["options"].(map[string]any)
	if got, want := request["model"], "qwen2.5:7b"; got != want {
		t.Fatalf("model = %v want %v", got, want)
	}
	if got, want := options["num_predict"], float64(12); got != want {
		t.Fatalf("num_predict = %v want %v", got, want)
	}

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	if !strings.Contains(string(body), `"chat.completion"`) || !strings.Contains(string(body), `"hello"`) {
		t.Fatalf("body = %s", body)
	}
}

func TestOllamaChatCompletionStream(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		_, _ = io.WriteString(w, "{\"model\":\"qwen2.5:7b\",\"message\":{\"role\":\"assistant\",\"content\":\"hel\"},\"done\":false}\n")
		_, _ = io.WriteString(w, "{\"model\":\"qwen2.5:7b\",\"message\":{\"role\":\"assistant\",\"content\":\"lo\"},\"done\":false}\n")
		_, _ = io.WriteString(w, "{\"model\":\"qwen2.5:7b\",\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true,\"done_reason\":\"stop\"}\n")
	}))
	defer srv.Close()

	backend, err := NewOllamaBackend(srv.URL, "", nil)
	if err != nil {
		t.Fatalf("NewOllamaBackend: %v", err)
	}
	resp, err := backend.ChatCompletions(t.Context(), RequestEnvelope{Model: "qwen2.5:7b", Stream: true}, []byte(`{"model":"qwen2.5:7b","messages":[{"role":"user","content":"hi"}],"stream":true}`))
	if err != nil {
		t.Fatalf("ChatCompletions: %v", err)
	}
	defer resp.Body.Close()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		t.Fatalf("ReadAll: %v", err)
	}
	text := string(body)
	if !strings.Contains(text, "chat.completion.chunk") || !strings.Contains(text, "[DONE]") || !strings.Contains(text, `"content":"hel"`) {
		t.Fatalf("body = %s", text)
	}
}
