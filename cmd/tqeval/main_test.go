package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/odvcencio/turboquant/internal/tqserve"
)

func TestDecodePromptsBlankSeparated(t *testing.T) {
	prompts, err := decodePrompts([]byte("first prompt\n\nsecond prompt\n"))
	if err != nil {
		t.Fatalf("decodePrompts: %v", err)
	}
	if len(prompts) != 2 || prompts[0] != "first prompt" || prompts[1] != "second prompt" {
		t.Fatalf("prompts = %#v", prompts)
	}
}

func TestDecodePromptsJSONArray(t *testing.T) {
	prompts, err := decodePrompts([]byte(`["first","second"]`))
	if err != nil {
		t.Fatalf("decodePrompts: %v", err)
	}
	if len(prompts) != 2 || prompts[0] != "first" || prompts[1] != "second" {
		t.Fatalf("prompts = %#v", prompts)
	}
}

func TestRunTargetCapturesSessionAndStatus(t *testing.T) {
	const sessionHeader = "X-TQ-Session-ID"
	var seenSessionID string
	var seenAuth string
	chatCalls := 0
	statusCalls := 0
	sessionsCalls := 0

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/chat/completions":
			chatCalls++
			seenSessionID = r.Header.Get(sessionHeader)
			seenAuth = r.Header.Get("Authorization")
			var req chatRequest
			if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
				t.Fatalf("decode chat request: %v", err)
			}
			if req.Model != "local-native-inline" {
				t.Fatalf("model = %q", req.Model)
			}
			assistant := "reply"
			if len(req.Messages) > 2 {
				assistant = "reply again"
			}
			_ = json.NewEncoder(w).Encode(map[string]any{
				"choices": []map[string]any{
					{
						"message": map[string]any{
							"content": assistant,
						},
					},
				},
			})
		case "/v1/tq/status":
			statusCalls++
			_ = json.NewEncoder(w).Encode(statusSnapshot{
				OK:             true,
				Service:        "tqserve",
				Version:        "test",
				ActiveSessions: 1,
				BackendCount:   1,
				RouteCount:     1,
			})
		case "/v1/tq/sessions":
			sessionsCalls++
			_ = json.NewEncoder(w).Encode(sessionsPayload{
				Object:        "list",
				SessionHeader: sessionHeader,
				Data: []tqserve.SessionInfo{
					{ID: seenSessionID},
				},
			})
		default:
			http.NotFound(w, r)
		}
	}))
	defer server.Close()

	target := targetConfig{
		Name:          "turbo",
		BaseURL:       server.URL + "/v1",
		Model:         "local-native-inline",
		APIKey:        "sk-test",
		SessionHeader: sessionHeader,
		SessionID:     "sess-123",
		StatusURL:     server.URL + "/v1/tq/status",
		SessionsURL:   server.URL + "/v1/tq/sessions",
	}
	report, err := runTarget(server.Client(), target, []string{"first", "second"})
	if err != nil {
		t.Fatalf("runTarget: %v", err)
	}
	if chatCalls != 2 {
		t.Fatalf("chatCalls = %d want 2", chatCalls)
	}
	if statusCalls != 3 {
		t.Fatalf("statusCalls = %d want 3", statusCalls)
	}
	if sessionsCalls != 3 {
		t.Fatalf("sessionsCalls = %d want 3", sessionsCalls)
	}
	if seenSessionID != "sess-123" {
		t.Fatalf("session header = %q", seenSessionID)
	}
	if seenAuth != "Bearer sk-test" {
		t.Fatalf("authorization = %q", seenAuth)
	}
	if report.Summary.SuccessfulTurns != 2 || report.Summary.FailedTurns != 0 {
		t.Fatalf("summary = %+v", report.Summary)
	}
	if report.FinalStatus == nil || !report.FinalStatus.OK {
		t.Fatalf("final status = %+v", report.FinalStatus)
	}
	if report.FinalSessions == nil || len(report.FinalSessions.Data) != 1 {
		t.Fatalf("final sessions = %+v", report.FinalSessions)
	}
	if report.Turns[0].AssistantPreview != "reply" {
		t.Fatalf("assistant preview = %q", report.Turns[0].AssistantPreview)
	}
	if report.Turns[1].AssistantPreview != "reply again" {
		t.Fatalf("assistant preview = %q", report.Turns[1].AssistantPreview)
	}
}

func TestLoadConfigResolvesPromptFileRelativeToConfig(t *testing.T) {
	dir := t.TempDir()
	promptsPath := filepath.Join(dir, "prompts.txt")
	if err := os.WriteFile(promptsPath, []byte("first\n\nsecond\n"), 0o644); err != nil {
		t.Fatalf("write prompts: %v", err)
	}
	configPath := filepath.Join(dir, "tqeval.json")
	configJSON := `{"prompts_file":"prompts.txt","targets":[{"name":"baseline","base_url":"http://127.0.0.1:8081/v1","model":"model"}]}`
	if err := os.WriteFile(configPath, []byte(configJSON), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}
	cfg, err := loadConfig(configPath)
	if err != nil {
		t.Fatalf("loadConfig: %v", err)
	}
	if len(cfg.Prompts) != 2 || cfg.Prompts[0] != "first" || cfg.Prompts[1] != "second" {
		t.Fatalf("prompts = %#v", cfg.Prompts)
	}
}

func TestMainWritesReportFile(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path != "/v1/chat/completions" {
			http.NotFound(w, r)
			return
		}
		_, _ = io.Copy(io.Discard, r.Body)
		_ = json.NewEncoder(w).Encode(map[string]any{
			"choices": []map[string]any{
				{"message": map[string]any{"content": "ok"}},
			},
		})
	}))
	defer server.Close()

	dir := t.TempDir()
	configPath := filepath.Join(dir, "tqeval.json")
	outputPath := filepath.Join(dir, "report.json")
	configJSON := `{
	  "prompts": ["hello"],
	  "targets": [
	    {"name":"baseline","base_url":"` + server.URL + `/v1","model":"model"}
	  ]
	}`
	if err := os.WriteFile(configPath, []byte(configJSON), 0o644); err != nil {
		t.Fatalf("write config: %v", err)
	}

	origArgs := os.Args
	defer func() { os.Args = origArgs }()
	os.Args = []string{"tqeval", "--config", configPath, "--out", outputPath}
	main()

	data, err := os.ReadFile(outputPath)
	if err != nil {
		t.Fatalf("read report: %v", err)
	}
	if !strings.Contains(string(data), `"successful_turns": 1`) {
		t.Fatalf("report = %s", string(data))
	}
}
