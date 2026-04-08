package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/odvcencio/turboquant/internal/tqserve"
)

type config struct {
	Prompts        []string       `json:"prompts"`
	PromptsFile    string         `json:"prompts_file"`
	RequestTimeout string         `json:"request_timeout"`
	Targets        []targetConfig `json:"targets"`
}

type targetConfig struct {
	Name          string `json:"name"`
	BaseURL       string `json:"base_url"`
	Model         string `json:"model"`
	APIKey        string `json:"api_key"`
	SessionHeader string `json:"session_header"`
	SessionID     string `json:"session_id"`
	StatusURL     string `json:"status_url"`
	SessionsURL   string `json:"sessions_url"`
}

type report struct {
	CreatedAt   time.Time      `json:"created_at"`
	PromptCount int            `json:"prompt_count"`
	Targets     []targetReport `json:"targets"`
}

type targetReport struct {
	Name          string           `json:"name"`
	BaseURL       string           `json:"base_url"`
	Model         string           `json:"model"`
	SessionHeader string           `json:"session_header,omitempty"`
	SessionID     string           `json:"session_id,omitempty"`
	Turns         []turnReport     `json:"turns"`
	Summary       targetSummary    `json:"summary"`
	FinalStatus   *statusSnapshot  `json:"final_status,omitempty"`
	FinalSessions *sessionsPayload `json:"final_sessions,omitempty"`
}

type turnReport struct {
	Index             int              `json:"index"`
	DurationMS        float64          `json:"duration_ms"`
	StatusCode        int              `json:"status_code,omitempty"`
	ResponseBytes     int              `json:"response_bytes,omitempty"`
	PromptChars       int              `json:"prompt_chars"`
	PromptPreview     string           `json:"prompt_preview"`
	AssistantChars    int              `json:"assistant_chars,omitempty"`
	AssistantPreview  string           `json:"assistant_preview,omitempty"`
	Error             string           `json:"error,omitempty"`
	Status            *statusSnapshot  `json:"status,omitempty"`
	StatusError       string           `json:"status_error,omitempty"`
	Sessions          *sessionsPayload `json:"sessions,omitempty"`
	SessionsError     string           `json:"sessions_error,omitempty"`
	ConversationTurns int              `json:"conversation_turns"`
}

type targetSummary struct {
	SuccessfulTurns     int     `json:"successful_turns"`
	FailedTurns         int     `json:"failed_turns"`
	TotalDurationMS     float64 `json:"total_duration_ms"`
	MeanDurationMS      float64 `json:"mean_duration_ms"`
	P50DurationMS       float64 `json:"p50_duration_ms"`
	P95DurationMS       float64 `json:"p95_duration_ms"`
	TotalAssistantChars int     `json:"total_assistant_chars"`
}

type statusSnapshot struct {
	OK             bool                    `json:"ok"`
	Service        string                  `json:"service"`
	Version        string                  `json:"version"`
	ActiveSessions int                     `json:"active_sessions"`
	BackendCount   int                     `json:"backend_count"`
	RouteCount     int                     `json:"route_count"`
	Backends       []tqserve.BackendStatus `json:"backends"`
}

type sessionsPayload struct {
	Object        string                `json:"object"`
	Data          []tqserve.SessionInfo `json:"data"`
	SessionHeader string                `json:"session_header,omitempty"`
	IdleTTL       string                `json:"idle_ttl,omitempty"`
}

type chatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type chatRequest struct {
	Model    string        `json:"model"`
	Messages []chatMessage `json:"messages"`
	Stream   bool          `json:"stream"`
}

type chatResponse struct {
	Choices []struct {
		Message struct {
			Content any `json:"content"`
		} `json:"message"`
	} `json:"choices"`
}

func main() {
	configPath := flag.String("config", "", "path to tqeval JSON config")
	outputPath := flag.String("out", "", "optional output JSON file (default stdout)")
	pretty := flag.Bool("pretty", true, "pretty-print JSON output")
	flag.Parse()

	if strings.TrimSpace(*configPath) == "" {
		fmt.Fprintln(os.Stderr, "tqeval: --config is required")
		os.Exit(2)
	}

	cfg, err := loadConfig(*configPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	result, err := run(cfg)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}

	data, err := marshalReport(result, *pretty)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	if strings.TrimSpace(*outputPath) == "" {
		_, _ = os.Stdout.Write(data)
		if len(data) == 0 || data[len(data)-1] != '\n' {
			_, _ = os.Stdout.Write([]byte{'\n'})
		}
		return
	}
	if err := os.WriteFile(*outputPath, append(data, '\n'), 0o644); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func loadConfig(path string) (config, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return config{}, err
	}
	var cfg config
	if err := json.Unmarshal(data, &cfg); err != nil {
		return config{}, fmt.Errorf("tqeval: decode config: %w", err)
	}
	baseDir := filepath.Dir(path)
	if strings.TrimSpace(cfg.PromptsFile) != "" {
		prompts, err := loadPrompts(resolvePath(baseDir, cfg.PromptsFile))
		if err != nil {
			return config{}, err
		}
		cfg.Prompts = prompts
	}
	if len(cfg.Prompts) == 0 {
		return config{}, fmt.Errorf("tqeval: config must include prompts or prompts_file")
	}
	if len(cfg.Targets) == 0 {
		return config{}, fmt.Errorf("tqeval: config must include at least one target")
	}
	return cfg, nil
}

func resolvePath(baseDir, value string) string {
	value = strings.TrimSpace(value)
	if value == "" || filepath.IsAbs(value) {
		return value
	}
	return filepath.Join(baseDir, value)
}

func loadPrompts(path string) ([]string, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return decodePrompts(data)
}

func decodePrompts(data []byte) ([]string, error) {
	trimmed := bytes.TrimSpace(data)
	if len(trimmed) == 0 {
		return nil, fmt.Errorf("tqeval: prompt file is empty")
	}
	if trimmed[0] == '[' {
		var prompts []string
		if err := json.Unmarshal(trimmed, &prompts); err != nil {
			return nil, fmt.Errorf("tqeval: decode prompt array: %w", err)
		}
		return normalizePrompts(prompts), nil
	}
	parts := strings.Split(string(data), "\n\n")
	return normalizePrompts(parts), nil
}

func normalizePrompts(parts []string) []string {
	prompts := make([]string, 0, len(parts))
	for _, part := range parts {
		prompt := strings.TrimSpace(part)
		if prompt == "" {
			continue
		}
		prompts = append(prompts, prompt)
	}
	return prompts
}

func run(cfg config) (report, error) {
	timeout, err := time.ParseDuration(defaultString(cfg.RequestTimeout, "30s"))
	if err != nil {
		return report{}, fmt.Errorf("tqeval: invalid request_timeout: %w", err)
	}
	client := &http.Client{Timeout: timeout}
	out := report{
		CreatedAt:   time.Now().UTC(),
		PromptCount: len(cfg.Prompts),
		Targets:     make([]targetReport, 0, len(cfg.Targets)),
	}
	for _, target := range cfg.Targets {
		run, err := runTarget(client, target, cfg.Prompts)
		if err != nil {
			return report{}, err
		}
		out.Targets = append(out.Targets, run)
	}
	return out, nil
}

func runTarget(client *http.Client, target targetConfig, prompts []string) (targetReport, error) {
	target.Name = strings.TrimSpace(target.Name)
	target.BaseURL = strings.TrimRight(strings.TrimSpace(target.BaseURL), "/")
	target.Model = strings.TrimSpace(target.Model)
	target.SessionHeader = strings.TrimSpace(target.SessionHeader)
	if target.Name == "" {
		return targetReport{}, fmt.Errorf("tqeval: target name is required")
	}
	if target.BaseURL == "" {
		return targetReport{}, fmt.Errorf("tqeval: target %q base_url is required", target.Name)
	}
	if target.Model == "" {
		return targetReport{}, fmt.Errorf("tqeval: target %q model is required", target.Name)
	}
	report := targetReport{
		Name:          target.Name,
		BaseURL:       target.BaseURL,
		Model:         target.Model,
		SessionHeader: defaultString(target.SessionHeader, tqserve.DefaultSessionHeader),
		SessionID:     strings.TrimSpace(target.SessionID),
		Turns:         make([]turnReport, 0, len(prompts)),
	}
	if target.StatusURL == "" && target.SessionsURL == "" {
		report.SessionHeader = ""
		report.SessionID = ""
	} else if report.SessionID == "" {
		report.SessionID = fmt.Sprintf("tqeval-%s-%d", sanitizeName(target.Name), time.Now().UnixNano())
	}

	messages := make([]chatMessage, 0, len(prompts)*2)
	for idx, prompt := range prompts {
		messages = append(messages, chatMessage{Role: "user", Content: prompt})
		start := time.Now()
		assistant, statusCode, responseBytes, err := sendChatCompletion(client, target, report.SessionHeader, report.SessionID, messages)
		durationMS := float64(time.Since(start)) / float64(time.Millisecond)
		turn := turnReport{
			Index:             idx,
			DurationMS:        durationMS,
			StatusCode:        statusCode,
			ResponseBytes:     responseBytes,
			PromptChars:       len(prompt),
			PromptPreview:     summarizeText(prompt, 160),
			ConversationTurns: len(messages),
		}
		if err != nil {
			turn.Error = err.Error()
			report.Turns = append(report.Turns, turn)
			report.Summary = buildSummary(report.Turns)
			report.FinalStatus, _ = maybeFetchStatus(client, target)
			report.FinalSessions, _ = maybeFetchSessions(client, target)
			return report, nil
		}
		messages = append(messages, chatMessage{Role: "assistant", Content: assistant})
		turn.AssistantChars = len(assistant)
		turn.AssistantPreview = summarizeText(assistant, 160)
		turn.ConversationTurns = len(messages)
		if snapshot, err := maybeFetchStatus(client, target); err != nil {
			turn.StatusError = err.Error()
		} else {
			turn.Status = snapshot
		}
		if sessions, err := maybeFetchSessions(client, target); err != nil {
			turn.SessionsError = err.Error()
		} else {
			turn.Sessions = sessions
		}
		report.Turns = append(report.Turns, turn)
	}
	report.Summary = buildSummary(report.Turns)
	report.FinalStatus, _ = maybeFetchStatus(client, target)
	report.FinalSessions, _ = maybeFetchSessions(client, target)
	return report, nil
}

func sendChatCompletion(client *http.Client, target targetConfig, sessionHeader, sessionID string, messages []chatMessage) (string, int, int, error) {
	body, err := json.Marshal(chatRequest{
		Model:    target.Model,
		Messages: messages,
		Stream:   false,
	})
	if err != nil {
		return "", 0, 0, err
	}
	req, err := http.NewRequest(http.MethodPost, target.BaseURL+"/chat/completions", bytes.NewReader(body))
	if err != nil {
		return "", 0, 0, err
	}
	req.Header.Set("Content-Type", "application/json")
	if strings.TrimSpace(target.APIKey) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(target.APIKey))
	}
	if sessionHeader != "" && sessionID != "" {
		req.Header.Set(sessionHeader, sessionID)
	}
	resp, err := client.Do(req)
	if err != nil {
		return "", 0, 0, err
	}
	defer resp.Body.Close()
	payload, err := io.ReadAll(io.LimitReader(resp.Body, 8<<20))
	if err != nil {
		return "", resp.StatusCode, 0, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", resp.StatusCode, len(payload), fmt.Errorf("tqeval: target %q returned status %d: %s", target.Name, resp.StatusCode, summarizeText(string(payload), 240))
	}
	var decoded chatResponse
	if err := json.Unmarshal(payload, &decoded); err != nil {
		return "", resp.StatusCode, len(payload), fmt.Errorf("tqeval: decode chat response: %w", err)
	}
	if len(decoded.Choices) == 0 {
		return "", resp.StatusCode, len(payload), errors.New("tqeval: chat response did not contain choices")
	}
	text := flattenContent(decoded.Choices[0].Message.Content)
	return text, resp.StatusCode, len(payload), nil
}

func maybeFetchStatus(client *http.Client, target targetConfig) (*statusSnapshot, error) {
	url := strings.TrimSpace(target.StatusURL)
	if url == "" {
		return nil, nil
	}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(target.APIKey) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(target.APIKey))
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		payload, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, fmt.Errorf("tqeval: status request failed with %d: %s", resp.StatusCode, summarizeText(string(payload), 200))
	}
	var snapshot statusSnapshot
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&snapshot); err != nil {
		return nil, err
	}
	return &snapshot, nil
}

func maybeFetchSessions(client *http.Client, target targetConfig) (*sessionsPayload, error) {
	url := strings.TrimSpace(target.SessionsURL)
	if url == "" {
		return nil, nil
	}
	req, err := http.NewRequest(http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	if strings.TrimSpace(target.APIKey) != "" {
		req.Header.Set("Authorization", "Bearer "+strings.TrimSpace(target.APIKey))
	}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		payload, _ := io.ReadAll(io.LimitReader(resp.Body, 1<<20))
		return nil, fmt.Errorf("tqeval: sessions request failed with %d: %s", resp.StatusCode, summarizeText(string(payload), 200))
	}
	var payload sessionsPayload
	if err := json.NewDecoder(io.LimitReader(resp.Body, 1<<20)).Decode(&payload); err != nil {
		return nil, err
	}
	return &payload, nil
}

func buildSummary(turns []turnReport) targetSummary {
	summary := targetSummary{}
	durations := make([]float64, 0, len(turns))
	for _, turn := range turns {
		summary.TotalDurationMS += turn.DurationMS
		if turn.Error == "" {
			summary.SuccessfulTurns++
			summary.TotalAssistantChars += turn.AssistantChars
			durations = append(durations, turn.DurationMS)
		} else {
			summary.FailedTurns++
		}
	}
	if len(turns) != 0 {
		summary.MeanDurationMS = summary.TotalDurationMS / float64(len(turns))
	}
	if len(durations) != 0 {
		slices.Sort(durations)
		summary.P50DurationMS = percentile(durations, 0.50)
		summary.P95DurationMS = percentile(durations, 0.95)
	}
	return summary
}

func percentile(values []float64, p float64) float64 {
	if len(values) == 0 {
		return 0
	}
	if len(values) == 1 {
		return values[0]
	}
	index := int(float64(len(values)-1) * p)
	if index < 0 {
		index = 0
	}
	if index >= len(values) {
		index = len(values) - 1
	}
	return values[index]
}

func marshalReport(report report, pretty bool) ([]byte, error) {
	if pretty {
		return json.MarshalIndent(report, "", "  ")
	}
	return json.Marshal(report)
}

func flattenContent(content any) string {
	switch value := content.(type) {
	case string:
		return strings.TrimSpace(value)
	case []any:
		parts := make([]string, 0, len(value))
		for _, item := range value {
			part, ok := item.(map[string]any)
			if !ok {
				continue
			}
			text := strings.TrimSpace(asString(part["text"]))
			if text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

func asString(value any) string {
	switch typed := value.(type) {
	case string:
		return typed
	default:
		return ""
	}
}

func summarizeText(value string, limit int) string {
	value = strings.Join(strings.Fields(strings.TrimSpace(value)), " ")
	if limit <= 0 || len(value) <= limit {
		return value
	}
	if limit <= 3 {
		return value[:limit]
	}
	return value[:limit-3] + "..."
}

func sanitizeName(value string) string {
	value = strings.ToLower(strings.TrimSpace(value))
	if value == "" {
		return "run"
	}
	var b strings.Builder
	for _, r := range value {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		default:
			b.WriteByte('-')
		}
	}
	return strings.Trim(b.String(), "-")
}

func defaultString(value, fallback string) string {
	value = strings.TrimSpace(value)
	if value == "" {
		return fallback
	}
	return value
}
