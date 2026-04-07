package tqserve

import (
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
)

const defaultNativeExecutorSystemPrompt = "You are the TurboQuant native executor. Use retrieved session memory when it is relevant, stay concise, and answer the user's latest request directly."

type NativeExecutor interface {
	Generate(ctx Context, req NativeExecutionRequest) (string, error)
}

type NativeExecutionRequest struct {
	SessionID   string                  `json:"session_id,omitempty"`
	Model       string                  `json:"model,omitempty"`
	Prompt      string                  `json:"prompt,omitempty"`
	Turns       []NativeMessage         `json:"turns,omitempty"`
	Memories    []NativeRetrievedMemory `json:"memories,omitempty"`
	KVTurns     int                     `json:"kv_turns,omitempty"`
	LiveKVBytes uint64                  `json:"live_kv_bytes,omitempty"`
	FocusNorm   float64                 `json:"focus_norm,omitempty"`
}

type NativeMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type NativeRetrievedMemory struct {
	Role    string  `json:"role"`
	Content string  `json:"content"`
	Score   float32 `json:"score"`
}

type memoryExecutor struct{}

type backendNativeExecutor struct {
	backend      Backend
	model        string
	systemPrompt string
}

func defaultNativeExecutor(executor NativeExecutor) NativeExecutor {
	if executor != nil {
		return executor
	}
	return NewMemoryExecutor()
}

func NewMemoryExecutor() NativeExecutor {
	return memoryExecutor{}
}

func (memoryExecutor) Generate(ctx Context, req NativeExecutionRequest) (string, error) {
	if strings.TrimSpace(req.SessionID) == "" {
		return "TurboQuant native runtime accepted the request.", nil
	}
	if strings.TrimSpace(req.Prompt) == "" {
		return fmt.Sprintf("TurboQuant native runtime started session %s with no prompt content yet.", req.SessionID), nil
	}
	if len(req.Memories) == 0 {
		return fmt.Sprintf(
			"TurboQuant native runtime is building memory for session %s. I do not have a close prior turn yet, but I stored your request and the KV cache now holds %d turns (live_kv_bytes=%d, focus_norm=%.3f). Current prompt: %s",
			req.SessionID,
			req.KVTurns,
			req.LiveKVBytes,
			req.FocusNorm,
			summarizeText(req.Prompt, 160),
		), nil
	}

	related := make([]nativeTurn, 0, len(req.Memories))
	lines := make([]string, 0, len(req.Memories))
	for _, memory := range req.Memories {
		related = append(related, nativeTurn{Role: memory.Role, Content: memory.Content})
		lines = append(lines, fmt.Sprintf("- [%s|score=%.3f] %s", memory.Role, float64(memory.Score), summarizeText(memory.Content, 140)))
	}
	return fmt.Sprintf(
		"TurboQuant native runtime found %d relevant prior turn(s) for session %s.\nRelevant memory:\n%s\nWorking answer: %s\nSession state: kv_turns=%d live_kv_bytes=%d focus_norm=%.3f",
		len(req.Memories),
		req.SessionID,
		strings.Join(lines, "\n"),
		composeWorkingAnswer(req.Prompt, related),
		req.KVTurns,
		req.LiveKVBytes,
		req.FocusNorm,
	), nil
}

func NewUpstreamNativeExecutor(baseURL, apiKey, model, systemPrompt string, client *http.Client) (NativeExecutor, error) {
	backend, err := NewUpstreamBackend(baseURL, apiKey, client)
	if err != nil {
		return nil, err
	}
	return newBackendNativeExecutor(backend, model, systemPrompt)
}

func NewOllamaNativeExecutor(baseURL, apiKey, model, systemPrompt string, client *http.Client) (NativeExecutor, error) {
	backend, err := NewOllamaBackend(baseURL, apiKey, client)
	if err != nil {
		return nil, err
	}
	return newBackendNativeExecutor(backend, model, systemPrompt)
}

func NewBackendNativeExecutor(backend Backend, model, systemPrompt string) (NativeExecutor, error) {
	return newBackendNativeExecutor(backend, model, systemPrompt)
}

func newBackendNativeExecutor(backend Backend, model, systemPrompt string) (NativeExecutor, error) {
	if backend == nil {
		return nil, fmt.Errorf("tqserve: native executor backend is required")
	}
	model = strings.TrimSpace(model)
	if model == "" {
		return nil, fmt.Errorf("tqserve: native executor model is required")
	}
	return &backendNativeExecutor{
		backend:      backend,
		model:        model,
		systemPrompt: strings.TrimSpace(systemPrompt),
	}, nil
}

func (e *backendNativeExecutor) Generate(ctx Context, req NativeExecutionRequest) (string, error) {
	payload, err := json.Marshal(map[string]any{
		"model":    e.model,
		"messages": buildNativeExecutorMessages(req, e.systemPrompt),
		"stream":   false,
	})
	if err != nil {
		return "", err
	}
	resp, err := e.backend.ChatCompletions(ctx, RequestEnvelope{Model: e.model}, payload)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	return readBackendText(resp)
}

func buildNativeExecutorMessages(req NativeExecutionRequest, systemPrompt string) []map[string]any {
	messages := []map[string]any{
		{
			"role":    "system",
			"content": defaultString(systemPrompt, defaultNativeExecutorSystemPrompt),
		},
	}
	if len(req.Memories) > 0 {
		messages = append(messages, map[string]any{
			"role":    "system",
			"content": buildNativeGrounding(req),
		})
	}
	turns := recentNativeTurns(req.Turns, 8)
	for _, turn := range turns {
		if strings.TrimSpace(turn.Content) == "" {
			continue
		}
		messages = append(messages, map[string]any{
			"role":    defaultString(strings.TrimSpace(turn.Role), "user"),
			"content": turn.Content,
		})
	}
	if len(turns) == 0 || !sameTurn(turns[len(turns)-1], NativeMessage{Role: "user", Content: req.Prompt}) {
		if prompt := strings.TrimSpace(req.Prompt); prompt != "" {
			messages = append(messages, map[string]any{
				"role":    "user",
				"content": prompt,
			})
		}
	}
	return messages
}

func buildNativeGrounding(req NativeExecutionRequest) string {
	lines := make([]string, 0, len(req.Memories)+2)
	lines = append(lines, "Retrieved session memory:")
	for _, memory := range req.Memories {
		lines = append(lines, fmt.Sprintf("- [%s|score=%.3f] %s", memory.Role, float64(memory.Score), summarizeText(memory.Content, 220)))
	}
	lines = append(lines, fmt.Sprintf("Session stats: kv_turns=%d live_kv_bytes=%d focus_norm=%.3f", req.KVTurns, req.LiveKVBytes, req.FocusNorm))
	lines = append(lines, "Use the retrieved memory when it helps answer the user's latest request, but do not invent details that are not in memory or the visible conversation.")
	return strings.Join(lines, "\n")
}

func recentNativeTurns(turns []NativeMessage, limit int) []NativeMessage {
	if limit <= 0 || len(turns) <= limit {
		return append([]NativeMessage(nil), turns...)
	}
	return append([]NativeMessage(nil), turns[len(turns)-limit:]...)
}

func sameTurn(left, right NativeMessage) bool {
	return strings.EqualFold(strings.TrimSpace(left.Role), strings.TrimSpace(right.Role)) &&
		strings.TrimSpace(left.Content) == strings.TrimSpace(right.Content)
}

func readBackendText(resp *BackendResponse) (string, error) {
	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return "", fmt.Errorf("tqserve: native executor request failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(payload)))
	}
	var body map[string]any
	if err := json.Unmarshal(payload, &body); err != nil {
		return "", fmt.Errorf("tqserve: decode native executor response: %w", err)
	}
	if choices, ok := body["choices"].([]any); ok && len(choices) > 0 {
		if choice, ok := choices[0].(map[string]any); ok {
			if message, ok := choice["message"].(map[string]any); ok {
				if text := strings.TrimSpace(flattenContent(message["content"])); text != "" {
					return text, nil
				}
			}
		}
	}
	if output, ok := body["output"].([]any); ok {
		for _, item := range output {
			message, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if text := strings.TrimSpace(flattenContent(message["content"])); text != "" {
				return text, nil
			}
		}
	}
	if text := strings.TrimSpace(asString(body["text"])); text != "" {
		return text, nil
	}
	return "", fmt.Errorf("tqserve: native executor response did not contain assistant text")
}
