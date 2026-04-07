package tqserve

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

type OllamaBackend struct {
	BaseURL string
	APIKey  string
	Client  *http.Client
}

type ollamaModelsResponse struct {
	Models []struct {
		Name string `json:"name"`
	} `json:"models"`
}

type ollamaChatResponse struct {
	Model   string `json:"model"`
	Message struct {
		Role    string `json:"role"`
		Content string `json:"content"`
	} `json:"message"`
	Done       bool   `json:"done"`
	DoneReason string `json:"done_reason"`
	Error      string `json:"error"`
}

func NewOllamaBackend(baseURL, apiKey string, client *http.Client) (*OllamaBackend, error) {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		return nil, fmt.Errorf("tqserve: ollama base URL is required")
	}
	if client == nil {
		client = &http.Client{Timeout: 0}
	}
	return &OllamaBackend{
		BaseURL: baseURL,
		APIKey:  strings.TrimSpace(apiKey),
		Client:  client,
	}, nil
}

func (b *OllamaBackend) Models(ctx context.Context) (ModelList, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.url("/api/tags"), nil)
	if err != nil {
		return ModelList{}, err
	}
	b.setAuth(req)
	resp, err := b.Client.Do(req)
	if err != nil {
		return ModelList{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return ModelList{}, fmt.Errorf("tqserve: ollama models request failed with status %d", resp.StatusCode)
	}
	var payload ollamaModelsResponse
	if err := json.NewDecoder(resp.Body).Decode(&payload); err != nil {
		return ModelList{}, err
	}
	models := ModelList{
		Object: "list",
		Data:   make([]Model, len(payload.Models)),
	}
	for i, model := range payload.Models {
		models.Data[i] = Model{
			ID:      model.Name,
			Object:  "model",
			OwnedBy: "ollama",
		}
	}
	return models, nil
}

func (b *OllamaBackend) ChatCompletions(ctx context.Context, req RequestEnvelope, body []byte) (*BackendResponse, error) {
	ollamaBody, err := buildOllamaChatBody(body, req.Model, false)
	if err != nil {
		return nil, err
	}
	resp, err := b.doChat(ctx, ollamaBody)
	if err != nil {
		return nil, err
	}
	if req.Stream {
		return b.streamChatCompletion(req.Model, resp)
	}
	return b.singleChatCompletion(req.Model, resp)
}

func (b *OllamaBackend) Responses(ctx context.Context, req RequestEnvelope, body []byte) (*BackendResponse, error) {
	ollamaBody, err := buildOllamaChatBody(body, req.Model, true)
	if err != nil {
		return nil, err
	}
	resp, err := b.doChat(ctx, ollamaBody)
	if err != nil {
		return nil, err
	}
	if req.Stream {
		return b.streamResponses(req.Model, resp)
	}
	return b.singleResponse(req.Model, resp)
}

func (b *OllamaBackend) Status(ctx context.Context) BackendStatus {
	checkCtx, cancel := statusContext(ctx)
	defer cancel()
	status := BackendStatus{
		Kind:      "ollama",
		BaseURL:   b.BaseURL,
		HealthURL: b.url("/api/tags"),
	}
	if _, err := b.Models(checkCtx); err != nil {
		status.LastError = err.Error()
		return status
	}
	status.Ready = true
	return status
}

func (b *OllamaBackend) doChat(ctx context.Context, body []byte) (*http.Response, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, b.url("/api/chat"), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	b.setAuth(req)
	resp, err := b.Client.Do(req)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		defer resp.Body.Close()
		payload, _ := io.ReadAll(resp.Body)
		if len(payload) == 0 {
			return nil, fmt.Errorf("tqserve: ollama chat request failed with status %d", resp.StatusCode)
		}
		return nil, fmt.Errorf("tqserve: ollama chat request failed with status %d: %s", resp.StatusCode, strings.TrimSpace(string(payload)))
	}
	return resp, nil
}

func (b *OllamaBackend) singleChatCompletion(model string, resp *http.Response) (*BackendResponse, error) {
	defer resp.Body.Close()
	var chunk ollamaChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chunk); err != nil {
		return nil, err
	}
	if chunk.Error != "" {
		return nil, fmt.Errorf("tqserve: ollama error: %s", chunk.Error)
	}
	payload := map[string]any{
		"id":      newID("chatcmpl"),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   model,
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]any{
					"role":    defaultString(chunk.Message.Role, "assistant"),
					"content": chunk.Message.Content,
				},
				"finish_reason": defaultString(chunk.DoneReason, "stop"),
			},
		},
	}
	return jsonResponse(http.StatusOK, payload)
}

func (b *OllamaBackend) streamChatCompletion(model string, resp *http.Response) (*BackendResponse, error) {
	body, err := translateOllamaStream(resp.Body, func(seq int, chunk ollamaChatResponse) []byte {
		delta := map[string]any{}
		if seq == 0 {
			delta["role"] = defaultString(chunk.Message.Role, "assistant")
		}
		if chunk.Message.Content != "" {
			delta["content"] = chunk.Message.Content
		}
		payload := map[string]any{
			"id":      newID("chatcmpl"),
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]any{
				{
					"index":         0,
					"delta":         delta,
					"finish_reason": nil,
				},
			},
		}
		if chunk.Done {
			payload["choices"] = []map[string]any{
				{
					"index":         0,
					"delta":         map[string]any{},
					"finish_reason": defaultString(chunk.DoneReason, "stop"),
				},
			}
		}
		return sseData(payload, chunk.Done)
	})
	if err != nil {
		return nil, err
	}
	return &BackendResponse{
		StatusCode: http.StatusOK,
		Header: http.Header{
			"Content-Type":  []string{"text/event-stream"},
			"Cache-Control": []string{"no-cache"},
			"Connection":    []string{"keep-alive"},
		},
		Body: body,
	}, nil
}

func (b *OllamaBackend) singleResponse(model string, resp *http.Response) (*BackendResponse, error) {
	defer resp.Body.Close()
	var chunk ollamaChatResponse
	if err := json.NewDecoder(resp.Body).Decode(&chunk); err != nil {
		return nil, err
	}
	if chunk.Error != "" {
		return nil, fmt.Errorf("tqserve: ollama error: %s", chunk.Error)
	}
	messageID := newID("msg")
	payload := map[string]any{
		"id":      newID("resp"),
		"object":  "response",
		"created": time.Now().Unix(),
		"status":  "completed",
		"model":   model,
		"output": []map[string]any{
			{
				"id":   messageID,
				"type": "message",
				"role": defaultString(chunk.Message.Role, "assistant"),
				"content": []map[string]any{
					{
						"type": "output_text",
						"text": chunk.Message.Content,
					},
				},
			},
		},
	}
	return jsonResponse(http.StatusOK, payload)
}

func (b *OllamaBackend) streamResponses(model string, resp *http.Response) (*BackendResponse, error) {
	responseID := newID("resp")
	messageID := newID("msg")
	body, err := translateOllamaStream(resp.Body, func(seq int, chunk ollamaChatResponse) []byte {
		if chunk.Done {
			payload := map[string]any{
				"type": "response.completed",
				"response": map[string]any{
					"id":      responseID,
					"object":  "response",
					"created": time.Now().Unix(),
					"status":  "completed",
					"model":   model,
				},
			}
			return sseData(payload, true)
		}
		payload := map[string]any{
			"type":          "response.output_text.delta",
			"response_id":   responseID,
			"output_index":  0,
			"content_index": 0,
			"item_id":       messageID,
			"delta":         chunk.Message.Content,
		}
		return sseData(payload, false)
	})
	if err != nil {
		return nil, err
	}
	return &BackendResponse{
		StatusCode: http.StatusOK,
		Header: http.Header{
			"Content-Type":  []string{"text/event-stream"},
			"Cache-Control": []string{"no-cache"},
			"Connection":    []string{"keep-alive"},
		},
		Body: body,
	}, nil
}

func (b *OllamaBackend) url(path string) string {
	return b.BaseURL + "/" + strings.TrimLeft(path, "/")
}

func (b *OllamaBackend) setAuth(req *http.Request) {
	if b.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.APIKey)
	}
}

func buildOllamaChatBody(body []byte, model string, allowInput bool) ([]byte, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return nil, fmt.Errorf("tqserve: invalid JSON request body")
	}
	messages, err := extractMessages(payload, allowInput)
	if err != nil {
		return nil, err
	}
	out := map[string]any{
		"model":    model,
		"stream":   truthy(payload["stream"]),
		"messages": messages,
	}
	if tools, ok := payload["tools"]; ok {
		out["tools"] = tools
	}
	if format, ok := payload["response_format"]; ok {
		out["format"] = format
	} else if format, ok := payload["format"]; ok {
		out["format"] = format
	}
	options := map[string]any{}
	if existing, ok := payload["options"].(map[string]any); ok {
		for key, value := range existing {
			options[key] = value
		}
	}
	remapOption(payload, options, "temperature", "temperature")
	remapOption(payload, options, "top_p", "top_p")
	remapOption(payload, options, "top_k", "top_k")
	remapOption(payload, options, "seed", "seed")
	remapOption(payload, options, "presence_penalty", "presence_penalty")
	remapOption(payload, options, "frequency_penalty", "frequency_penalty")
	remapOption(payload, options, "stop", "stop")
	remapOption(payload, options, "max_tokens", "num_predict")
	if len(options) != 0 {
		out["options"] = options
	}
	return json.Marshal(out)
}

func extractMessages(payload map[string]any, allowInput bool) ([]map[string]any, error) {
	if raw, ok := payload["messages"]; ok {
		return normalizeMessages(raw)
	}
	if allowInput {
		if raw, ok := payload["input"]; ok {
			return normalizeInputMessages(raw)
		}
	}
	return nil, fmt.Errorf("tqserve: request messages are required")
}

func normalizeMessages(raw any) ([]map[string]any, error) {
	items, ok := raw.([]any)
	if !ok {
		return nil, fmt.Errorf("tqserve: request messages must be an array")
	}
	messages := make([]map[string]any, 0, len(items))
	for _, item := range items {
		msg, ok := item.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("tqserve: request message must be an object")
		}
		role, _ := msg["role"].(string)
		if strings.TrimSpace(role) == "" {
			return nil, fmt.Errorf("tqserve: request message role is required")
		}
		messages = append(messages, map[string]any{
			"role":    role,
			"content": normalizeContent(msg["content"]),
		})
	}
	return messages, nil
}

func normalizeInputMessages(raw any) ([]map[string]any, error) {
	switch v := raw.(type) {
	case string:
		return []map[string]any{{"role": "user", "content": v}}, nil
	case []any:
		messages := make([]map[string]any, 0, len(v))
		for _, item := range v {
			switch msg := item.(type) {
			case string:
				messages = append(messages, map[string]any{"role": "user", "content": msg})
			case map[string]any:
				role, _ := msg["role"].(string)
				if role == "" {
					role = "user"
				}
				content := msg["content"]
				if text := flattenResponseContent(content); text != "" {
					content = text
				}
				messages = append(messages, map[string]any{
					"role":    role,
					"content": normalizeContent(content),
				})
			default:
				return nil, fmt.Errorf("tqserve: request input item must be string or object")
			}
		}
		return messages, nil
	default:
		return nil, fmt.Errorf("tqserve: request input must be string or array")
	}
}

func normalizeContent(raw any) any {
	switch v := raw.(type) {
	case string:
		return v
	case []any:
		if text := flattenResponseContent(v); text != "" {
			return text
		}
		return raw
	default:
		return raw
	}
}

func flattenResponseContent(raw any) string {
	items, ok := raw.([]any)
	if !ok {
		return ""
	}
	var b strings.Builder
	for _, item := range items {
		part, ok := item.(map[string]any)
		if !ok {
			continue
		}
		typ, _ := part["type"].(string)
		switch typ {
		case "text", "input_text", "output_text":
			text, _ := part["text"].(string)
			b.WriteString(text)
		}
	}
	return b.String()
}

func remapOption(payload map[string]any, options map[string]any, from, to string) {
	if value, ok := payload[from]; ok {
		options[to] = value
	}
}

func truthy(value any) bool {
	b, _ := value.(bool)
	return b
}

func translateOllamaStream(src io.ReadCloser, encode func(int, ollamaChatResponse) []byte) (io.ReadCloser, error) {
	defer src.Close()
	var out bytes.Buffer
	scanner := bufio.NewScanner(src)
	seq := 0
	for scanner.Scan() {
		line := bytes.TrimSpace(scanner.Bytes())
		if len(line) == 0 {
			continue
		}
		var chunk ollamaChatResponse
		if err := json.Unmarshal(line, &chunk); err != nil {
			return nil, err
		}
		if chunk.Error != "" {
			return nil, fmt.Errorf("tqserve: ollama error: %s", chunk.Error)
		}
		out.Write(encode(seq, chunk))
		seq++
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	return io.NopCloser(bytes.NewReader(out.Bytes())), nil
}

func sseData(payload any, done bool) []byte {
	body, _ := json.Marshal(payload)
	var out bytes.Buffer
	out.WriteString("data: ")
	out.Write(body)
	out.WriteString("\n\n")
	if done {
		out.WriteString("data: [DONE]\n\n")
	}
	return out.Bytes()
}

func jsonResponse(status int, payload any) (*BackendResponse, error) {
	body, err := json.Marshal(payload)
	if err != nil {
		return nil, err
	}
	return &BackendResponse{
		StatusCode: status,
		Header: http.Header{
			"Content-Type": []string{"application/json"},
		},
		Body: io.NopCloser(bytes.NewReader(body)),
	}, nil
}

func newID(prefix string) string {
	return fmt.Sprintf("%s-%d", prefix, time.Now().UnixNano())
}
