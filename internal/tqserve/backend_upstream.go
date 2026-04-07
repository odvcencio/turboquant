package tqserve

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

type UpstreamBackend struct {
	BaseURL string
	APIKey  string
	Client  *http.Client
}

func NewUpstreamBackend(baseURL, apiKey string, client *http.Client) (*UpstreamBackend, error) {
	baseURL = strings.TrimRight(strings.TrimSpace(baseURL), "/")
	if baseURL == "" {
		return nil, fmt.Errorf("tqserve: upstream base URL is required")
	}
	if client == nil {
		client = &http.Client{Timeout: 0}
	}
	return &UpstreamBackend{
		BaseURL: baseURL,
		APIKey:  strings.TrimSpace(apiKey),
		Client:  client,
	}, nil
}

func (b *UpstreamBackend) Models(ctx context.Context) (ModelList, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, b.url("/models"), nil)
	if err != nil {
		return ModelList{}, err
	}
	if b.APIKey != "" {
		req.Header.Set("Authorization", "Bearer "+b.APIKey)
	}
	resp, err := b.Client.Do(req)
	if err != nil {
		return ModelList{}, err
	}
	defer resp.Body.Close()
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return ModelList{}, fmt.Errorf("tqserve: upstream models request failed with status %d", resp.StatusCode)
	}
	var models ModelList
	if err := json.NewDecoder(resp.Body).Decode(&models); err != nil {
		return ModelList{}, err
	}
	if models.Object == "" {
		models.Object = "list"
	}
	return models, nil
}

func (b *UpstreamBackend) ChatCompletions(ctx context.Context, req RequestEnvelope, body []byte) (*BackendResponse, error) {
	return b.forwardJSON(ctx, "/chat/completions", body)
}

func (b *UpstreamBackend) Responses(ctx context.Context, req RequestEnvelope, body []byte) (*BackendResponse, error) {
	return b.forwardJSON(ctx, "/responses", body)
}

func (b *UpstreamBackend) Status(ctx context.Context) BackendStatus {
	checkCtx, cancel := statusContext(ctx)
	defer cancel()
	status := BackendStatus{
		Kind:    "upstream",
		BaseURL: b.BaseURL,
	}
	if _, err := b.Models(checkCtx); err != nil {
		status.LastError = err.Error()
		return status
	}
	status.Ready = true
	return status
}

func (b *UpstreamBackend) forwardJSON(ctx context.Context, path string, body []byte) (*BackendResponse, error) {
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, b.url(path), bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if b.APIKey != "" {
		httpReq.Header.Set("Authorization", "Bearer "+b.APIKey)
	}
	resp, err := b.Client.Do(httpReq)
	if err != nil {
		return nil, err
	}
	return &BackendResponse{
		StatusCode: resp.StatusCode,
		Header:     cloneHeader(resp.Header),
		Body:       resp.Body,
	}, nil
}

func (b *UpstreamBackend) url(path string) string {
	return b.BaseURL + "/" + strings.TrimLeft(path, "/")
}

func cloneHeader(src http.Header) http.Header {
	dst := make(http.Header, len(src))
	for key, values := range src {
		copied := make([]string, len(values))
		copy(copied, values)
		dst[key] = copied
	}
	return dst
}
