package tqserve

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"net/http"
	"strings"
	"sync"
	"time"
	"unicode"

	turboquant "github.com/odvcencio/turboquant"
)

type NativeBackendConfig struct {
	Name             string
	ModelIDs         []string
	OwnedBy          string
	Accelerator      string
	Device           string
	DeviceCount      int
	TotalMemoryBytes uint64
	WeightsBytes     uint64
	MaxSessions      int
	KeyDim           int
	KeyBits          int
	ValueDim         int
	ValueBits        int
	PageCapacity     int
	Seed             int64
	Executor         NativeExecutor
}

type NativeBackend struct {
	name         string
	models       []Model
	ownedBy      string
	accelerator  string
	device       string
	deviceCount  int
	totalMemory  uint64
	weightsBytes uint64
	maxSessions  int
	keyDim       int
	keyBits      int
	valueDim     int
	valueBits    int
	pageCapacity int
	seed         int64
	executor     NativeExecutor

	mu       sync.Mutex
	sessions map[string]*nativeSession
}

type nativeSession struct {
	info  SessionInfo
	model string
	page  *turboquant.KVCachePage
	turns []nativeTurn
}

type nativeTurn struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type nativeState struct {
	Model string       `json:"model"`
	Turns []nativeTurn `json:"turns,omitempty"`
	Page  []byte       `json:"page,omitempty"`
}

func NewNativeBackend(cfg NativeBackendConfig) (*NativeBackend, error) {
	keyDim := defaultInt(cfg.KeyDim, 384)
	keyBits := defaultInt(cfg.KeyBits, 3)
	valueDim := defaultInt(cfg.ValueDim, 384)
	valueBits := defaultInt(cfg.ValueBits, 2)
	pageCapacity := defaultInt(cfg.PageCapacity, 2048)
	if pageCapacity <= 0 {
		return nil, fmt.Errorf("tqserve: native backend page capacity must be > 0")
	}
	modelIDs := cfg.ModelIDs
	if len(modelIDs) == 0 {
		modelIDs = []string{defaultString(cfg.Name, "TurboQuant-Local-Executor")}
	}
	ownedBy := defaultString(cfg.OwnedBy, "turboquant-native")
	models := make([]Model, 0, len(modelIDs))
	for _, modelID := range modelIDs {
		modelID = strings.TrimSpace(modelID)
		if modelID == "" {
			continue
		}
		models = append(models, Model{
			ID:      modelID,
			Object:  "model",
			OwnedBy: ownedBy,
		})
	}
	if len(models) == 0 {
		return nil, fmt.Errorf("tqserve: native backend requires at least one model id")
	}
	return &NativeBackend{
		name:         defaultString(cfg.Name, "native"),
		models:       models,
		ownedBy:      ownedBy,
		accelerator:  defaultString(cfg.Accelerator, "turboquant"),
		device:       defaultString(cfg.Device, "cpu"),
		deviceCount:  defaultInt(cfg.DeviceCount, 1),
		totalMemory:  defaultUint64(cfg.TotalMemoryBytes, 16<<30),
		weightsBytes: defaultUint64(cfg.WeightsBytes, 8<<30),
		maxSessions:  defaultInt(cfg.MaxSessions, 8),
		keyDim:       keyDim,
		keyBits:      keyBits,
		valueDim:     valueDim,
		valueBits:    valueBits,
		pageCapacity: pageCapacity,
		seed:         cfg.Seed,
		executor:     defaultNativeExecutor(cfg.Executor),
		sessions:     make(map[string]*nativeSession),
	}, nil
}

func (b *NativeBackend) Models(ctx Context) (ModelList, error) {
	models := make([]Model, len(b.models))
	copy(models, b.models)
	return ModelList{
		Object: "list",
		Data:   models,
	}, nil
}

func (b *NativeBackend) ChatCompletions(ctx Context, req RequestEnvelope, body []byte) (*BackendResponse, error) {
	text, turns, err := extractPrompt(body)
	if err != nil {
		return nil, err
	}
	session := b.updateSession(req.SessionID, req.Model, turns)
	responseText, err := b.generateText(ctx, session, req.Model, text)
	if err != nil {
		return nil, err
	}
	b.commitAssistantTurn(session, responseText)
	if req.Stream {
		return b.streamChatCompletion(req.Model, responseText)
	}
	return jsonResponse(http.StatusOK, map[string]any{
		"id":      newID("chatcmpl"),
		"object":  "chat.completion",
		"created": time.Now().Unix(),
		"model":   req.Model,
		"choices": []map[string]any{
			{
				"index": 0,
				"message": map[string]any{
					"role":    "assistant",
					"content": responseText,
				},
				"finish_reason": "stop",
			},
		},
	})
}

func (b *NativeBackend) Responses(ctx Context, req RequestEnvelope, body []byte) (*BackendResponse, error) {
	text, turns, err := extractPrompt(body)
	if err != nil {
		return nil, err
	}
	session := b.updateSession(req.SessionID, req.Model, turns)
	responseText, err := b.generateText(ctx, session, req.Model, text)
	if err != nil {
		return nil, err
	}
	b.commitAssistantTurn(session, responseText)
	if req.Stream {
		return b.streamResponses(req.Model, responseText)
	}
	return jsonResponse(http.StatusOK, map[string]any{
		"id":      newID("resp"),
		"object":  "response",
		"created": time.Now().Unix(),
		"status":  "completed",
		"model":   req.Model,
		"output": []map[string]any{
			{
				"id":   newID("msg"),
				"type": "message",
				"role": "assistant",
				"content": []map[string]any{
					{
						"type": "output_text",
						"text": responseText,
					},
				},
			},
		},
	})
}

func (b *NativeBackend) Status(ctx context.Context) BackendStatus {
	status := BackendStatus{
		Name:  b.name,
		Kind:  "native",
		Ready: true,
	}
	capacity := b.Capacity(ctx)
	status.Capacity = &capacity
	return status
}

func (b *NativeBackend) Capacity(ctx context.Context) CapacitySnapshot {
	b.mu.Lock()
	defer b.mu.Unlock()
	var kvBytes uint64
	for _, session := range b.sessions {
		if session.page != nil {
			kvBytes += session.page.StorageBytes()
		}
	}
	used := b.weightsBytes + kvBytes
	free := uint64(0)
	if b.totalMemory > used {
		free = b.totalMemory - used
	}
	return CapacitySnapshot{
		Accelerator:      b.accelerator,
		Device:           b.device,
		DeviceCount:      b.deviceCount,
		TotalMemoryBytes: b.totalMemory,
		UsedMemoryBytes:  used,
		FreeMemoryBytes:  free,
		WeightsBytes:     b.weightsBytes,
		KVCacheBytes:     kvBytes,
		KVHeadroomBytes:  free,
		MaxSessions:      b.maxSessions,
		ActiveSessions:   len(b.sessions),
		Notes:            "native TurboQuant session backend",
	}
}

func (b *NativeBackend) CaptureSessionState(ctx Context, sessionID string) (json.RawMessage, error) {
	b.mu.Lock()
	session := b.sessions[strings.TrimSpace(sessionID)]
	b.mu.Unlock()
	if session == nil {
		return nil, fmt.Errorf("%w %q", ErrUnknownSession, sessionID)
	}
	page, err := session.page.MarshalBinary()
	if err != nil {
		return nil, err
	}
	state, err := json.Marshal(nativeState{
		Model: session.model,
		Turns: append([]nativeTurn(nil), session.turns...),
		Page:  page,
	})
	if err != nil {
		return nil, err
	}
	return state, nil
}

func (b *NativeBackend) RestoreSessionState(ctx Context, sessionID string, state json.RawMessage) error {
	var payload nativeState
	if err := json.Unmarshal(state, &payload); err != nil {
		return err
	}
	page, err := turboquant.UnmarshalKVCachePage(payload.Page)
	if err != nil {
		return err
	}
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return fmt.Errorf("tqserve: session id is required for native restore")
	}
	now := time.Now().UTC()
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.maxSessions > 0 && len(b.sessions) >= b.maxSessions {
		if _, ok := b.sessions[sessionID]; !ok {
			return fmt.Errorf("tqserve: native backend at max sessions (%d)", b.maxSessions)
		}
	}
	b.sessions[sessionID] = &nativeSession{
		info: SessionInfo{
			ID:           sessionID,
			Model:        payload.Model,
			Backend:      "default",
			LastEndpoint: "restore",
			CreatedAt:    now,
			LastSeenAt:   now,
			RequestCount: 1,
		},
		model: payload.Model,
		page:  page,
		turns: append([]nativeTurn(nil), payload.Turns...),
	}
	return nil
}

func (b *NativeBackend) updateSession(sessionID, model string, turns []nativeTurn) *nativeSession {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		sessionID = newID("sess")
	}
	now := time.Now().UTC()

	b.mu.Lock()
	defer b.mu.Unlock()

	session := b.sessions[sessionID]
	if session == nil {
		if b.maxSessions > 0 && len(b.sessions) >= b.maxSessions {
			// Reuse the least recently seen session to keep the backend live rather than failing.
			session = b.evictOldestLocked()
		}
		session = &nativeSession{
			info: SessionInfo{
				ID:        sessionID,
				CreatedAt: now,
			},
			page: turboquant.NewKVCachePageWithSeed(b.keyDim, b.keyBits, b.valueDim, b.valueBits, b.pageCapacity, b.seed^int64(len(b.sessions)+1)),
		}
		b.sessions[sessionID] = session
	}
	session.info.ID = sessionID
	session.info.Model = model
	session.info.Backend = "default"
	session.info.LastEndpoint = "chat_completions"
	session.info.LastSeenAt = now
	if session.info.CreatedAt.IsZero() {
		session.info.CreatedAt = now
	}
	session.info.RequestCount++
	session.model = model
	if len(turns) != 0 {
		b.syncTurnsLocked(session, turns)
	}
	return session
}

func (b *NativeBackend) syncTurnsLocked(session *nativeSession, turns []nativeTurn) {
	normalized := normalizedTurns(turns)
	if len(normalized) == 0 {
		return
	}
	if turnsEqual(session.turns, normalized) {
		return
	}
	if isPrefix(session.turns, normalized) {
		for _, turn := range normalized[len(session.turns):] {
			b.appendTurnLocked(session, turn)
		}
		return
	}
	if overlap := turnOverlap(session.turns, normalized); overlap > 0 {
		for _, turn := range normalized[overlap:] {
			b.appendTurnLocked(session, turn)
		}
		return
	}
	if len(normalized) <= 2 {
		for _, turn := range normalized {
			b.appendTurnLocked(session, turn)
		}
		return
	}
	session.turns = nil
	session.page = turboquant.NewKVCachePageWithSeed(b.keyDim, b.keyBits, b.valueDim, b.valueBits, b.pageCapacity, b.seed^int64(len(b.sessions)+1))
	for _, turn := range normalized {
		b.appendTurnLocked(session, turn)
	}
}

func (b *NativeBackend) appendTurnLocked(session *nativeSession, turn nativeTurn) {
	turn.Role = defaultString(strings.TrimSpace(turn.Role), "user")
	turn.Content = strings.TrimSpace(turn.Content)
	if turn.Content == "" {
		return
	}
	key := syntheticVector(b.keyDim, session.info.ID, turn.Content, len(session.turns), 0)
	value := syntheticVector(b.valueDim, session.info.ID, turn.Content, len(session.turns), 1)
	session.page.Append(key, value)
	session.turns = append(session.turns, turn)
}

func (b *NativeBackend) commitAssistantTurn(session *nativeSession, text string) {
	if session == nil {
		return
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	live := b.sessions[session.info.ID]
	if live == nil {
		return
	}
	if len(live.turns) > 0 {
		last := live.turns[len(live.turns)-1]
		if last.Role == "assistant" && last.Content == text {
			return
		}
	}
	b.appendTurnLocked(live, nativeTurn{Role: "assistant", Content: text})
	live.info.LastSeenAt = time.Now().UTC()
}

func (b *NativeBackend) generateText(ctx Context, session *nativeSession, model, prompt string) (string, error) {
	return b.executor.Generate(ctx, b.buildExecutionRequest(session, model, prompt))
}

func (b *NativeBackend) buildExecutionRequest(session *nativeSession, model, prompt string) NativeExecutionRequest {
	req := NativeExecutionRequest{
		Model:  defaultString(model, ""),
		Prompt: strings.TrimSpace(prompt),
	}
	if session == nil {
		return req
	}
	req.SessionID = session.info.ID
	if req.Model == "" {
		req.Model = session.model
	}
	req.Prompt = currentPromptText(session.turns, req.Prompt)
	req.KVTurns = session.page.Len()
	req.LiveKVBytes = session.page.LiveBytes()
	req.Turns = make([]NativeMessage, 0, len(session.turns))
	for _, turn := range session.turns {
		req.Turns = append(req.Turns, NativeMessage{Role: turn.Role, Content: turn.Content})
	}
	if req.Prompt == "" || session.page.Len() == 0 {
		return req
	}

	pq := session.page.PrepareQuery(syntheticVector(b.keyDim, session.info.ID, req.Prompt, session.page.Len(), 0))
	k := minInt(4, session.page.Len())
	if k > 0 {
		indices, scores := session.page.TopKPrepared(pq, k)
		req.Memories = make([]NativeRetrievedMemory, 0, len(indices))
		for i, idx := range indices {
			slot := int(idx)
			if slot < 0 || slot >= len(session.turns) {
				continue
			}
			turn := session.turns[slot]
			if strings.EqualFold(turn.Role, "user") && turn.Content == req.Prompt {
				continue
			}
			memory := NativeRetrievedMemory{Role: turn.Role, Content: turn.Content, Score: scores[i]}
			if containsMemory(req.Memories, memory) {
				continue
			}
			req.Memories = append(req.Memories, memory)
		}
	}
	focus := make([]float32, b.valueDim)
	entries := maxInt(1, minInt(4, session.page.Len()))
	session.page.AttentionOutputPreparedInto(focus, make([]uint32, entries), make([]float32, entries), pq)
	req.FocusNorm = vectorNorm(focus)
	return req
}

func (b *NativeBackend) streamChatCompletion(model, text string) (*BackendResponse, error) {
	return streamedResponse(http.StatusOK, "text/event-stream", [][]byte{
		sseData(map[string]any{
			"id":      newID("chatcmpl"),
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]any{
				{"index": 0, "delta": map[string]any{"role": "assistant", "content": text}, "finish_reason": nil},
			},
		}, false),
		sseData(map[string]any{
			"id":      newID("chatcmpl"),
			"object":  "chat.completion.chunk",
			"created": time.Now().Unix(),
			"model":   model,
			"choices": []map[string]any{
				{"index": 0, "delta": map[string]any{}, "finish_reason": "stop"},
			},
		}, true),
	})
}

func (b *NativeBackend) streamResponses(model, text string) (*BackendResponse, error) {
	return streamedResponse(http.StatusOK, "text/event-stream", [][]byte{
		sseData(map[string]any{
			"id":      newID("resp"),
			"object":  "response.output_text.delta",
			"created": time.Now().Unix(),
			"model":   model,
			"delta":   text,
		}, false),
		sseData(map[string]any{
			"id":      newID("resp"),
			"object":  "response.completed",
			"created": time.Now().Unix(),
			"model":   model,
		}, true),
	})
}

func (b *NativeBackend) evictOldestLocked() *nativeSession {
	var oldestID string
	var oldest *nativeSession
	for id, session := range b.sessions {
		if oldest == nil || session.info.LastSeenAt.Before(oldest.info.LastSeenAt) {
			oldestID, oldest = id, session
		}
	}
	if oldest != nil {
		delete(b.sessions, oldestID)
	}
	return oldest
}

func extractPrompt(body []byte) (string, []nativeTurn, error) {
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return "", nil, fmt.Errorf("tqserve: invalid native request body")
	}
	if messages, ok := payload["messages"].([]any); ok {
		turns := parseTurns(messages)
		return currentPromptText(turns, flattenMessages(messages)), turns, nil
	}
	if input, ok := payload["input"]; ok {
		turns := parseInputTurns(input)
		return currentPromptText(turns, flattenInput(input)), turns, nil
	}
	return "", nil, nil
}

func flattenMessages(messages []any) string {
	parts := make([]string, 0, len(messages))
	for _, item := range messages {
		message, ok := item.(map[string]any)
		if !ok {
			continue
		}
		content := flattenContent(message["content"])
		if content != "" {
			parts = append(parts, content)
		}
	}
	return strings.Join(parts, "\n")
}

func parseTurns(messages []any) []nativeTurn {
	turns := make([]nativeTurn, 0, len(messages))
	for _, item := range messages {
		message, ok := item.(map[string]any)
		if !ok {
			continue
		}
		turns = append(turns, nativeTurn{
			Role:    defaultString(asString(message["role"]), "user"),
			Content: flattenContent(message["content"]),
		})
	}
	return turns
}

func flattenInput(input any) string {
	switch value := input.(type) {
	case string:
		return value
	case []any:
		parts := make([]string, 0, len(value))
		for _, item := range value {
			if message, ok := item.(map[string]any); ok {
				parts = append(parts, flattenContent(message["content"]))
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

func parseInputTurns(input any) []nativeTurn {
	switch value := input.(type) {
	case string:
		return []nativeTurn{{Role: "user", Content: value}}
	case []any:
		turns := make([]nativeTurn, 0, len(value))
		for _, item := range value {
			if message, ok := item.(map[string]any); ok {
				turns = append(turns, nativeTurn{
					Role:    defaultString(asString(message["role"]), "user"),
					Content: flattenContent(message["content"]),
				})
			}
		}
		return turns
	default:
		return nil
	}
}

func flattenContent(content any) string {
	switch value := content.(type) {
	case string:
		return value
	case []any:
		parts := make([]string, 0, len(value))
		for _, item := range value {
			part, ok := item.(map[string]any)
			if !ok {
				continue
			}
			if text := asString(part["text"]); text != "" {
				parts = append(parts, text)
			}
		}
		return strings.Join(parts, "\n")
	default:
		return ""
	}
}

func asString(value any) string {
	text, _ := value.(string)
	return text
}

func estimateEntries(prompt string) int {
	parts := strings.Fields(prompt)
	if len(parts) == 0 {
		return 1
	}
	if len(parts) > 128 {
		return 128
	}
	return len(parts)
}

func latestTurnContent(turns []nativeTurn) string {
	for i := len(turns) - 1; i >= 0; i-- {
		if text := strings.TrimSpace(turns[i].Content); text != "" {
			return text
		}
	}
	return ""
}

func latestRoleContent(turns []nativeTurn, role string) string {
	role = strings.TrimSpace(strings.ToLower(role))
	for i := len(turns) - 1; i >= 0; i-- {
		if strings.ToLower(strings.TrimSpace(turns[i].Role)) != role {
			continue
		}
		if text := strings.TrimSpace(turns[i].Content); text != "" {
			return text
		}
	}
	return ""
}

func currentPromptText(turns []nativeTurn, fallback string) string {
	if text := latestRoleContent(turns, "user"); text != "" {
		return text
	}
	if text := strings.TrimSpace(fallback); text != "" {
		return text
	}
	return latestTurnContent(turns)
}

func normalizedTurns(turns []nativeTurn) []nativeTurn {
	out := make([]nativeTurn, 0, len(turns))
	for _, turn := range turns {
		turn.Role = defaultString(strings.TrimSpace(turn.Role), "user")
		turn.Content = strings.TrimSpace(turn.Content)
		if turn.Content == "" {
			continue
		}
		out = append(out, turn)
	}
	return out
}

func turnsEqual(left, right []nativeTurn) bool {
	if len(left) != len(right) {
		return false
	}
	for i := range left {
		if left[i] != right[i] {
			return false
		}
	}
	return true
}

func isPrefix(prefix, full []nativeTurn) bool {
	if len(prefix) > len(full) {
		return false
	}
	for i := range prefix {
		if prefix[i] != full[i] {
			return false
		}
	}
	return true
}

func containsTurn(turns []nativeTurn, target nativeTurn) bool {
	for _, turn := range turns {
		if turn == target {
			return true
		}
	}
	return false
}

func containsMemory(memories []NativeRetrievedMemory, target NativeRetrievedMemory) bool {
	for _, memory := range memories {
		if memory.Role == target.Role && memory.Content == target.Content {
			return true
		}
	}
	return false
}

func turnOverlap(existing, incoming []nativeTurn) int {
	maxOverlap := minInt(len(existing), len(incoming))
	for overlap := maxOverlap; overlap > 0; overlap-- {
		if turnsEqual(existing[len(existing)-overlap:], incoming[:overlap]) {
			return overlap
		}
	}
	return 0
}

func composeWorkingAnswer(prompt string, related []nativeTurn) string {
	if len(related) == 0 {
		return summarizeText(prompt, 160)
	}
	primary := related[0]
	if strings.Contains(strings.ToLower(prompt), "what") || strings.Contains(strings.ToLower(prompt), "recall") || strings.Contains(strings.ToLower(prompt), "remember") {
		return fmt.Sprintf("The closest prior context was from the %s turn: %s", primary.Role, summarizeText(primary.Content, 180))
	}
	return fmt.Sprintf("I would anchor the next step on the %s memory: %s", primary.Role, summarizeText(primary.Content, 180))
}

func summarizeText(text string, limit int) string {
	text = strings.TrimSpace(strings.ReplaceAll(text, "\n", " "))
	if len(text) <= limit {
		return text
	}
	if limit <= 3 {
		return text[:limit]
	}
	return text[:limit-3] + "..."
}

func vectorNorm(values []float32) float64 {
	var sum float64
	for _, value := range values {
		sum += float64(value * value)
	}
	return math.Sqrt(sum)
}

func syntheticVector(dim int, sessionID, prompt string, index, lane int) []float32 {
	vector := make([]float32, dim)
	if dim == 0 {
		return vector
	}
	_ = sessionID
	tokens := textTokens(prompt)
	if len(tokens) == 0 {
		tokens = []string{"_empty"}
	}
	for _, token := range tokens {
		for step := 0; step < 4; step++ {
			slot := syntheticSlot(token, index, lane, step, dim)
			weight := syntheticWeight(token, lane, step)
			vector[slot] += weight
		}
	}
	norm := vectorNorm(vector)
	if norm == 0 {
		return vector
	}
	scale := float32(1 / norm)
	for i := range vector {
		vector[i] *= scale
	}
	return vector
}

func textTokens(text string) []string {
	text = strings.ToLower(strings.TrimSpace(text))
	if text == "" {
		return nil
	}
	fields := strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	if len(fields) == 0 {
		return []string{text}
	}
	return fields
}

func syntheticSlot(token string, index, lane, step, dim int) int {
	h := fnv.New64a()
	_, _ = h.Write([]byte(token))
	_, _ = h.Write([]byte{byte(index), byte(lane), byte(step)})
	return int(h.Sum64() % uint64(dim))
}

func syntheticWeight(token string, lane, step int) float32 {
	h := fnv.New64a()
	_, _ = h.Write([]byte(token))
	_, _ = h.Write([]byte{byte(lane), byte(step), 0x7f})
	if h.Sum64()&1 == 0 {
		return 1
	}
	return -1
}

func streamedResponse(status int, contentType string, chunks [][]byte) (*BackendResponse, error) {
	var buf bytes.Buffer
	for _, chunk := range chunks {
		buf.Write(chunk)
	}
	return &BackendResponse{
		StatusCode: status,
		Header: http.Header{
			"Content-Type":  []string{contentType},
			"Cache-Control": []string{"no-cache"},
			"Connection":    []string{"keep-alive"},
		},
		Body: io.NopCloser(bytes.NewReader(buf.Bytes())),
	}, nil
}

func defaultInt(value, fallback int) int {
	if value != 0 {
		return value
	}
	return fallback
}

func defaultUint64(value, fallback uint64) uint64 {
	if value != 0 {
		return value
	}
	return fallback
}

func minInt(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func maxInt(a, b int) int {
	if a > b {
		return a
	}
	return b
}
