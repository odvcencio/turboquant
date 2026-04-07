package tqserve

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os/exec"
	"strings"
	"sync"
	"testing"
	"time"

	turboquant "github.com/odvcencio/turboquant"
)

func TestHealthz(t *testing.T) {
	srv := newTestServer(t, nil, nil)
	req := httptest.NewRequest(http.MethodGet, "/healthz", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d want %d", rec.Code, http.StatusOK)
	}
	if got := rec.Header().Get("Content-Type"); !strings.Contains(got, "application/json") {
		t.Fatalf("content-type = %q", got)
	}
	var payload map[string]any
	if err := json.NewDecoder(rec.Body).Decode(&payload); err != nil {
		t.Fatalf("decode health payload: %v", err)
	}
	if payload["version"] != turboquant.Version {
		t.Fatalf("health version = %v want %q", payload["version"], turboquant.Version)
	}
}

func TestModelsUsesPublicMapping(t *testing.T) {
	srv := newTestServer(t, nil, map[string]string{
		"local-chat": "upstream/model-a",
		"local-code": "upstream/model-b",
	})
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer sk-test")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d want %d", rec.Code, http.StatusOK)
	}
	var models ModelList
	if err := json.NewDecoder(rec.Body).Decode(&models); err != nil {
		t.Fatalf("decode models: %v", err)
	}
	if len(models.Data) != 2 || models.Data[0].ID != "local-chat" || models.Data[1].ID != "local-code" {
		t.Fatalf("models = %+v", models.Data)
	}
}

func TestChatCompletionsRewritesModelAndForwardsAuth(t *testing.T) {
	var gotAuth string
	var gotModel string
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		gotAuth = r.Header.Get("Authorization")
		defer r.Body.Close()
		var req RequestEnvelope
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		gotModel = req.Model
		w.Header().Set("Content-Type", "application/json")
		_, _ = io.WriteString(w, `{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer upstream.Close()

	backend, err := NewUpstreamBackend(upstream.URL+"/v1", "sk-upstream", nil)
	if err != nil {
		t.Fatalf("NewUpstreamBackend: %v", err)
	}
	srv, err := New(Config{
		APIKeys: []string{"sk-test"},
		ModelMap: map[string]string{
			"local-chat": "upstream/model-a",
		},
		Backend: backend,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"local-chat","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer sk-test")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d want %d", rec.Code, http.StatusOK)
	}
	if gotAuth != "Bearer sk-upstream" {
		t.Fatalf("upstream auth = %q want %q", gotAuth, "Bearer sk-upstream")
	}
	if gotModel != "upstream/model-a" {
		t.Fatalf("model = %q want %q", gotModel, "upstream/model-a")
	}
}

func TestChatCompletionsRejectsUnknownModel(t *testing.T) {
	srv := newTestServer(t, nil, map[string]string{"local-chat": "upstream/model-a"})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"missing","messages":[]}`))
	req.Header.Set("Authorization", "Bearer sk-test")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusNotFound {
		t.Fatalf("status = %d want %d", rec.Code, http.StatusNotFound)
	}
}

func TestAuthRequired(t *testing.T) {
	srv := newTestServer(t, nil, nil)
	req := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusUnauthorized {
		t.Fatalf("status = %d want %d", rec.Code, http.StatusUnauthorized)
	}
}

func TestRoutesAcrossMultipleBackends(t *testing.T) {
	var upstreamHits int
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		upstreamHits++
		defer r.Body.Close()
		var req RequestEnvelope
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		if req.Model != "upstream/model-a" {
			t.Fatalf("model = %q want %q", req.Model, "upstream/model-a")
		}
		_, _ = io.WriteString(w, `{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
	}))
	defer upstream.Close()
	ollama := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/tags" {
			_, _ = io.WriteString(w, `{"models":[{"name":"qwen2.5:7b"}]}`)
			return
		}
		defer r.Body.Close()
		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Fatalf("decode ollama request: %v", err)
		}
		if req["model"] != "qwen2.5:7b" {
			t.Fatalf("ollama model = %v want qwen2.5:7b", req["model"])
		}
		_, _ = io.WriteString(w, `{"model":"qwen2.5:7b","message":{"role":"assistant","content":"ollama"},"done":true,"done_reason":"stop"}`)
	}))
	defer ollama.Close()

	openaiBackend, err := NewUpstreamBackend(upstream.URL+"/v1", "", nil)
	if err != nil {
		t.Fatalf("NewUpstreamBackend: %v", err)
	}
	ollamaBackend, err := NewOllamaBackend(ollama.URL, "", nil)
	if err != nil {
		t.Fatalf("NewOllamaBackend: %v", err)
	}
	srv, err := New(Config{
		APIKeys: []string{"sk-test"},
		Backends: map[string]Backend{
			"openai": openaiBackend,
			"ollama": ollamaBackend,
		},
		Routes: []ModelRoute{
			{PublicModel: "public-openai", BackendName: "openai", BackendModel: "upstream/model-a"},
			{PublicModel: "public-ollama", BackendName: "ollama", BackendModel: "qwen2.5:7b"},
		},
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	for _, model := range []string{"public-openai", "public-ollama"} {
		req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"`+model+`","messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("Authorization", "Bearer sk-test")
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, req)
		if rec.Code != http.StatusOK {
			t.Fatalf("%s status = %d want %d", model, rec.Code, http.StatusOK)
		}
	}
	if upstreamHits != 1 {
		t.Fatalf("upstreamHits = %d want 1", upstreamHits)
	}
}

func TestStatusIncludesRuntimeSurface(t *testing.T) {
	srv := newTestServer(t, nil, map[string]string{"local-chat": "upstream/model-a"})
	req := httptest.NewRequest(http.MethodGet, "/v1/tq/status", nil)
	req.Header.Set("Authorization", "Bearer sk-test")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d want %d", rec.Code, http.StatusOK)
	}
	var status ServerStatus
	if err := json.NewDecoder(rec.Body).Decode(&status); err != nil {
		t.Fatalf("decode status: %v", err)
	}
	if !status.OK || status.Service != "tqserve" {
		t.Fatalf("status payload = %+v", status)
	}
	if status.Version != turboquant.Version {
		t.Fatalf("status version = %q want %q", status.Version, turboquant.Version)
	}
	if status.SessionHeader != DefaultSessionHeader || status.ActiveSessions != 0 {
		t.Fatalf("runtime surface = %+v", status)
	}
	if len(status.Backends) != 1 || !status.Backends[0].Ready {
		t.Fatalf("backends = %+v", status.Backends)
	}
	if len(status.Routes) != 1 || status.Routes[0].PublicModel != "local-chat" {
		t.Fatalf("routes = %+v", status.Routes)
	}
}

func TestStatusIncludesBackendCapacity(t *testing.T) {
	backend := &statusCapacityBackend{
		models: ModelList{
			Object: "list",
			Data:   []Model{{ID: "capacity/model", Object: "model", OwnedBy: "native"}},
		},
		status: BackendStatus{
			Kind:  "native",
			Ready: true,
		},
		capacity: CapacitySnapshot{
			Accelerator:      "cuda",
			Device:           "RTX 4090",
			DeviceCount:      1,
			TotalMemoryBytes: 24 << 30,
			KVHeadroomBytes:  8 << 30,
			MaxSessions:      12,
		},
	}
	srv, err := New(Config{
		APIKeys: []string{"sk-test"},
		ModelMap: map[string]string{
			"local-chat": "capacity/model",
		},
		Backend: backend,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := httptest.NewRequest(http.MethodGet, "/v1/tq/status", nil)
	req.Header.Set("Authorization", "Bearer sk-test")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d want %d", rec.Code, http.StatusOK)
	}
	var status ServerStatus
	if err := json.NewDecoder(rec.Body).Decode(&status); err != nil {
		t.Fatalf("decode status: %v", err)
	}
	if len(status.Backends) != 1 || status.Backends[0].Capacity == nil {
		t.Fatalf("backends = %+v", status.Backends)
	}
	capacity := status.Backends[0].Capacity
	if capacity.Accelerator != "cuda" || capacity.Device != "RTX 4090" || capacity.MaxSessions != 12 {
		t.Fatalf("capacity = %+v", *capacity)
	}
}

func TestSessionsTrackByHeader(t *testing.T) {
	srv := newTestServer(t, nil, map[string]string{"local-chat": "upstream/model-a"})
	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"local-chat","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer sk-test")
	req.Header.Set(DefaultSessionHeader, "sess-123")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("chat status = %d want %d", rec.Code, http.StatusOK)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v1/tq/sessions", nil)
	listReq.Header.Set("Authorization", "Bearer sk-test")
	listRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("sessions status = %d want %d", listRec.Code, http.StatusOK)
	}
	var payload struct {
		Data          []SessionInfo `json:"data"`
		SessionHeader string        `json:"session_header"`
	}
	if err := json.NewDecoder(listRec.Body).Decode(&payload); err != nil {
		t.Fatalf("decode sessions: %v", err)
	}
	if payload.SessionHeader != DefaultSessionHeader {
		t.Fatalf("session header = %q want %q", payload.SessionHeader, DefaultSessionHeader)
	}
	if len(payload.Data) != 1 {
		t.Fatalf("sessions = %+v", payload.Data)
	}
	session := payload.Data[0]
	if session.ID != "sess-123" || session.Model != "local-chat" || session.Backend != "default" || session.RequestCount != 1 {
		t.Fatalf("session = %+v", session)
	}
}

func TestAgentsClaimsAndEvents(t *testing.T) {
	srv := newTestServer(t, nil, map[string]string{"local-chat": "upstream/model-a"})

	postJSON := func(path, body string) *httptest.ResponseRecorder {
		req := httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
		req.Header.Set("Authorization", "Bearer sk-test")
		req.Header.Set("Content-Type", "application/json")
		rec := httptest.NewRecorder()
		srv.Handler().ServeHTTP(rec, req)
		return rec
	}

	agentOak := postJSON("/v1/tq/agents", `{"session_id":"sess-collab","agent_id":"oak","role":"worker","status":"editing","capabilities":["go","turboquant"]}`)
	if agentOak.Code != http.StatusOK {
		t.Fatalf("oak status = %d want %d", agentOak.Code, http.StatusOK)
	}
	agentBirch := postJSON("/v1/tq/agents", `{"session_id":"sess-collab","agent_id":"birch","role":"reviewer","status":"watching"}`)
	if agentBirch.Code != http.StatusOK {
		t.Fatalf("birch status = %d want %d", agentBirch.Code, http.StatusOK)
	}

	claimOak := postJSON("/v1/tq/claims", `{"session_id":"sess-collab","agent_id":"oak","entity":"internal/tqserve/server.go#handleProxy","mode":"exclusive","scope":"entity"}`)
	if claimOak.Code != http.StatusOK {
		t.Fatalf("claim oak status = %d want %d", claimOak.Code, http.StatusOK)
	}
	claimBirch := postJSON("/v1/tq/claims", `{"session_id":"sess-collab","agent_id":"birch","entity":"internal/tqserve/server.go#handleProxy","mode":"exclusive","scope":"entity"}`)
	if claimBirch.Code != http.StatusConflict {
		t.Fatalf("claim birch status = %d want %d", claimBirch.Code, http.StatusConflict)
	}

	note := postJSON("/v1/tq/events", `{"session_id":"sess-collab","type":"agent.note","agent_id":"oak","summary":"waiting on review"}`)
	if note.Code != http.StatusOK {
		t.Fatalf("note status = %d want %d", note.Code, http.StatusOK)
	}

	agentsReq := httptest.NewRequest(http.MethodGet, "/v1/tq/agents?session_id=sess-collab", nil)
	agentsReq.Header.Set("Authorization", "Bearer sk-test")
	agentsRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(agentsRec, agentsReq)
	if agentsRec.Code != http.StatusOK {
		t.Fatalf("agents status = %d want %d", agentsRec.Code, http.StatusOK)
	}
	var agents struct {
		Data []AgentPresence `json:"data"`
	}
	if err := json.NewDecoder(agentsRec.Body).Decode(&agents); err != nil {
		t.Fatalf("decode agents: %v", err)
	}
	if len(agents.Data) != 2 {
		t.Fatalf("agents = %+v", agents.Data)
	}

	claimsReq := httptest.NewRequest(http.MethodGet, "/v1/tq/claims?session_id=sess-collab", nil)
	claimsReq.Header.Set("Authorization", "Bearer sk-test")
	claimsRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(claimsRec, claimsReq)
	if claimsRec.Code != http.StatusOK {
		t.Fatalf("claims status = %d want %d", claimsRec.Code, http.StatusOK)
	}
	var claims struct {
		Data []AgentClaim `json:"data"`
	}
	if err := json.NewDecoder(claimsRec.Body).Decode(&claims); err != nil {
		t.Fatalf("decode claims: %v", err)
	}
	if len(claims.Data) != 1 || claims.Data[0].AgentID != "oak" {
		t.Fatalf("claims = %+v", claims.Data)
	}

	eventsReq := httptest.NewRequest(http.MethodGet, "/v1/tq/events?session_id=sess-collab", nil)
	eventsReq.Header.Set("Authorization", "Bearer sk-test")
	eventsRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(eventsRec, eventsReq)
	if eventsRec.Code != http.StatusOK {
		t.Fatalf("events status = %d want %d", eventsRec.Code, http.StatusOK)
	}
	var events struct {
		Data []CollaborationEvent `json:"data"`
	}
	if err := json.NewDecoder(eventsRec.Body).Decode(&events); err != nil {
		t.Fatalf("decode events: %v", err)
	}
	var sawConflict, sawNote bool
	for _, event := range events.Data {
		if event.Type == "claim.conflict" {
			sawConflict = true
		}
		if event.Type == "agent.note" && event.AgentID == "oak" {
			sawNote = true
		}
	}
	if !sawConflict || !sawNote {
		t.Fatalf("events = %+v", events.Data)
	}

	statusReq := httptest.NewRequest(http.MethodGet, "/v1/tq/status", nil)
	statusReq.Header.Set("Authorization", "Bearer sk-test")
	statusRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(statusRec, statusReq)
	if statusRec.Code != http.StatusOK {
		t.Fatalf("status code = %d want %d", statusRec.Code, http.StatusOK)
	}
	var status ServerStatus
	if err := json.NewDecoder(statusRec.Body).Decode(&status); err != nil {
		t.Fatalf("decode status: %v", err)
	}
	if status.ActiveAgents != 2 || status.ActiveClaims != 1 {
		t.Fatalf("status collaboration counts = %+v", status)
	}
}

func TestServerUsesCustomSessionStore(t *testing.T) {
	store := &spySessionStore{}
	srv, err := New(Config{
		APIKeys: []string{"sk-test"},
		ModelMap: map[string]string{
			"local-chat": "upstream/model-a",
		},
		Backend:      testUpstreamBackend(t),
		SessionStore: store,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"local-chat","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer sk-test")
	req.Header.Set(DefaultSessionHeader, "sess-custom")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("chat status = %d want %d", rec.Code, http.StatusOK)
	}
	if store.count() != 1 {
		t.Fatalf("custom session count = %d want 1", store.count())
	}
	if got := store.last(); got.ID != "sess-custom" || got.Model != "local-chat" || got.Backend != "default" {
		t.Fatalf("stored session = %+v", got)
	}
}

func TestCheckpointCaptureAndRestore(t *testing.T) {
	backend := &statefulBackend{
		models: ModelList{
			Object: "list",
			Data:   []Model{{ID: "stateful/model", Object: "model", OwnedBy: "native"}},
		},
	}
	srv, err := New(Config{
		APIKeys: []string{"sk-test"},
		ModelMap: map[string]string{
			"local-chat": "stateful/model",
		},
		Backend: backend,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}

	req := httptest.NewRequest(http.MethodPost, "/v1/chat/completions", strings.NewReader(`{"model":"local-chat","messages":[{"role":"user","content":"hi"}]}`))
	req.Header.Set("Authorization", "Bearer sk-test")
	req.Header.Set(DefaultSessionHeader, "sess-checkpoint")
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(rec, req)
	if rec.Code != http.StatusOK {
		t.Fatalf("chat status = %d want %d", rec.Code, http.StatusOK)
	}

	captureReq := httptest.NewRequest(http.MethodPost, "/v1/tq/checkpoints", strings.NewReader(`{"session_id":"sess-checkpoint"}`))
	captureReq.Header.Set("Authorization", "Bearer sk-test")
	captureRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(captureRec, captureReq)
	if captureRec.Code != http.StatusOK {
		t.Fatalf("capture status = %d want %d", captureRec.Code, http.StatusOK)
	}
	var checkpoint SessionCheckpoint
	if err := json.NewDecoder(captureRec.Body).Decode(&checkpoint); err != nil {
		t.Fatalf("decode checkpoint: %v", err)
	}
	if checkpoint.Session.ID != "sess-checkpoint" || string(checkpoint.State) != `{"cursor":7}` {
		t.Fatalf("checkpoint = %+v", checkpoint)
	}

	listReq := httptest.NewRequest(http.MethodGet, "/v1/tq/checkpoints", nil)
	listReq.Header.Set("Authorization", "Bearer sk-test")
	listRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(listRec, listReq)
	if listRec.Code != http.StatusOK {
		t.Fatalf("list status = %d want %d", listRec.Code, http.StatusOK)
	}
	var listed struct {
		Data []SessionCheckpoint `json:"data"`
	}
	if err := json.NewDecoder(listRec.Body).Decode(&listed); err != nil {
		t.Fatalf("decode listed checkpoints: %v", err)
	}
	if len(listed.Data) != 1 || listed.Data[0].Session.ID != "sess-checkpoint" {
		t.Fatalf("listed checkpoints = %+v", listed.Data)
	}

	restoreReq := httptest.NewRequest(http.MethodPost, "/v1/tq/checkpoints/restore", strings.NewReader(`{"session_id":"sess-restored","checkpoint":{"version":"tqserve.session.v1","captured_at":"`+checkpoint.CapturedAt.Format(time.RFC3339Nano)+`","session":{"id":"sess-checkpoint","model":"local-chat","backend":"default","last_endpoint":"chat_completions","created_at":"`+checkpoint.Session.CreatedAt.Format(time.RFC3339Nano)+`","last_seen_at":"`+checkpoint.Session.LastSeenAt.Format(time.RFC3339Nano)+`","request_count":1},"state":{"cursor":7}}}`))
	restoreReq.Header.Set("Authorization", "Bearer sk-test")
	restoreRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(restoreRec, restoreReq)
	if restoreRec.Code != http.StatusOK {
		t.Fatalf("restore status = %d want %d", restoreRec.Code, http.StatusOK)
	}
	if backend.restoredID != "sess-restored" || string(backend.restoredState) != `{"cursor":7}` {
		t.Fatalf("restored = (%q, %s)", backend.restoredID, backend.restoredState)
	}

	sessionsReq := httptest.NewRequest(http.MethodGet, "/v1/tq/sessions", nil)
	sessionsReq.Header.Set("Authorization", "Bearer sk-test")
	sessionsRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(sessionsRec, sessionsReq)
	if sessionsRec.Code != http.StatusOK {
		t.Fatalf("sessions status = %d want %d", sessionsRec.Code, http.StatusOK)
	}
	var sessions struct {
		Data []SessionInfo `json:"data"`
	}
	if err := json.NewDecoder(sessionsRec.Body).Decode(&sessions); err != nil {
		t.Fatalf("decode sessions: %v", err)
	}
	found := false
	for _, session := range sessions.Data {
		if session.ID == "sess-restored" {
			found = true
		}
	}
	if !found {
		t.Fatalf("restored sessions = %+v", sessions.Data)
	}
}

func TestManagedBackendFetchesRemoteStatusAndCheckpoint(t *testing.T) {
	statusSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			writeJSON(w, http.StatusOK, ModelList{
				Object: "list",
				Data:   []Model{{ID: "child/model", Object: "model", OwnedBy: "child"}},
			})
		case "/v1/tq/status":
			writeJSON(w, http.StatusOK, ServerStatus{
				OK: true,
				Backends: []BackendStatus{
					{
						Name:  "native",
						Kind:  "native",
						Ready: true,
						Capacity: &CapacitySnapshot{
							Accelerator:     "cuda",
							Device:          "L4",
							KVHeadroomBytes: 4 << 30,
							MaxSessions:     6,
						},
					},
				},
			})
		case "/v1/tq/checkpoints":
			writeJSON(w, http.StatusOK, SessionCheckpoint{
				Version:    "tqserve.session.v1",
				CapturedAt: time.Now().UTC(),
				Session: SessionInfo{
					ID:      "sess-remote",
					Model:   "local-chat",
					Backend: "default",
				},
				State: json.RawMessage(`{"cursor":9}`),
			})
		case "/v1/tq/checkpoints/restore":
			writeJSON(w, http.StatusOK, map[string]any{"ok": true})
		default:
			w.WriteHeader(http.StatusNotFound)
		}
	}))
	defer statusSrv.Close()

	if _, err := exec.LookPath("sleep"); err != nil {
		t.Skip("sleep not available")
	}
	backend, err := NewManagedBackend(ManagedBackendConfig{
		Name:            "managed-test",
		Kind:            "managed_upstream",
		BaseURL:         statusSrv.URL + "/v1",
		Command:         "sleep",
		Args:            []string{"60"},
		HealthURL:       statusSrv.URL + "/v1/models",
		StatusURL:       statusSrv.URL + "/v1/tq/status",
		CheckpointURL:   statusSrv.URL + "/v1/tq/checkpoints",
		RestoreURL:      statusSrv.URL + "/v1/tq/checkpoints/restore",
		StartupTimeout:  2 * time.Second,
		ShutdownTimeout: 100 * time.Millisecond,
	})
	if err != nil {
		t.Fatalf("NewManagedBackend: %v", err)
	}
	defer backend.Close()

	status := backend.Status(t.Context())
	if !status.Ready || status.Capacity == nil || status.Capacity.Device != "L4" || status.Capacity.MaxSessions != 6 {
		t.Fatalf("status = %+v", status)
	}
	state, err := backend.CaptureSessionState(t.Context(), "sess-remote")
	if err != nil {
		t.Fatalf("CaptureSessionState: %v", err)
	}
	if string(state) != `{"cursor":9}` {
		t.Fatalf("state = %s", state)
	}
	if err := backend.RestoreSessionState(t.Context(), "sess-remote", json.RawMessage(`{"cursor":10}`)); err != nil {
		t.Fatalf("RestoreSessionState: %v", err)
	}
}

func TestMetricsExposeCounters(t *testing.T) {
	srv := newTestServer(t, nil, map[string]string{"local-chat": "upstream/model-a"})
	modelsReq := httptest.NewRequest(http.MethodGet, "/v1/models", nil)
	modelsReq.Header.Set("Authorization", "Bearer sk-test")
	modelsRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(modelsRec, modelsReq)
	if modelsRec.Code != http.StatusOK {
		t.Fatalf("models status = %d want %d", modelsRec.Code, http.StatusOK)
	}

	metricsReq := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	metricsReq.Header.Set("Authorization", "Bearer sk-test")
	metricsRec := httptest.NewRecorder()
	srv.Handler().ServeHTTP(metricsRec, metricsReq)
	if metricsRec.Code != http.StatusOK {
		t.Fatalf("metrics status = %d want %d", metricsRec.Code, http.StatusOK)
	}
	text := metricsRec.Body.String()
	if !strings.Contains(text, `tqserve_requests_total{endpoint="models",backend="server",status_code="200"} 1`) {
		t.Fatalf("metrics = %s", text)
	}
	if !strings.Contains(text, "tqserve_backends_configured 1") {
		t.Fatalf("metrics = %s", text)
	}
}

func newTestServer(t *testing.T, backend Backend, modelMap map[string]string) *Server {
	t.Helper()
	if backend == nil {
		backend = testUpstreamBackend(t)
	}
	srv, err := New(Config{
		APIKeys:  []string{"sk-test"},
		ModelMap: modelMap,
		Backend:  backend,
	})
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return srv
}

func testUpstreamBackend(t *testing.T) Backend {
	t.Helper()
	upstream := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		switch r.URL.Path {
		case "/v1/models":
			writeJSON(w, http.StatusOK, ModelList{
				Object: "list",
				Data: []Model{
					{ID: "upstream/model-a", Object: "model", OwnedBy: "upstream"},
				},
			})
		case "/v1/chat/completions", "/v1/responses":
			w.Header().Set("Content-Type", "application/json")
			_, _ = io.WriteString(w, `{"id":"chatcmpl-test","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"ok"},"finish_reason":"stop"}]}`)
		default:
			w.WriteHeader(http.StatusNotImplemented)
		}
	}))
	t.Cleanup(upstream.Close)
	backend, err := NewUpstreamBackend(upstream.URL+"/v1", "", nil)
	if err != nil {
		t.Fatalf("NewUpstreamBackend: %v", err)
	}
	return backend
}

type statusCapacityBackend struct {
	models   ModelList
	status   BackendStatus
	capacity CapacitySnapshot
}

func (b *statusCapacityBackend) Models(ctx Context) (ModelList, error) {
	return b.models, nil
}

func (b *statusCapacityBackend) ChatCompletions(ctx Context, req RequestEnvelope, body []byte) (*BackendResponse, error) {
	return jsonResponse(http.StatusOK, map[string]any{"ok": true})
}

func (b *statusCapacityBackend) Responses(ctx Context, req RequestEnvelope, body []byte) (*BackendResponse, error) {
	return jsonResponse(http.StatusOK, map[string]any{"ok": true})
}

func (b *statusCapacityBackend) Status(ctx context.Context) BackendStatus {
	return b.status
}

func (b *statusCapacityBackend) Capacity(ctx context.Context) CapacitySnapshot {
	return b.capacity
}

type statefulBackend struct {
	models        ModelList
	restoredID    string
	restoredState json.RawMessage
}

func (b *statefulBackend) Models(ctx Context) (ModelList, error) {
	return b.models, nil
}

func (b *statefulBackend) ChatCompletions(ctx Context, req RequestEnvelope, body []byte) (*BackendResponse, error) {
	return jsonResponse(http.StatusOK, map[string]any{"id": "chatcmpl-test", "object": "chat.completion", "choices": []map[string]any{{"index": 0, "message": map[string]any{"role": "assistant", "content": "ok"}, "finish_reason": "stop"}}})
}

func (b *statefulBackend) Responses(ctx Context, req RequestEnvelope, body []byte) (*BackendResponse, error) {
	return jsonResponse(http.StatusOK, map[string]any{"ok": true})
}

func (b *statefulBackend) CaptureSessionState(ctx Context, sessionID string) (json.RawMessage, error) {
	return json.RawMessage(`{"cursor":7}`), nil
}

func (b *statefulBackend) RestoreSessionState(ctx Context, sessionID string, state json.RawMessage) error {
	b.restoredID = sessionID
	b.restoredState = append(json.RawMessage(nil), state...)
	return nil
}

type spySessionStore struct {
	mu       sync.Mutex
	sessions []SessionInfo
}

func (s *spySessionStore) Touch(id, model, backend, endpoint string, stream bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	now := time.Now().UTC()
	s.sessions = append(s.sessions, SessionInfo{
		ID:           id,
		Model:        model,
		Backend:      backend,
		LastEndpoint: endpoint,
		CreatedAt:    now,
		LastSeenAt:   now,
		RequestCount: 1,
		Stream:       stream,
	})
}

func (s *spySessionStore) Get(id string) (SessionInfo, bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, session := range s.sessions {
		if session.ID == id {
			return session, true
		}
	}
	return SessionInfo{}, false
}

func (s *spySessionStore) List() []SessionInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	items := make([]SessionInfo, len(s.sessions))
	copy(items, s.sessions)
	return items
}

func (s *spySessionStore) Count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.sessions)
}

func (s *spySessionStore) count() int {
	return s.Count()
}

func (s *spySessionStore) last() SessionInfo {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.sessions[len(s.sessions)-1]
}
