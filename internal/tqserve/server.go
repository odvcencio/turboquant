package tqserve

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"time"

	turboquant "github.com/odvcencio/turboquant"
)

var ErrUnknownModel = errors.New("tqserve: unknown model")

type Config struct {
	APIKeys         []string
	Backend         Backend
	Backends        map[string]Backend
	ModelMap        map[string]string
	Routes          []ModelRoute
	DefaultOwn      string
	RequireAuth     bool
	SessionHeader   string
	SessionIdleTTL  time.Duration
	SessionStore    SessionStore
	CheckpointStore CheckpointStore
	Collaboration   CollaborationStore
}

type Server struct {
	apiKeys        map[string]struct{}
	backends       map[string]Backend
	routes         map[string]ModelRoute
	backend        Backend
	defaultOwn     string
	sessionHeader  string
	sessionIdleTTL time.Duration
	sessions       SessionStore
	checkpoints    CheckpointStore
	collaboration  CollaborationStore
	metrics        *metricsState
}

func New(cfg Config) (*Server, error) {
	backends := make(map[string]Backend, len(cfg.Backends)+1)
	for name, backend := range cfg.Backends {
		name = strings.TrimSpace(name)
		if name == "" || backend == nil {
			return nil, fmt.Errorf("tqserve: invalid backend %q", name)
		}
		backends[name] = backend
	}
	if cfg.Backend != nil {
		backends["default"] = cfg.Backend
	}
	if len(backends) == 0 {
		return nil, fmt.Errorf("tqserve: backend is required")
	}
	keys := make(map[string]struct{}, len(cfg.APIKeys))
	for _, key := range cfg.APIKeys {
		key = strings.TrimSpace(key)
		if key == "" {
			continue
		}
		keys[key] = struct{}{}
	}
	routes := make(map[string]ModelRoute, len(cfg.Routes)+len(cfg.ModelMap))
	for _, route := range cfg.Routes {
		route.PublicModel = strings.TrimSpace(route.PublicModel)
		route.BackendName = strings.TrimSpace(route.BackendName)
		route.BackendModel = strings.TrimSpace(route.BackendModel)
		route.OwnedBy = strings.TrimSpace(route.OwnedBy)
		if route.PublicModel == "" || route.BackendName == "" {
			return nil, fmt.Errorf("tqserve: invalid route %+v", route)
		}
		if _, ok := backends[route.BackendName]; !ok {
			return nil, fmt.Errorf("tqserve: route %q references unknown backend %q", route.PublicModel, route.BackendName)
		}
		if route.BackendModel == "" {
			route.BackendModel = route.PublicModel
		}
		routes[route.PublicModel] = route
	}
	for public, backendModel := range cfg.ModelMap {
		public = strings.TrimSpace(public)
		backendModel = strings.TrimSpace(backendModel)
		if public == "" || backendModel == "" {
			return nil, fmt.Errorf("tqserve: invalid model mapping %q=%q", public, backendModel)
		}
		if _, ok := routes[public]; ok {
			return nil, fmt.Errorf("tqserve: duplicate route for model %q", public)
		}
		routes[public] = ModelRoute{
			PublicModel:  public,
			BackendName:  "default",
			BackendModel: backendModel,
		}
	}
	var defaultBackend Backend
	if len(backends) == 1 {
		for _, backend := range backends {
			defaultBackend = backend
		}
	} else if len(routes) == 0 {
		return nil, fmt.Errorf("tqserve: routes are required when multiple backends are configured")
	}
	sessionStore := cfg.SessionStore
	if sessionStore == nil {
		sessionStore = NewMemorySessionStore(cfg.SessionIdleTTL)
	}
	checkpointStore := cfg.CheckpointStore
	if checkpointStore == nil {
		if store, ok := sessionStore.(CheckpointStore); ok {
			checkpointStore = store
		}
	}
	collaborationStore := cfg.Collaboration
	if collaborationStore == nil {
		collaborationStore = NewMemoryCollaborationStore(cfg.SessionIdleTTL)
	}
	return &Server{
		apiKeys:        keys,
		backends:       backends,
		routes:         routes,
		backend:        defaultBackend,
		defaultOwn:     defaultString(cfg.DefaultOwn, "turboquant"),
		sessionHeader:  defaultString(cfg.SessionHeader, DefaultSessionHeader),
		sessionIdleTTL: defaultDuration(cfg.SessionIdleTTL, DefaultSessionIdleTTL),
		sessions:       sessionStore,
		checkpoints:    checkpointStore,
		collaboration:  collaborationStore,
		metrics:        newMetricsState(),
	}, nil
}

func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("/healthz", s.handleHealth)
	mux.HandleFunc("/v1/models", s.handleModels)
	mux.HandleFunc("/v1/chat/completions", s.handleChatCompletions)
	mux.HandleFunc("/v1/responses", s.handleResponses)
	mux.HandleFunc("/v1/tq/status", s.handleStatus)
	mux.HandleFunc("/v1/tq/sessions", s.handleSessions)
	mux.HandleFunc("/v1/tq/agents", s.handleAgents)
	mux.HandleFunc("/v1/tq/claims", s.handleClaims)
	mux.HandleFunc("/v1/tq/events", s.handleEvents)
	mux.HandleFunc("/v1/tq/checkpoints", s.handleCheckpoints)
	mux.HandleFunc("/v1/tq/checkpoints/restore", s.handleCheckpointRestore)
	mux.HandleFunc("/metrics", s.handleMetrics)
	return mux
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeMethodNotAllowed(w)
		s.metrics.Record("healthz", "server", http.StatusMethodNotAllowed)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"ok":      true,
		"service": "tqserve",
		"version": turboquant.Version,
	})
	s.metrics.Record("healthz", "server", http.StatusOK)
}

func (s *Server) handleModels(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeMethodNotAllowed(w)
		s.metrics.Record("models", "server", http.StatusMethodNotAllowed)
		return
	}
	if !s.authorized(r) {
		writeAuthError(w)
		s.metrics.Record("models", "server", http.StatusUnauthorized)
		return
	}
	ctx := r.Context()
	models, err := s.models(ctx)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error(), "backend_error", "")
		s.metrics.Record("models", "server", http.StatusBadGateway)
		return
	}
	writeJSON(w, http.StatusOK, models)
	s.metrics.Record("models", "server", http.StatusOK)
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	s.handleProxy(w, r, func(ctx context.Context, backend Backend, req RequestEnvelope, body []byte) (*BackendResponse, error) {
		return backend.ChatCompletions(ctx, req, body)
	})
}

func (s *Server) handleResponses(w http.ResponseWriter, r *http.Request) {
	s.handleProxy(w, r, func(ctx context.Context, backend Backend, req RequestEnvelope, body []byte) (*BackendResponse, error) {
		return backend.Responses(ctx, req, body)
	})
}

func (s *Server) handleProxy(w http.ResponseWriter, r *http.Request, fn func(context.Context, Backend, RequestEnvelope, []byte) (*BackendResponse, error)) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w)
		s.metrics.Record(requestName(r.URL.Path), "server", http.StatusMethodNotAllowed)
		return
	}
	if !s.authorized(r) {
		writeAuthError(w)
		s.metrics.Record(requestName(r.URL.Path), "server", http.StatusUnauthorized)
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, 16<<20))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request body", "invalid_request_error", "")
		s.metrics.Record(requestName(r.URL.Path), "server", http.StatusBadRequest)
		return
	}
	resolved, err := s.resolveRequest(body)
	if err != nil {
		status := http.StatusBadRequest
		code := "invalid_request_error"
		if errors.Is(err, ErrUnknownModel) {
			status = http.StatusNotFound
			code = "model_not_found"
		}
		writeError(w, status, err.Error(), code, "model")
		s.metrics.Record(requestName(r.URL.Path), "server", status)
		return
	}
	s.sessions.Touch(r.Header.Get(s.sessionHeader), resolved.PublicModel, resolved.BackendName, requestName(r.URL.Path), resolved.Request.Stream)
	resolved.Request.SessionID = strings.TrimSpace(r.Header.Get(s.sessionHeader))
	resp, err := fn(r.Context(), resolved.Backend, resolved.Request, resolved.Body)
	if err != nil {
		writeError(w, http.StatusBadGateway, err.Error(), "backend_error", "")
		s.metrics.Record(requestName(r.URL.Path), resolved.BackendName, http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	copyResponse(w, resp)
	s.metrics.Record(requestName(r.URL.Path), resolved.BackendName, resp.StatusCode)
}

type resolvedRequest struct {
	Request     RequestEnvelope
	Body        []byte
	Backend     Backend
	BackendName string
	PublicModel string
}

func (s *Server) resolveRequest(body []byte) (resolvedRequest, error) {
	var req RequestEnvelope
	if err := json.Unmarshal(body, &req); err != nil {
		return resolvedRequest{}, fmt.Errorf("tqserve: invalid JSON request body")
	}
	if strings.TrimSpace(req.Model) == "" {
		return resolvedRequest{}, fmt.Errorf("tqserve: request model is required")
	}
	if len(s.routes) == 0 {
		if s.backend == nil {
			return resolvedRequest{}, fmt.Errorf("tqserve: no backend available")
		}
		return resolvedRequest{
			Request:     req,
			Body:        body,
			Backend:     s.backend,
			BackendName: "default",
			PublicModel: req.Model,
		}, nil
	}
	route, ok := s.routes[req.Model]
	if !ok {
		return resolvedRequest{}, fmt.Errorf("%w %q", ErrUnknownModel, req.Model)
	}
	backend := s.backends[route.BackendName]
	if backend == nil {
		return resolvedRequest{}, fmt.Errorf("tqserve: backend %q is unavailable", route.BackendName)
	}
	if route.BackendModel == req.Model {
		return resolvedRequest{
			Request:     req,
			Body:        body,
			Backend:     backend,
			BackendName: route.BackendName,
			PublicModel: route.PublicModel,
		}, nil
	}
	var payload map[string]any
	if err := json.Unmarshal(body, &payload); err != nil {
		return resolvedRequest{}, fmt.Errorf("tqserve: invalid JSON request body")
	}
	payload["model"] = route.BackendModel
	rewritten, err := json.Marshal(payload)
	if err != nil {
		return resolvedRequest{}, err
	}
	req.Model = route.BackendModel
	return resolvedRequest{
		Request:     req,
		Body:        rewritten,
		Backend:     backend,
		BackendName: route.BackendName,
		PublicModel: route.PublicModel,
	}, nil
}

func (s *Server) models(ctx context.Context) (ModelList, error) {
	if len(s.routes) == 0 {
		return s.backend.Models(ctx)
	}
	names := make([]string, 0, len(s.routes))
	for name := range s.routes {
		names = append(names, name)
	}
	slices.Sort(names)
	models := ModelList{
		Object: "list",
		Data:   make([]Model, len(names)),
	}
	for i, name := range names {
		route := s.routes[name]
		ownedBy := route.OwnedBy
		if ownedBy == "" {
			ownedBy = s.defaultOwn
		}
		models.Data[i] = Model{
			ID:      name,
			Object:  "model",
			OwnedBy: ownedBy,
		}
	}
	return models, nil
}

func (s *Server) handleStatus(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeMethodNotAllowed(w)
		s.metrics.Record("status", "server", http.StatusMethodNotAllowed)
		return
	}
	if !s.authorized(r) {
		writeAuthError(w)
		s.metrics.Record("status", "server", http.StatusUnauthorized)
		return
	}
	writeJSON(w, http.StatusOK, s.status(r.Context()))
	s.metrics.Record("status", "server", http.StatusOK)
}

func (s *Server) handleSessions(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeMethodNotAllowed(w)
		s.metrics.Record("sessions", "server", http.StatusMethodNotAllowed)
		return
	}
	if !s.authorized(r) {
		writeAuthError(w)
		s.metrics.Record("sessions", "server", http.StatusUnauthorized)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object":         "list",
		"data":           s.sessions.List(),
		"session_header": s.sessionHeader,
		"idle_ttl":       s.sessionIdleTTL.String(),
	})
	s.metrics.Record("sessions", "server", http.StatusOK)
}

func (s *Server) handleAgents(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeAuthError(w)
		s.metrics.Record("agents", "server", http.StatusUnauthorized)
		return
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, map[string]any{
			"object": "list",
			"data":   s.collaboration.ListAgents(r.URL.Query().Get("session_id")),
		})
		s.metrics.Record("agents", "server", http.StatusOK)
	case http.MethodPost:
		var agent AgentPresence
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&agent); err != nil {
			writeError(w, http.StatusBadRequest, "invalid agent request body", "invalid_request_error", "")
			s.metrics.Record("agents", "server", http.StatusBadRequest)
			return
		}
		record, err := s.collaboration.UpsertAgent(agent)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "")
			s.metrics.Record("agents", "server", http.StatusBadRequest)
			return
		}
		writeJSON(w, http.StatusOK, record)
		s.metrics.Record("agents", "server", http.StatusOK)
	case http.MethodDelete:
		if !s.collaboration.RemoveAgent(r.URL.Query().Get("session_id"), r.URL.Query().Get("agent_id")) {
			writeError(w, http.StatusNotFound, "agent not found", "not_found", "agent_id")
			s.metrics.Record("agents", "server", http.StatusNotFound)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
		s.metrics.Record("agents", "server", http.StatusOK)
	default:
		writeMethodNotAllowed(w)
		s.metrics.Record("agents", "server", http.StatusMethodNotAllowed)
	}
}

func (s *Server) handleClaims(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeAuthError(w)
		s.metrics.Record("claims", "server", http.StatusUnauthorized)
		return
	}
	switch r.Method {
	case http.MethodGet:
		writeJSON(w, http.StatusOK, map[string]any{
			"object": "list",
			"data":   s.collaboration.ListClaims(r.URL.Query().Get("session_id")),
		})
		s.metrics.Record("claims", "server", http.StatusOK)
	case http.MethodPost:
		var claim AgentClaim
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&claim); err != nil {
			writeError(w, http.StatusBadRequest, "invalid claim request body", "invalid_request_error", "")
			s.metrics.Record("claims", "server", http.StatusBadRequest)
			return
		}
		record, err := s.collaboration.PutClaim(claim)
		if err != nil {
			status := http.StatusBadRequest
			code := "invalid_request_error"
			if errors.Is(err, ErrClaimConflict) {
				status = http.StatusConflict
				code = "claim_conflict"
			}
			writeError(w, status, err.Error(), code, "entity")
			s.metrics.Record("claims", "server", status)
			return
		}
		writeJSON(w, http.StatusOK, record)
		s.metrics.Record("claims", "server", http.StatusOK)
	case http.MethodDelete:
		if !s.collaboration.RemoveClaim(r.URL.Query().Get("session_id"), r.URL.Query().Get("entity"), r.URL.Query().Get("agent_id")) {
			writeError(w, http.StatusNotFound, "claim not found", "not_found", "entity")
			s.metrics.Record("claims", "server", http.StatusNotFound)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{"ok": true})
		s.metrics.Record("claims", "server", http.StatusOK)
	default:
		writeMethodNotAllowed(w)
		s.metrics.Record("claims", "server", http.StatusMethodNotAllowed)
	}
}

func (s *Server) handleEvents(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeAuthError(w)
		s.metrics.Record("events", "server", http.StatusUnauthorized)
		return
	}
	switch r.Method {
	case http.MethodGet:
		sessionID := strings.TrimSpace(r.URL.Query().Get("session_id"))
		if sessionID == "" {
			writeError(w, http.StatusBadRequest, "session_id is required", "invalid_request_error", "session_id")
			s.metrics.Record("events", "server", http.StatusBadRequest)
			return
		}
		after, err := parseUint64(r.URL.Query().Get("after"))
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "after")
			s.metrics.Record("events", "server", http.StatusBadRequest)
			return
		}
		limit, err := parseOptionalInt(r.URL.Query().Get("limit"))
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "limit")
			s.metrics.Record("events", "server", http.StatusBadRequest)
			return
		}
		writeJSON(w, http.StatusOK, map[string]any{
			"object": "list",
			"data":   s.collaboration.ListEvents(sessionID, after, limit),
		})
		s.metrics.Record("events", "server", http.StatusOK)
	case http.MethodPost:
		var event CollaborationEvent
		if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&event); err != nil {
			writeError(w, http.StatusBadRequest, "invalid event request body", "invalid_request_error", "")
			s.metrics.Record("events", "server", http.StatusBadRequest)
			return
		}
		record, err := s.collaboration.AppendEvent(event)
		if err != nil {
			writeError(w, http.StatusBadRequest, err.Error(), "invalid_request_error", "")
			s.metrics.Record("events", "server", http.StatusBadRequest)
			return
		}
		writeJSON(w, http.StatusOK, record)
		s.metrics.Record("events", "server", http.StatusOK)
	default:
		writeMethodNotAllowed(w)
		s.metrics.Record("events", "server", http.StatusMethodNotAllowed)
	}
}

func (s *Server) handleMetrics(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		writeMethodNotAllowed(w)
		s.metrics.Record("metrics", "server", http.StatusMethodNotAllowed)
		return
	}
	if !s.authorized(r) {
		writeAuthError(w)
		s.metrics.Record("metrics", "server", http.StatusUnauthorized)
		return
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	w.WriteHeader(http.StatusOK)
	s.metrics.WritePrometheus(w, s.sessions.Count(), s.collaboration.AgentCount(), s.collaboration.ClaimCount(), len(s.backends), len(s.routes))
	s.metrics.Record("metrics", "server", http.StatusOK)
}

func (s *Server) handleCheckpoints(w http.ResponseWriter, r *http.Request) {
	switch r.Method {
	case http.MethodGet:
		s.handleCheckpointList(w, r)
	case http.MethodPost:
		s.handleCheckpointCapture(w, r)
	default:
		writeMethodNotAllowed(w)
		s.metrics.Record("checkpoints", "server", http.StatusMethodNotAllowed)
	}
}

func (s *Server) handleCheckpointList(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeAuthError(w)
		s.metrics.Record("checkpoints", "server", http.StatusUnauthorized)
		return
	}
	if s.checkpoints == nil {
		writeError(w, http.StatusNotImplemented, "checkpoint store is not configured", "not_supported_error", "")
		s.metrics.Record("checkpoints", "server", http.StatusNotImplemented)
		return
	}
	writeJSON(w, http.StatusOK, map[string]any{
		"object": "list",
		"data":   s.checkpoints.ListCheckpoints(),
	})
	s.metrics.Record("checkpoints", "server", http.StatusOK)
}

func (s *Server) handleCheckpointCapture(w http.ResponseWriter, r *http.Request) {
	if !s.authorized(r) {
		writeAuthError(w)
		s.metrics.Record("checkpoints", "server", http.StatusUnauthorized)
		return
	}
	var req struct {
		SessionID string `json:"session_id"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 1<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid checkpoint request body", "invalid_request_error", "")
		s.metrics.Record("checkpoints", "server", http.StatusBadRequest)
		return
	}
	checkpoint, err := s.captureCheckpoint(r.Context(), req.SessionID)
	if err != nil {
		status := http.StatusBadRequest
		code := "invalid_request_error"
		if errors.Is(err, ErrUnknownSession) {
			status = http.StatusNotFound
			code = "session_not_found"
		}
		writeError(w, status, err.Error(), code, "session_id")
		s.metrics.Record("checkpoints", "server", status)
		return
	}
	writeJSON(w, http.StatusOK, checkpoint)
	s.metrics.Record("checkpoints", "server", http.StatusOK)
}

func (s *Server) handleCheckpointRestore(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodPost {
		writeMethodNotAllowed(w)
		s.metrics.Record("checkpoints_restore", "server", http.StatusMethodNotAllowed)
		return
	}
	if !s.authorized(r) {
		writeAuthError(w)
		s.metrics.Record("checkpoints_restore", "server", http.StatusUnauthorized)
		return
	}
	var req struct {
		Checkpoint SessionCheckpoint `json:"checkpoint"`
		SessionID  string            `json:"session_id,omitempty"`
	}
	if err := json.NewDecoder(io.LimitReader(r.Body, 4<<20)).Decode(&req); err != nil {
		writeError(w, http.StatusBadRequest, "invalid checkpoint restore body", "invalid_request_error", "")
		s.metrics.Record("checkpoints_restore", "server", http.StatusBadRequest)
		return
	}
	if override := strings.TrimSpace(req.SessionID); override != "" {
		req.Checkpoint.Session.ID = override
	}
	if err := s.restoreCheckpoint(r.Context(), req.Checkpoint); err != nil {
		status := http.StatusBadRequest
		code := "invalid_request_error"
		if errors.Is(err, ErrUnknownSession) {
			status = http.StatusNotFound
			code = "session_not_found"
		}
		writeError(w, status, err.Error(), code, "checkpoint")
		s.metrics.Record("checkpoints_restore", "server", status)
		return
	}
	writeJSON(w, http.StatusOK, req.Checkpoint)
	s.metrics.Record("checkpoints_restore", "server", http.StatusOK)
}

func (s *Server) authorized(r *http.Request) bool {
	if len(s.apiKeys) == 0 {
		return true
	}
	const prefix = "Bearer "
	auth := r.Header.Get("Authorization")
	if !strings.HasPrefix(auth, prefix) {
		return false
	}
	_, ok := s.apiKeys[strings.TrimSpace(strings.TrimPrefix(auth, prefix))]
	return ok
}

func copyResponse(w http.ResponseWriter, resp *BackendResponse) {
	for key, values := range resp.Header {
		if hopByHopHeader(key) {
			continue
		}
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
	if w.Header().Get("Content-Type") == "" {
		w.Header().Set("Content-Type", "application/json")
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}

func writeJSON(w http.ResponseWriter, status int, value any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(value)
}

func writeError(w http.ResponseWriter, status int, message, typ, param string) {
	writeJSON(w, status, map[string]any{
		"error": map[string]any{
			"message": message,
			"type":    typ,
			"param":   emptyToNil(param),
		},
	})
}

func writeMethodNotAllowed(w http.ResponseWriter) {
	writeError(w, http.StatusMethodNotAllowed, "method not allowed", "invalid_request_error", "")
}

func writeAuthError(w http.ResponseWriter) {
	w.Header().Set("WWW-Authenticate", "Bearer")
	writeError(w, http.StatusUnauthorized, "missing or invalid API key", "authentication_error", "")
}

func hopByHopHeader(key string) bool {
	switch http.CanonicalHeaderKey(key) {
	case "Connection", "Keep-Alive", "Proxy-Authenticate", "Proxy-Authorization", "Te", "Trailer", "Transfer-Encoding", "Upgrade":
		return true
	default:
		return false
	}
}

func defaultString(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}

func parseUint64(raw string) (uint64, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, nil
	}
	var value uint64
	if _, err := fmt.Sscanf(raw, "%d", &value); err != nil {
		return 0, fmt.Errorf("tqserve: invalid uint64 %q", raw)
	}
	return value, nil
}

func parseOptionalInt(raw string) (int, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, nil
	}
	var value int
	if _, err := fmt.Sscanf(raw, "%d", &value); err != nil {
		return 0, fmt.Errorf("tqserve: invalid int %q", raw)
	}
	return value, nil
}

func emptyToNil(value string) any {
	if value == "" {
		return nil
	}
	return value
}

func requestName(path string) string {
	switch path {
	case "/v1/chat/completions":
		return "chat_completions"
	case "/v1/responses":
		return "responses"
	default:
		return strings.Trim(strings.ReplaceAll(path, "/", "_"), "_")
	}
}

func (s *Server) captureCheckpoint(ctx context.Context, sessionID string) (SessionCheckpoint, error) {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return SessionCheckpoint{}, fmt.Errorf("tqserve: session_id is required")
	}
	session, ok := s.sessions.Get(sessionID)
	if !ok {
		return SessionCheckpoint{}, fmt.Errorf("%w %q", ErrUnknownSession, sessionID)
	}
	checkpoint := SessionCheckpoint{
		Version:    "tqserve.session.v1",
		CapturedAt: time.Now().UTC(),
		Session:    session,
	}
	if backend := s.backends[session.Backend]; backend != nil {
		if provider, ok := backend.(SessionStateProvider); ok {
			state, err := provider.CaptureSessionState(ctx, session.ID)
			if err != nil {
				return SessionCheckpoint{}, err
			}
			checkpoint.State = state
		}
	}
	if s.checkpoints != nil {
		if err := s.checkpoints.SaveCheckpoint(checkpoint); err != nil {
			return SessionCheckpoint{}, err
		}
	}
	return checkpoint, nil
}

func (s *Server) restoreCheckpoint(ctx context.Context, checkpoint SessionCheckpoint) error {
	checkpoint.Session.ID = strings.TrimSpace(checkpoint.Session.ID)
	if checkpoint.Session.ID == "" {
		return fmt.Errorf("tqserve: checkpoint session id is required")
	}
	if checkpoint.Session.Backend == "" {
		checkpoint.Session.Backend = "default"
	}
	backend := s.backends[checkpoint.Session.Backend]
	if len(checkpoint.State) > 0 {
		if backend == nil {
			return fmt.Errorf("tqserve: backend %q is unavailable", checkpoint.Session.Backend)
		}
		provider, ok := backend.(SessionStateProvider)
		if !ok {
			return fmt.Errorf("tqserve: backend %q does not support session restore", checkpoint.Session.Backend)
		}
		if err := provider.RestoreSessionState(ctx, checkpoint.Session.ID, checkpoint.State); err != nil {
			return err
		}
	}
	if s.checkpoints != nil {
		return s.checkpoints.RestoreCheckpoint(checkpoint)
	}
	s.sessions.Touch(checkpoint.Session.ID, checkpoint.Session.Model, checkpoint.Session.Backend, checkpoint.Session.LastEndpoint, checkpoint.Session.Stream)
	return nil
}
