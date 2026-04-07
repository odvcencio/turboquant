package tqserve

import (
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"sync"
	"time"
)

const (
	DefaultSessionHeader  = "X-TQ-Session-ID"
	DefaultSessionIdleTTL = 30 * time.Minute
)

type SessionInfo struct {
	ID           string    `json:"id"`
	Model        string    `json:"model,omitempty"`
	Backend      string    `json:"backend,omitempty"`
	LastEndpoint string    `json:"last_endpoint,omitempty"`
	CreatedAt    time.Time `json:"created_at"`
	LastSeenAt   time.Time `json:"last_seen_at"`
	RequestCount uint64    `json:"request_count"`
	Stream       bool      `json:"stream,omitempty"`
}

var ErrUnknownSession = errors.New("tqserve: unknown session")

type SessionCheckpoint struct {
	Version    string          `json:"version"`
	CapturedAt time.Time       `json:"captured_at"`
	Session    SessionInfo     `json:"session"`
	State      json.RawMessage `json:"state,omitempty"`
}

type SessionStore interface {
	Touch(id, model, backend, endpoint string, stream bool)
	Get(id string) (SessionInfo, bool)
	List() []SessionInfo
	Count() int
}

type CheckpointStore interface {
	SaveCheckpoint(SessionCheckpoint) error
	GetCheckpoint(id string) (SessionCheckpoint, bool)
	ListCheckpoints() []SessionCheckpoint
	RestoreCheckpoint(SessionCheckpoint) error
}

type SessionStateProvider interface {
	CaptureSessionState(ctx Context, sessionID string) (json.RawMessage, error)
	RestoreSessionState(ctx Context, sessionID string, state json.RawMessage) error
}

type MemorySessionStore struct {
	mu          sync.Mutex
	idleTTL     time.Duration
	sessions    map[string]*SessionInfo
	checkpoints map[string]SessionCheckpoint
}

func NewMemorySessionStore(idleTTL time.Duration) *MemorySessionStore {
	return &MemorySessionStore{
		idleTTL:     defaultDuration(idleTTL, DefaultSessionIdleTTL),
		sessions:    make(map[string]*SessionInfo),
		checkpoints: make(map[string]SessionCheckpoint),
	}
}

func (r *MemorySessionStore) Touch(id, model, backend, endpoint string, stream bool) {
	id = strings.TrimSpace(id)
	if id == "" {
		return
	}
	now := time.Now().UTC()
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pruneLocked(now)
	session := r.sessions[id]
	if session == nil {
		session = &SessionInfo{
			ID:        id,
			CreatedAt: now,
		}
		r.sessions[id] = session
	}
	session.Model = strings.TrimSpace(model)
	session.Backend = strings.TrimSpace(backend)
	session.LastEndpoint = strings.TrimSpace(endpoint)
	session.LastSeenAt = now
	session.RequestCount++
	session.Stream = stream
}

func (r *MemorySessionStore) Get(id string) (SessionInfo, bool) {
	now := time.Now().UTC()
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pruneLocked(now)
	session := r.sessions[strings.TrimSpace(id)]
	if session == nil {
		return SessionInfo{}, false
	}
	return *session, true
}

func (r *MemorySessionStore) List() []SessionInfo {
	now := time.Now().UTC()
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pruneLocked(now)
	items := make([]SessionInfo, 0, len(r.sessions))
	for _, session := range r.sessions {
		items = append(items, *session)
	}
	slices.SortFunc(items, func(left, right SessionInfo) int {
		switch {
		case left.LastSeenAt.After(right.LastSeenAt):
			return -1
		case left.LastSeenAt.Before(right.LastSeenAt):
			return 1
		default:
			return strings.Compare(left.ID, right.ID)
		}
	})
	return items
}

func (r *MemorySessionStore) Count() int {
	now := time.Now().UTC()
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pruneLocked(now)
	return len(r.sessions)
}

func (r *MemorySessionStore) SaveCheckpoint(cp SessionCheckpoint) error {
	if strings.TrimSpace(cp.Session.ID) == "" {
		return ErrUnknownSession
	}
	now := time.Now().UTC()
	r.mu.Lock()
	defer r.mu.Unlock()
	if cp.Version == "" {
		cp.Version = "tqserve.session.v1"
	}
	if cp.CapturedAt.IsZero() {
		cp.CapturedAt = now
	}
	cp.Session = normalizedSessionInfo(cp.Session, now)
	r.checkpoints[cp.Session.ID] = cp
	return nil
}

func (r *MemorySessionStore) GetCheckpoint(id string) (SessionCheckpoint, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()
	cp, ok := r.checkpoints[strings.TrimSpace(id)]
	return cp, ok
}

func (r *MemorySessionStore) ListCheckpoints() []SessionCheckpoint {
	r.mu.Lock()
	defer r.mu.Unlock()
	items := make([]SessionCheckpoint, 0, len(r.checkpoints))
	for _, checkpoint := range r.checkpoints {
		items = append(items, checkpoint)
	}
	slices.SortFunc(items, func(left, right SessionCheckpoint) int {
		switch {
		case left.CapturedAt.After(right.CapturedAt):
			return -1
		case left.CapturedAt.Before(right.CapturedAt):
			return 1
		default:
			return strings.Compare(left.Session.ID, right.Session.ID)
		}
	})
	return items
}

func (r *MemorySessionStore) RestoreCheckpoint(cp SessionCheckpoint) error {
	if strings.TrimSpace(cp.Session.ID) == "" {
		return ErrUnknownSession
	}
	now := time.Now().UTC()
	cp.Session = normalizedSessionInfo(cp.Session, now)
	if cp.Version == "" {
		cp.Version = "tqserve.session.v1"
	}
	if cp.CapturedAt.IsZero() {
		cp.CapturedAt = now
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.pruneLocked(now)
	session := cp.Session
	r.sessions[session.ID] = &session
	r.checkpoints[session.ID] = cp
	return nil
}

func (r *MemorySessionStore) pruneLocked(now time.Time) {
	for id, session := range r.sessions {
		if now.Sub(session.LastSeenAt) > r.idleTTL {
			delete(r.sessions, id)
		}
	}
}

func normalizedSessionInfo(session SessionInfo, now time.Time) SessionInfo {
	session.ID = strings.TrimSpace(session.ID)
	session.Model = strings.TrimSpace(session.Model)
	session.Backend = strings.TrimSpace(session.Backend)
	session.LastEndpoint = strings.TrimSpace(session.LastEndpoint)
	if session.CreatedAt.IsZero() {
		session.CreatedAt = now
	}
	if session.LastSeenAt.IsZero() {
		session.LastSeenAt = now
	}
	if session.RequestCount == 0 {
		session.RequestCount = 1
	}
	return session
}
