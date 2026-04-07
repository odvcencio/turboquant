package tqserve

import (
	"encoding/json"
	"errors"
	"slices"
	"strings"
	"sync"
	"time"
)

const defaultEventLimit = 100
const maxSessionEvents = 512

var ErrClaimConflict = errors.New("tqserve: claim conflict")

type AgentPresence struct {
	SessionID     string    `json:"session_id"`
	AgentID       string    `json:"agent_id"`
	ParentAgentID string    `json:"parent_agent_id,omitempty"`
	Role          string    `json:"role,omitempty"`
	Branch        string    `json:"branch,omitempty"`
	CheckpointID  string    `json:"checkpoint_id,omitempty"`
	Status        string    `json:"status,omitempty"`
	Capabilities  []string  `json:"capabilities,omitempty"`
	CreatedAt     time.Time `json:"created_at"`
	UpdatedAt     time.Time `json:"updated_at"`
}

type AgentClaim struct {
	SessionID  string    `json:"session_id"`
	Entity     string    `json:"entity"`
	AgentID    string    `json:"agent_id"`
	Mode       string    `json:"mode,omitempty"`
	Scope      string    `json:"scope,omitempty"`
	AcquiredAt time.Time `json:"acquired_at"`
	UpdatedAt  time.Time `json:"updated_at"`
}

type CollaborationEvent struct {
	SessionID string          `json:"session_id"`
	Seq       uint64          `json:"seq"`
	Type      string          `json:"type"`
	AgentID   string          `json:"agent_id,omitempty"`
	Entity    string          `json:"entity,omitempty"`
	Summary   string          `json:"summary,omitempty"`
	Data      json.RawMessage `json:"data,omitempty"`
	CreatedAt time.Time       `json:"created_at"`
}

type CollaborationStore interface {
	UpsertAgent(AgentPresence) (AgentPresence, error)
	RemoveAgent(sessionID, agentID string) bool
	ListAgents(sessionID string) []AgentPresence
	PutClaim(AgentClaim) (AgentClaim, error)
	RemoveClaim(sessionID, entity, agentID string) bool
	ListClaims(sessionID string) []AgentClaim
	AppendEvent(CollaborationEvent) (CollaborationEvent, error)
	ListEvents(sessionID string, after uint64, limit int) []CollaborationEvent
	AgentCount() int
	ClaimCount() int
}

type MemoryCollaborationStore struct {
	mu      sync.Mutex
	idleTTL time.Duration
	agents  map[string]map[string]*AgentPresence
	claims  map[string]map[string]map[string]*AgentClaim
	events  map[string][]CollaborationEvent
	nextSeq map[string]uint64
}

func NewMemoryCollaborationStore(idleTTL time.Duration) *MemoryCollaborationStore {
	return &MemoryCollaborationStore{
		idleTTL: defaultDuration(idleTTL, DefaultSessionIdleTTL),
		agents:  make(map[string]map[string]*AgentPresence),
		claims:  make(map[string]map[string]map[string]*AgentClaim),
		events:  make(map[string][]CollaborationEvent),
		nextSeq: make(map[string]uint64),
	}
}

func (s *MemoryCollaborationStore) UpsertAgent(agent AgentPresence) (AgentPresence, error) {
	now := time.Now().UTC()
	agent, err := normalizeAgentPresence(agent, now)
	if err != nil {
		return AgentPresence{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked(now)
	sessionAgents := s.agents[agent.SessionID]
	if sessionAgents == nil {
		sessionAgents = make(map[string]*AgentPresence)
		s.agents[agent.SessionID] = sessionAgents
	}
	if existing := sessionAgents[agent.AgentID]; existing != nil {
		agent.CreatedAt = existing.CreatedAt
	}
	copied := agent
	sessionAgents[agent.AgentID] = &copied
	s.appendEventLocked(CollaborationEvent{
		SessionID: agent.SessionID,
		Type:      "agent.upsert",
		AgentID:   agent.AgentID,
		Summary:   defaultString(agent.Status, "active"),
	})
	return copied, nil
}

func (s *MemoryCollaborationStore) RemoveAgent(sessionID, agentID string) bool {
	sessionID = strings.TrimSpace(sessionID)
	agentID = strings.TrimSpace(agentID)
	if sessionID == "" || agentID == "" {
		return false
	}
	now := time.Now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked(now)
	sessionAgents := s.agents[sessionID]
	if sessionAgents == nil || sessionAgents[agentID] == nil {
		return false
	}
	delete(sessionAgents, agentID)
	s.releaseClaimsLocked(sessionID, agentID)
	s.appendEventLocked(CollaborationEvent{
		SessionID: sessionID,
		Type:      "agent.remove",
		AgentID:   agentID,
	})
	if len(sessionAgents) == 0 {
		delete(s.agents, sessionID)
	}
	return true
}

func (s *MemoryCollaborationStore) ListAgents(sessionID string) []AgentPresence {
	now := time.Now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked(now)
	items := make([]AgentPresence, 0)
	if sessionID = strings.TrimSpace(sessionID); sessionID != "" {
		for _, agent := range s.agents[sessionID] {
			items = append(items, *agent)
		}
	} else {
		for _, sessionAgents := range s.agents {
			for _, agent := range sessionAgents {
				items = append(items, *agent)
			}
		}
	}
	slices.SortFunc(items, func(left, right AgentPresence) int {
		if left.SessionID != right.SessionID {
			return strings.Compare(left.SessionID, right.SessionID)
		}
		if left.UpdatedAt.After(right.UpdatedAt) {
			return -1
		}
		if left.UpdatedAt.Before(right.UpdatedAt) {
			return 1
		}
		return strings.Compare(left.AgentID, right.AgentID)
	})
	return items
}

func (s *MemoryCollaborationStore) PutClaim(claim AgentClaim) (AgentClaim, error) {
	now := time.Now().UTC()
	claim, err := normalizeAgentClaim(claim, now)
	if err != nil {
		return AgentClaim{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked(now)
	sessionClaims := s.claims[claim.SessionID]
	if sessionClaims == nil {
		sessionClaims = make(map[string]map[string]*AgentClaim)
		s.claims[claim.SessionID] = sessionClaims
	}
	entityClaims := sessionClaims[claim.Entity]
	if entityClaims == nil {
		entityClaims = make(map[string]*AgentClaim)
		sessionClaims[claim.Entity] = entityClaims
	}
	for ownerID, existing := range entityClaims {
		if ownerID == claim.AgentID {
			continue
		}
		if existing.Mode == "exclusive" || claim.Mode == "exclusive" {
			s.appendEventLocked(CollaborationEvent{
				SessionID: claim.SessionID,
				Type:      "claim.conflict",
				AgentID:   claim.AgentID,
				Entity:    claim.Entity,
				Summary:   ownerID,
			})
			return AgentClaim{}, ErrClaimConflict
		}
	}
	if existing := entityClaims[claim.AgentID]; existing != nil {
		claim.AcquiredAt = existing.AcquiredAt
	}
	copied := claim
	entityClaims[claim.AgentID] = &copied
	s.appendEventLocked(CollaborationEvent{
		SessionID: claim.SessionID,
		Type:      "claim.acquire",
		AgentID:   claim.AgentID,
		Entity:    claim.Entity,
		Summary:   claim.Mode,
	})
	return copied, nil
}

func (s *MemoryCollaborationStore) RemoveClaim(sessionID, entity, agentID string) bool {
	sessionID = strings.TrimSpace(sessionID)
	entity = strings.TrimSpace(entity)
	agentID = strings.TrimSpace(agentID)
	if sessionID == "" || entity == "" || agentID == "" {
		return false
	}
	now := time.Now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked(now)
	sessionClaims := s.claims[sessionID]
	if sessionClaims == nil {
		return false
	}
	entityClaims := sessionClaims[entity]
	if entityClaims == nil || entityClaims[agentID] == nil {
		return false
	}
	delete(entityClaims, agentID)
	s.appendEventLocked(CollaborationEvent{
		SessionID: sessionID,
		Type:      "claim.release",
		AgentID:   agentID,
		Entity:    entity,
	})
	if len(entityClaims) == 0 {
		delete(sessionClaims, entity)
	}
	if len(sessionClaims) == 0 {
		delete(s.claims, sessionID)
	}
	return true
}

func (s *MemoryCollaborationStore) ListClaims(sessionID string) []AgentClaim {
	now := time.Now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked(now)
	items := make([]AgentClaim, 0)
	if sessionID = strings.TrimSpace(sessionID); sessionID != "" {
		for _, entityClaims := range s.claims[sessionID] {
			for _, claim := range entityClaims {
				items = append(items, *claim)
			}
		}
	} else {
		for _, sessionClaims := range s.claims {
			for _, entityClaims := range sessionClaims {
				for _, claim := range entityClaims {
					items = append(items, *claim)
				}
			}
		}
	}
	slices.SortFunc(items, func(left, right AgentClaim) int {
		if left.SessionID != right.SessionID {
			return strings.Compare(left.SessionID, right.SessionID)
		}
		if left.Entity != right.Entity {
			return strings.Compare(left.Entity, right.Entity)
		}
		return strings.Compare(left.AgentID, right.AgentID)
	})
	return items
}

func (s *MemoryCollaborationStore) AppendEvent(event CollaborationEvent) (CollaborationEvent, error) {
	now := time.Now().UTC()
	event, err := normalizeCollaborationEvent(event, now)
	if err != nil {
		return CollaborationEvent{}, err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked(now)
	return s.appendEventLocked(event), nil
}

func (s *MemoryCollaborationStore) ListEvents(sessionID string, after uint64, limit int) []CollaborationEvent {
	sessionID = strings.TrimSpace(sessionID)
	if sessionID == "" {
		return nil
	}
	if limit <= 0 {
		limit = defaultEventLimit
	}
	if limit > maxSessionEvents {
		limit = maxSessionEvents
	}
	now := time.Now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked(now)
	source := s.events[sessionID]
	items := make([]CollaborationEvent, 0, minInt(limit, len(source)))
	for _, event := range source {
		if event.Seq <= after {
			continue
		}
		items = append(items, event)
		if len(items) >= limit {
			break
		}
	}
	return items
}

func (s *MemoryCollaborationStore) AgentCount() int {
	now := time.Now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked(now)
	var count int
	for _, sessionAgents := range s.agents {
		count += len(sessionAgents)
	}
	return count
}

func (s *MemoryCollaborationStore) ClaimCount() int {
	now := time.Now().UTC()
	s.mu.Lock()
	defer s.mu.Unlock()
	s.pruneLocked(now)
	var count int
	for _, sessionClaims := range s.claims {
		for _, entityClaims := range sessionClaims {
			count += len(entityClaims)
		}
	}
	return count
}

func (s *MemoryCollaborationStore) releaseClaimsLocked(sessionID, agentID string) {
	sessionClaims := s.claims[sessionID]
	for entity, entityClaims := range sessionClaims {
		if entityClaims[agentID] == nil {
			continue
		}
		delete(entityClaims, agentID)
		s.appendEventLocked(CollaborationEvent{
			SessionID: sessionID,
			Type:      "claim.release",
			AgentID:   agentID,
			Entity:    entity,
			Summary:   "agent_removed",
		})
		if len(entityClaims) == 0 {
			delete(sessionClaims, entity)
		}
	}
	if len(sessionClaims) == 0 {
		delete(s.claims, sessionID)
	}
}

func (s *MemoryCollaborationStore) appendEventLocked(event CollaborationEvent) CollaborationEvent {
	event.CreatedAt = defaultTime(event.CreatedAt, time.Now().UTC())
	event.SessionID = strings.TrimSpace(event.SessionID)
	if event.SessionID == "" {
		return event
	}
	s.nextSeq[event.SessionID]++
	event.Seq = s.nextSeq[event.SessionID]
	s.events[event.SessionID] = append(s.events[event.SessionID], event)
	if len(s.events[event.SessionID]) > maxSessionEvents {
		s.events[event.SessionID] = append([]CollaborationEvent(nil), s.events[event.SessionID][len(s.events[event.SessionID])-maxSessionEvents:]...)
	}
	return event
}

func (s *MemoryCollaborationStore) pruneLocked(now time.Time) {
	for sessionID, sessionAgents := range s.agents {
		for agentID, agent := range sessionAgents {
			if now.Sub(agent.UpdatedAt) > s.idleTTL {
				delete(sessionAgents, agentID)
				s.releaseClaimsLocked(sessionID, agentID)
			}
		}
		if len(sessionAgents) == 0 {
			delete(s.agents, sessionID)
		}
	}
}

func normalizeAgentPresence(agent AgentPresence, now time.Time) (AgentPresence, error) {
	agent.SessionID = strings.TrimSpace(agent.SessionID)
	agent.AgentID = strings.TrimSpace(agent.AgentID)
	agent.ParentAgentID = strings.TrimSpace(agent.ParentAgentID)
	agent.Role = strings.TrimSpace(agent.Role)
	agent.Branch = strings.TrimSpace(agent.Branch)
	agent.CheckpointID = strings.TrimSpace(agent.CheckpointID)
	agent.Status = defaultString(agent.Status, "active")
	if agent.SessionID == "" || agent.AgentID == "" {
		return AgentPresence{}, ErrUnknownSession
	}
	agent.Capabilities = normalizeStringSlice(agent.Capabilities)
	if agent.CreatedAt.IsZero() {
		agent.CreatedAt = now
	}
	agent.UpdatedAt = now
	return agent, nil
}

func normalizeAgentClaim(claim AgentClaim, now time.Time) (AgentClaim, error) {
	claim.SessionID = strings.TrimSpace(claim.SessionID)
	claim.Entity = strings.TrimSpace(claim.Entity)
	claim.AgentID = strings.TrimSpace(claim.AgentID)
	claim.Mode = defaultString(strings.TrimSpace(claim.Mode), "exclusive")
	claim.Scope = defaultString(strings.TrimSpace(claim.Scope), "entity")
	if claim.SessionID == "" || claim.Entity == "" || claim.AgentID == "" {
		return AgentClaim{}, ErrUnknownSession
	}
	if claim.AcquiredAt.IsZero() {
		claim.AcquiredAt = now
	}
	claim.UpdatedAt = now
	return claim, nil
}

func normalizeCollaborationEvent(event CollaborationEvent, now time.Time) (CollaborationEvent, error) {
	event.SessionID = strings.TrimSpace(event.SessionID)
	event.Type = strings.TrimSpace(event.Type)
	event.AgentID = strings.TrimSpace(event.AgentID)
	event.Entity = strings.TrimSpace(event.Entity)
	event.Summary = strings.TrimSpace(event.Summary)
	if event.SessionID == "" || event.Type == "" {
		return CollaborationEvent{}, ErrUnknownSession
	}
	if event.CreatedAt.IsZero() {
		event.CreatedAt = now
	}
	return event, nil
}

func normalizeStringSlice(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		value = strings.TrimSpace(value)
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}

func defaultTime(value, fallback time.Time) time.Time {
	if value.IsZero() {
		return fallback
	}
	return value
}
