package tqserve

import (
	"context"
	"fmt"
	"slices"
	"strings"
	"time"

	turboquant "github.com/odvcencio/turboquant"
)

type BackendStatus struct {
	Name           string            `json:"name"`
	Kind           string            `json:"kind"`
	BaseURL        string            `json:"base_url,omitempty"`
	HealthURL      string            `json:"health_url,omitempty"`
	Ready          bool              `json:"ready"`
	Managed        bool              `json:"managed,omitempty"`
	ProcessRunning bool              `json:"process_running,omitempty"`
	PID            int               `json:"pid,omitempty"`
	LastError      string            `json:"last_error,omitempty"`
	Capacity       *CapacitySnapshot `json:"capacity,omitempty"`
}

type StatusProvider interface {
	Status(ctx context.Context) BackendStatus
}

type ServerStatus struct {
	OK             bool            `json:"ok"`
	Service        string          `json:"service"`
	Version        string          `json:"version"`
	DefaultOwner   string          `json:"default_owner"`
	SessionHeader  string          `json:"session_header"`
	SessionIdleTTL string          `json:"session_idle_ttl"`
	ActiveSessions int             `json:"active_sessions"`
	ActiveAgents   int             `json:"active_agents"`
	ActiveClaims   int             `json:"active_claims"`
	BackendCount   int             `json:"backend_count"`
	RouteCount     int             `json:"route_count"`
	Metrics        MetricsSnapshot `json:"metrics"`
	Backends       []BackendStatus `json:"backends"`
	Routes         []ModelRoute    `json:"routes,omitempty"`
}

func (s *Server) status(ctx context.Context) ServerStatus {
	names := make([]string, 0, len(s.backends))
	for name := range s.backends {
		names = append(names, name)
	}
	slices.Sort(names)
	backends := make([]BackendStatus, 0, len(names))
	for _, name := range names {
		backends = append(backends, backendStatus(ctx, name, s.backends[name]))
	}
	routes := make([]ModelRoute, 0, len(s.routes))
	for _, name := range routeNames(s.routes) {
		routes = append(routes, s.routes[name])
	}
	return ServerStatus{
		OK:             true,
		Service:        "tqserve",
		Version:        turboquant.Version,
		DefaultOwner:   s.defaultOwn,
		SessionHeader:  s.sessionHeader,
		SessionIdleTTL: s.sessionIdleTTL.String(),
		ActiveSessions: s.sessions.Count(),
		ActiveAgents:   s.collaboration.AgentCount(),
		ActiveClaims:   s.collaboration.ClaimCount(),
		BackendCount:   len(s.backends),
		RouteCount:     len(s.routes),
		Metrics:        s.metrics.Snapshot(),
		Backends:       backends,
		Routes:         routes,
	}
}

func backendStatus(ctx context.Context, name string, backend Backend) BackendStatus {
	var capacity CapacitySnapshot
	if provider, ok := backend.(CapacityProvider); ok {
		capacity = provider.Capacity(ctx)
	}
	if provider, ok := backend.(StatusProvider); ok {
		status := provider.Status(ctx)
		status.Name = defaultString(status.Name, name)
		status.Kind = defaultString(status.Kind, fmt.Sprintf("%T", backend))
		if status.Capacity == nil && !capacity.Empty() {
			copied := capacity
			status.Capacity = &copied
		}
		return status
	}
	status := BackendStatus{
		Name:  name,
		Kind:  strings.TrimPrefix(fmt.Sprintf("%T", backend), "*"),
		Ready: true,
	}
	if !capacity.Empty() {
		copied := capacity
		status.Capacity = &copied
	}
	return status
}

func routeNames(routes map[string]ModelRoute) []string {
	names := make([]string, 0, len(routes))
	for name := range routes {
		names = append(names, name)
	}
	slices.Sort(names)
	return names
}

func statusContext(ctx context.Context) (context.Context, context.CancelFunc) {
	return context.WithTimeout(ctx, 2*time.Second)
}
