package tqserve

import (
	"fmt"
	"io"
	"slices"
	"strconv"
	"strings"
	"sync"
)

type MetricsSnapshot struct {
	RequestsTotal      uint64 `json:"requests_total"`
	AuthFailuresTotal  uint64 `json:"auth_failures_total"`
	InvalidRequests    uint64 `json:"invalid_requests_total"`
	BackendErrorsTotal uint64 `json:"backend_errors_total"`
}

type requestMetricKey struct {
	endpoint string
	backend  string
	status   int
}

type metricsState struct {
	mu                sync.Mutex
	requestsByOutcome map[requestMetricKey]uint64
	requestsTotal     uint64
	authFailuresTotal uint64
	invalidRequests   uint64
	backendErrors     uint64
}

func newMetricsState() *metricsState {
	return &metricsState{
		requestsByOutcome: make(map[requestMetricKey]uint64),
	}
}

func (m *metricsState) Record(endpoint, backend string, status int) {
	m.mu.Lock()
	defer m.mu.Unlock()
	key := requestMetricKey{
		endpoint: defaultString(endpoint, "unknown"),
		backend:  defaultString(backend, "server"),
		status:   status,
	}
	m.requestsByOutcome[key]++
	m.requestsTotal++
	switch {
	case status == 401:
		m.authFailuresTotal++
	case status >= 400 && status < 500:
		m.invalidRequests++
	case status >= 500:
		m.backendErrors++
	}
}

func (m *metricsState) Snapshot() MetricsSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	return MetricsSnapshot{
		RequestsTotal:      m.requestsTotal,
		AuthFailuresTotal:  m.authFailuresTotal,
		InvalidRequests:    m.invalidRequests,
		BackendErrorsTotal: m.backendErrors,
	}
}

func (m *metricsState) WritePrometheus(w io.Writer, activeSessions, activeAgents, activeClaims, backendCount, routeCount int) {
	m.mu.Lock()
	defer m.mu.Unlock()

	_, _ = io.WriteString(w, "# HELP tqserve_requests_total Total HTTP requests handled by tqserve.\n")
	_, _ = io.WriteString(w, "# TYPE tqserve_requests_total counter\n")
	keys := make([]requestMetricKey, 0, len(m.requestsByOutcome))
	for key := range m.requestsByOutcome {
		keys = append(keys, key)
	}
	slices.SortFunc(keys, func(left, right requestMetricKey) int {
		if left.endpoint != right.endpoint {
			return strings.Compare(left.endpoint, right.endpoint)
		}
		if left.backend != right.backend {
			return strings.Compare(left.backend, right.backend)
		}
		return left.status - right.status
	})
	for _, key := range keys {
		_, _ = fmt.Fprintf(
			w,
			"tqserve_requests_total{endpoint=%s,backend=%s,status_code=%q} %d\n",
			promQuote(key.endpoint),
			promQuote(key.backend),
			strconv.Itoa(key.status),
			m.requestsByOutcome[key],
		)
	}

	writeCounter(w, "tqserve_requests_seen_total", int(m.requestsTotal))
	writeCounter(w, "tqserve_auth_failures_total", int(m.authFailuresTotal))
	writeCounter(w, "tqserve_invalid_requests_total", int(m.invalidRequests))
	writeCounter(w, "tqserve_backend_errors_total", int(m.backendErrors))
	writeGauge(w, "tqserve_active_sessions", activeSessions)
	writeGauge(w, "tqserve_active_agents", activeAgents)
	writeGauge(w, "tqserve_active_claims", activeClaims)
	writeGauge(w, "tqserve_backends_configured", backendCount)
	writeGauge(w, "tqserve_routes_configured", routeCount)
}

func writeCounter(w io.Writer, name string, value int) {
	_, _ = fmt.Fprintf(w, "# TYPE %s counter\n%s %d\n", name, name, value)
}

func writeGauge(w io.Writer, name string, value int) {
	_, _ = fmt.Fprintf(w, "# TYPE %s gauge\n%s %d\n", name, name, value)
}

func promQuote(value string) string {
	return strconv.Quote(value)
}
