package tqserve

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"strings"
	"sync"
	"syscall"
	"time"
)

type ManagedBackend struct {
	Backend

	name            string
	kind            string
	baseURL         string
	capacity        CapacitySnapshot
	apiKey          string
	cmd             *exec.Cmd
	healthURL       string
	statusURL       string
	checkpointURL   string
	restoreURL      string
	startupTimeout  time.Duration
	shutdownTimeout time.Duration

	mu         sync.Mutex
	waitCh     chan error
	closed     bool
	exited     bool
	processErr string
}

type ManagedBackendConfig struct {
	Name            string
	Kind            string
	BaseURL         string
	APIKey          string
	Command         string
	Args            []string
	Env             map[string]string
	HealthURL       string
	StatusURL       string
	CheckpointURL   string
	RestoreURL      string
	StartupTimeout  time.Duration
	ShutdownTimeout time.Duration
	Capacity        CapacitySnapshot
	Client          *http.Client
}

func NewManagedBackend(cfg ManagedBackendConfig) (*ManagedBackend, error) {
	if strings.TrimSpace(cfg.Command) == "" {
		return nil, fmt.Errorf("tqserve: managed backend command is required")
	}
	kind := managedBaseKind(cfg.Kind)
	if kind == "" {
		return nil, fmt.Errorf("tqserve: unsupported managed backend type %q", cfg.Kind)
	}
	baseURL := strings.TrimRight(strings.TrimSpace(cfg.BaseURL), "/")
	var (
		backend Backend
		err     error
	)
	switch kind {
	case "upstream":
		backend, err = NewUpstreamBackend(baseURL, cfg.APIKey, cfg.Client)
	case "ollama":
		backend, err = NewOllamaBackend(baseURL, cfg.APIKey, cfg.Client)
	}
	if err != nil {
		return nil, err
	}
	cmd := exec.Command(cfg.Command, cfg.Args...)
	cmd.Stdout = os.Stdout
	cmd.Stderr = os.Stderr
	cmd.Env = os.Environ()
	for key, value := range cfg.Env {
		cmd.Env = append(cmd.Env, key+"="+value)
	}
	managed := &ManagedBackend{
		Backend:         backend,
		name:            defaultString(cfg.Name, kind),
		kind:            kind,
		baseURL:         baseURL,
		capacity:        cfg.Capacity,
		apiKey:          strings.TrimSpace(cfg.APIKey),
		cmd:             cmd,
		healthURL:       defaultString(cfg.HealthURL, managedDefaultHealthURL(kind, baseURL)),
		statusURL:       strings.TrimSpace(cfg.StatusURL),
		checkpointURL:   strings.TrimSpace(cfg.CheckpointURL),
		restoreURL:      strings.TrimSpace(cfg.RestoreURL),
		startupTimeout:  defaultDuration(cfg.StartupTimeout, 60*time.Second),
		shutdownTimeout: defaultDuration(cfg.ShutdownTimeout, 10*time.Second),
		waitCh:          make(chan error, 1),
	}
	if err := managed.start(); err != nil {
		return nil, err
	}
	return managed, nil
}

func (m *ManagedBackend) start() error {
	if err := m.cmd.Start(); err != nil {
		return fmt.Errorf("tqserve: start managed backend %q: %w", m.name, err)
	}
	go func() {
		err := m.cmd.Wait()
		m.mu.Lock()
		m.exited = true
		if err != nil {
			m.processErr = err.Error()
		}
		m.mu.Unlock()
		m.waitCh <- err
		close(m.waitCh)
	}()
	client := &http.Client{Timeout: 2 * time.Second}
	deadline := time.NewTimer(m.startupTimeout)
	defer deadline.Stop()
	ticker := time.NewTicker(200 * time.Millisecond)
	defer ticker.Stop()
	for {
		select {
		case err := <-m.waitCh:
			if err == nil {
				return fmt.Errorf("tqserve: managed backend %q exited before becoming ready", m.name)
			}
			return fmt.Errorf("tqserve: managed backend %q exited before becoming ready: %w", m.name, err)
		case <-ticker.C:
			if ready(client, m.healthURL) {
				return nil
			}
		case <-deadline.C:
			_ = m.Close()
			return fmt.Errorf("tqserve: managed backend %q did not become ready at %s within %s", m.name, m.healthURL, m.startupTimeout)
		}
	}
}

func (m *ManagedBackend) Status(ctx context.Context) BackendStatus {
	m.mu.Lock()
	exited := m.exited
	closed := m.closed
	lastErr := m.processErr
	pid := 0
	if m.cmd != nil && m.cmd.Process != nil {
		pid = m.cmd.Process.Pid
	}
	m.mu.Unlock()

	status := BackendStatus{
		Name:           m.name,
		Kind:           "managed_" + m.kind,
		BaseURL:        m.baseURL,
		HealthURL:      m.healthURL,
		Managed:        true,
		ProcessRunning: !closed && !exited,
		PID:            pid,
		LastError:      lastErr,
	}
	if !m.capacity.Empty() {
		capacity := m.capacity
		status.Capacity = &capacity
	}
	if !status.ProcessRunning {
		return status
	}
	if remote, err := m.fetchRemoteStatus(ctx); err == nil && remote != nil {
		status.Ready = remote.OK
		if len(remote.Backends) > 0 {
			inner := remote.Backends[0]
			status.Ready = status.Ready && inner.Ready
			if status.LastError == "" {
				status.LastError = inner.LastError
			}
			if inner.Capacity != nil {
				merged := mergeCapacity(*inner.Capacity, m.capacity)
				status.Capacity = &merged
			}
		}
		return status
	}
	if provider, ok := m.Backend.(StatusProvider); ok {
		inner := provider.Status(ctx)
		status.Ready = inner.Ready
		if status.LastError == "" {
			status.LastError = inner.LastError
		}
		if inner.Capacity != nil {
			merged := mergeCapacity(*inner.Capacity, m.capacity)
			status.Capacity = &merged
		}
		return status
	}
	status.Ready = ready(&http.Client{Timeout: 2 * time.Second}, m.healthURL)
	return status
}

func (m *ManagedBackend) Capacity(ctx context.Context) CapacitySnapshot {
	if provider, ok := m.Backend.(CapacityProvider); ok {
		return mergeCapacity(provider.Capacity(ctx), m.capacity)
	}
	return m.capacity
}

func (m *ManagedBackend) CaptureSessionState(ctx Context, sessionID string) (json.RawMessage, error) {
	if m.checkpointURL == "" {
		return nil, fmt.Errorf("tqserve: managed backend %q does not expose checkpoint capture", m.name)
	}
	payload, err := m.postJSON(ctx, m.checkpointURL, map[string]any{"session_id": sessionID})
	if err != nil {
		return nil, err
	}
	var checkpoint SessionCheckpoint
	if err := json.Unmarshal(payload, &checkpoint); err != nil {
		return nil, err
	}
	return checkpoint.State, nil
}

func (m *ManagedBackend) RestoreSessionState(ctx Context, sessionID string, state json.RawMessage) error {
	if m.restoreURL == "" {
		return fmt.Errorf("tqserve: managed backend %q does not expose checkpoint restore", m.name)
	}
	_, err := m.postJSON(ctx, m.restoreURL, map[string]any{
		"checkpoint": SessionCheckpoint{
			Version:    "tqserve.session.v1",
			CapturedAt: time.Now().UTC(),
			Session: SessionInfo{
				ID:      sessionID,
				Backend: "default",
			},
			State: state,
		},
	})
	return err
}

func (m *ManagedBackend) Close() error {
	if m == nil {
		return nil
	}
	m.mu.Lock()
	if m.closed {
		m.mu.Unlock()
		return nil
	}
	m.closed = true
	cmd := m.cmd
	waitCh := m.waitCh
	timeout := m.shutdownTimeout
	m.mu.Unlock()

	if cmd == nil || cmd.Process == nil {
		return nil
	}
	select {
	case <-waitCh:
		return nil
	default:
	}
	_ = cmd.Process.Signal(os.Interrupt)
	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-waitCh:
		return nil
	case <-timer.C:
		_ = cmd.Process.Signal(syscall.SIGKILL)
		<-waitCh
		return nil
	}
}

func managedBaseKind(kind string) string {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "managed_upstream", "managed-openai", "managed_openai":
		return "upstream"
	case "managed_ollama", "managed-ollama":
		return "ollama"
	default:
		return ""
	}
}

func managedDefaultHealthURL(kind, baseURL string) string {
	switch kind {
	case "ollama":
		return baseURL + "/api/tags"
	default:
		return baseURL + "/models"
	}
}

func (m *ManagedBackend) fetchRemoteStatus(ctx context.Context) (*ServerStatus, error) {
	if m.statusURL == "" {
		return nil, nil
	}
	payload, err := m.getJSON(ctx, m.statusURL)
	if err != nil {
		return nil, err
	}
	var status ServerStatus
	if err := json.Unmarshal(payload, &status); err != nil {
		return nil, err
	}
	return &status, nil
}

func (m *ManagedBackend) getJSON(ctx context.Context, url string) ([]byte, error) {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, url, nil)
	if err != nil {
		return nil, err
	}
	m.setAuth(req)
	client := &http.Client{Timeout: 2 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	payload, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("tqserve: managed backend %q status request failed with status %d", m.name, resp.StatusCode)
	}
	return payload, nil
}

func (m *ManagedBackend) postJSON(ctx context.Context, url string, body any) ([]byte, error) {
	payload, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, strings.NewReader(string(payload)))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	m.setAuth(req)
	client := &http.Client{Timeout: 5 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	out, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return nil, fmt.Errorf("tqserve: managed backend %q request failed with status %d", m.name, resp.StatusCode)
	}
	return out, nil
}

func (m *ManagedBackend) setAuth(req *http.Request) {
	if m.apiKey != "" {
		req.Header.Set("Authorization", "Bearer "+m.apiKey)
	}
}

func ready(client *http.Client, url string) bool {
	req, err := http.NewRequestWithContext(context.Background(), http.MethodGet, url, nil)
	if err != nil {
		return false
	}
	resp, err := client.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	_, _ = io.Copy(io.Discard, resp.Body)
	return resp.StatusCode >= 200 && resp.StatusCode < 300
}

func defaultDuration(value, fallback time.Duration) time.Duration {
	if value > 0 {
		return value
	}
	return fallback
}
