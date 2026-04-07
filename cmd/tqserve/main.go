package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	turboquant "github.com/odvcencio/turboquant"
	"github.com/odvcencio/turboquant/internal/tqserve"
)

type fileConfig struct {
	Listen         string                 `json:"listen"`
	APIKeys        []string               `json:"api_keys"`
	DefaultOwner   string                 `json:"default_owner"`
	SessionHeader  string                 `json:"session_header"`
	SessionIdleTTL string                 `json:"session_idle_ttl"`
	Backends       map[string]fileBackend `json:"backends"`
	Models         []fileModel            `json:"models"`
}

type fileBackend struct {
	Type             string                   `json:"type"`
	BaseURL          string                   `json:"base_url"`
	APIKey           string                   `json:"api_key"`
	ExecutorType     string                   `json:"executor_type"`
	ExecutorBackend  string                   `json:"executor_backend"`
	ExecutorBaseURL  string                   `json:"executor_base_url"`
	ExecutorAPIKey   string                   `json:"executor_api_key"`
	ExecutorModel    string                   `json:"executor_model"`
	ExecutorPrompt   string                   `json:"executor_system_prompt"`
	ModelIDs         []string                 `json:"model_ids"`
	OwnedBy          string                   `json:"owned_by"`
	Accelerator      string                   `json:"accelerator"`
	Device           string                   `json:"device"`
	DeviceCount      int                      `json:"device_count"`
	TotalMemoryBytes uint64                   `json:"total_memory_bytes"`
	WeightsBytes     uint64                   `json:"weights_bytes"`
	MaxSessions      int                      `json:"max_sessions"`
	KeyDim           int                      `json:"key_dim"`
	KeyBits          int                      `json:"key_bits"`
	ValueDim         int                      `json:"value_dim"`
	ValueBits        int                      `json:"value_bits"`
	PageCapacity     int                      `json:"page_capacity"`
	Seed             int64                    `json:"seed"`
	Command          string                   `json:"command"`
	Args             []string                 `json:"args"`
	Env              map[string]string        `json:"env"`
	HealthURL        string                   `json:"health_url"`
	StatusURL        string                   `json:"status_url"`
	CheckpointURL    string                   `json:"checkpoint_url"`
	RestoreURL       string                   `json:"restore_url"`
	StartupTimeout   string                   `json:"startup_timeout"`
	ShutdownTimeout  string                   `json:"shutdown_timeout"`
	Capacity         tqserve.CapacitySnapshot `json:"capacity"`
}

type fileModel struct {
	Name    string `json:"name"`
	Backend string `json:"backend"`
	Target  string `json:"target"`
	OwnedBy string `json:"owned_by"`
}

func main() {
	configPath := flag.String("config", os.Getenv("TQSERVE_CONFIG"), "path to tqserve JSON config")
	addr := flag.String("listen", envOr("TQSERVE_ADDR", ":8080"), "listen address")
	apiKeys := flag.String("api-keys", os.Getenv("TQSERVE_API_KEYS"), "comma-separated API keys required by tqserve")
	backendType := flag.String("backend-type", envOr("TQSERVE_BACKEND_TYPE", "upstream"), "backend type: upstream or ollama")
	upstreamBase := flag.String("upstream-base-url", os.Getenv("TQSERVE_UPSTREAM_BASE_URL"), "OpenAI-compatible upstream base URL, for example http://127.0.0.1:8081/v1")
	upstreamKey := flag.String("upstream-api-key", os.Getenv("TQSERVE_UPSTREAM_API_KEY"), "optional upstream bearer API key")
	ollamaBase := flag.String("ollama-base-url", envOr("TQSERVE_OLLAMA_BASE_URL", "http://127.0.0.1:11434"), "Ollama base URL")
	ollamaKey := flag.String("ollama-api-key", os.Getenv("TQSERVE_OLLAMA_API_KEY"), "optional Ollama bearer API key")
	models := flag.String("models", os.Getenv("TQSERVE_MODELS"), "comma-separated public=backend model mappings")
	owner := flag.String("owner", envOr("TQSERVE_MODEL_OWNER", "turboquant"), "model owner label returned by /v1/models")
	sessionHeader := flag.String("session-header", envOr("TQSERVE_SESSION_HEADER", tqserve.DefaultSessionHeader), "session header used for server-side runtime state")
	sessionIdleTTL := flag.String("session-idle-ttl", envOr("TQSERVE_SESSION_IDLE_TTL", tqserve.DefaultSessionIdleTTL.String()), "idle TTL before tqserve expires tracked sessions")
	showVersion := flag.Bool("version", false, "print tqserve version and exit")
	flag.Parse()

	if *showVersion {
		fmt.Println(turboquant.Version)
		return
	}

	serverCfg, listenAddr, closers, err := buildServerConfig(options{
		ConfigPath:     strings.TrimSpace(*configPath),
		Listen:         *addr,
		APIKeys:        parseCSV(*apiKeys),
		BackendType:    *backendType,
		UpstreamBase:   *upstreamBase,
		UpstreamKey:    *upstreamKey,
		OllamaBase:     *ollamaBase,
		OllamaKey:      *ollamaKey,
		Models:         parseModelMap(*models),
		DefaultOwner:   *owner,
		SessionHeader:  *sessionHeader,
		SessionIdleTTL: *sessionIdleTTL,
	})
	if err != nil {
		log.Fatal(err)
	}
	defer closeAll(closers)

	server, err := tqserve.New(serverCfg)
	if err != nil {
		log.Fatal(err)
	}
	httpServer := &http.Server{
		Addr:    listenAddr,
		Handler: server.Handler(),
	}
	log.Printf("tqserve listening on %s", listenAddr)
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()
	go func() {
		<-ctx.Done()
		shutdownCtx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		defer cancel()
		_ = httpServer.Shutdown(shutdownCtx)
	}()
	if err := httpServer.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
}

type options struct {
	ConfigPath     string
	Listen         string
	APIKeys        []string
	BackendType    string
	UpstreamBase   string
	UpstreamKey    string
	OllamaBase     string
	OllamaKey      string
	Models         map[string]string
	DefaultOwner   string
	SessionHeader  string
	SessionIdleTTL string
}

func buildServerConfig(opts options) (tqserve.Config, string, []io.Closer, error) {
	if opts.ConfigPath != "" {
		return buildServerConfigFromFile(opts.ConfigPath)
	}
	backend, closer, err := buildBackend(fileBackend{
		Type:    opts.BackendType,
		BaseURL: chooseBaseURL(opts.BackendType, opts.UpstreamBase, opts.OllamaBase),
		APIKey:  chooseAPIKey(opts.BackendType, opts.UpstreamKey, opts.OllamaKey),
	})
	if err != nil {
		return tqserve.Config{}, "", nil, err
	}
	sessionIdleTTL, err := parseDuration(opts.SessionIdleTTL)
	if err != nil {
		return tqserve.Config{}, "", nil, err
	}
	return tqserve.Config{
		APIKeys:        opts.APIKeys,
		Backend:        backend,
		ModelMap:       opts.Models,
		DefaultOwn:     opts.DefaultOwner,
		SessionHeader:  opts.SessionHeader,
		SessionIdleTTL: sessionIdleTTL,
	}, defaultString(opts.Listen, ":8080"), closersOf(closer), nil
}

func buildServerConfigFromFile(path string) (tqserve.Config, string, []io.Closer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return tqserve.Config{}, "", nil, err
	}
	var cfg fileConfig
	if err := json.Unmarshal(data, &cfg); err != nil {
		return tqserve.Config{}, "", nil, fmt.Errorf("tqserve: decode config: %w", err)
	}
	sessionIdleTTL, err := parseDuration(cfg.SessionIdleTTL)
	if err != nil {
		return tqserve.Config{}, "", nil, err
	}
	backends, closers, err := buildConfiguredBackends(cfg.Backends)
	if err != nil {
		return tqserve.Config{}, "", nil, err
	}
	routes := make([]tqserve.ModelRoute, 0, len(cfg.Models))
	for _, model := range cfg.Models {
		routes = append(routes, tqserve.ModelRoute{
			PublicModel:  model.Name,
			BackendName:  model.Backend,
			BackendModel: defaultString(model.Target, model.Name),
			OwnedBy:      model.OwnedBy,
		})
	}
	return tqserve.Config{
		APIKeys:        cfg.APIKeys,
		Backends:       backends,
		Routes:         routes,
		DefaultOwn:     defaultString(cfg.DefaultOwner, "turboquant"),
		SessionHeader:  cfg.SessionHeader,
		SessionIdleTTL: sessionIdleTTL,
	}, defaultString(cfg.Listen, ":8080"), closers, nil
}

func buildBackend(spec fileBackend) (tqserve.Backend, io.Closer, error) {
	return buildBackendWithExecutorResolver(spec, buildNativeExecutor)
}

func buildBackendWithExecutorResolver(spec fileBackend, resolveExecutor func(fileBackend) (tqserve.NativeExecutor, error)) (tqserve.Backend, io.Closer, error) {
	switch strings.ToLower(strings.TrimSpace(spec.Type)) {
	case "", "openai", "upstream":
		backend, err := tqserve.NewUpstreamBackend(spec.BaseURL, spec.APIKey, nil)
		return backend, nil, err
	case "ollama":
		backend, err := tqserve.NewOllamaBackend(spec.BaseURL, spec.APIKey, nil)
		return backend, nil, err
	case "native", "turboquant":
		var executor tqserve.NativeExecutor
		var err error
		if resolveExecutor != nil {
			executor, err = resolveExecutor(spec)
		}
		if err != nil {
			return nil, nil, err
		}
		backend, err := tqserve.NewNativeBackend(tqserve.NativeBackendConfig{
			Name:             spec.Type,
			ModelIDs:         spec.ModelIDs,
			OwnedBy:          spec.OwnedBy,
			Accelerator:      spec.Accelerator,
			Device:           spec.Device,
			DeviceCount:      spec.DeviceCount,
			TotalMemoryBytes: spec.TotalMemoryBytes,
			WeightsBytes:     spec.WeightsBytes,
			MaxSessions:      spec.MaxSessions,
			KeyDim:           spec.KeyDim,
			KeyBits:          spec.KeyBits,
			ValueDim:         spec.ValueDim,
			ValueBits:        spec.ValueBits,
			PageCapacity:     spec.PageCapacity,
			Seed:             spec.Seed,
			Executor:         executor,
		})
		return backend, nil, err
	case "managed_upstream", "managed-openai", "managed_openai", "managed_ollama", "managed-ollama":
		startupTimeout, err := parseDuration(spec.StartupTimeout)
		if err != nil {
			return nil, nil, err
		}
		shutdownTimeout, err := parseDuration(spec.ShutdownTimeout)
		if err != nil {
			return nil, nil, err
		}
		backend, err := tqserve.NewManagedBackend(tqserve.ManagedBackendConfig{
			Name:            spec.Type,
			Kind:            spec.Type,
			BaseURL:         spec.BaseURL,
			APIKey:          spec.APIKey,
			Command:         spec.Command,
			Args:            spec.Args,
			Env:             spec.Env,
			HealthURL:       spec.HealthURL,
			StatusURL:       spec.StatusURL,
			CheckpointURL:   spec.CheckpointURL,
			RestoreURL:      spec.RestoreURL,
			StartupTimeout:  startupTimeout,
			ShutdownTimeout: shutdownTimeout,
			Capacity:        spec.Capacity,
		})
		if err != nil {
			return nil, nil, err
		}
		return backend, backend, nil
	default:
		return nil, nil, fmt.Errorf("tqserve: unsupported backend type %q", spec.Type)
	}
}

func buildConfiguredBackends(specs map[string]fileBackend) (map[string]tqserve.Backend, []io.Closer, error) {
	built := make(map[string]tqserve.Backend, len(specs))
	building := make(map[string]bool, len(specs))
	var closers []io.Closer
	buildOne := func(name string) (tqserve.Backend, error) {
		return nil, nil
	}
	buildOne = func(name string) (tqserve.Backend, error) {
		if backend, ok := built[name]; ok {
			return backend, nil
		}
		spec, ok := specs[name]
		if !ok {
			return nil, fmt.Errorf("tqserve: unknown backend %q", name)
		}
		if building[name] {
			return nil, fmt.Errorf("tqserve: cyclic backend dependency involving %q", name)
		}
		building[name] = true
		backend, closer, err := buildBackendWithExecutorResolver(spec, func(spec fileBackend) (tqserve.NativeExecutor, error) {
			return buildNativeExecutorWithLookup(spec, buildOne)
		})
		delete(building, name)
		if err != nil {
			return nil, fmt.Errorf("tqserve: backend %q: %w", name, err)
		}
		built[name] = backend
		closers = append(closers, closersOf(closer)...)
		return backend, nil
	}
	for name := range specs {
		if _, err := buildOne(name); err != nil {
			return nil, nil, err
		}
	}
	return built, closers, nil
}

func buildNativeExecutor(spec fileBackend) (tqserve.NativeExecutor, error) {
	switch strings.ToLower(strings.TrimSpace(spec.ExecutorType)) {
	case "":
		return nil, nil
	case "upstream", "openai":
		return tqserve.NewUpstreamNativeExecutor(spec.ExecutorBaseURL, spec.ExecutorAPIKey, spec.ExecutorModel, spec.ExecutorPrompt, nil)
	case "ollama":
		return tqserve.NewOllamaNativeExecutor(spec.ExecutorBaseURL, spec.ExecutorAPIKey, spec.ExecutorModel, spec.ExecutorPrompt, nil)
	default:
		return nil, fmt.Errorf("tqserve: unsupported native executor type %q", spec.ExecutorType)
	}
}

func buildNativeExecutorWithLookup(spec fileBackend, lookup func(string) (tqserve.Backend, error)) (tqserve.NativeExecutor, error) {
	if backendName := strings.TrimSpace(spec.ExecutorBackend); backendName != "" {
		backend, err := lookup(backendName)
		if err != nil {
			return nil, err
		}
		if _, ok := backend.(*tqserve.NativeBackend); ok {
			return nil, fmt.Errorf("tqserve: native executor backend %q cannot itself be native", backendName)
		}
		return tqserve.NewBackendNativeExecutor(backend, spec.ExecutorModel, spec.ExecutorPrompt)
	}
	return buildNativeExecutor(spec)
}

func parseDuration(raw string) (time.Duration, error) {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return 0, nil
	}
	d, err := time.ParseDuration(raw)
	if err != nil {
		return 0, fmt.Errorf("tqserve: invalid duration %q: %w", raw, err)
	}
	return d, nil
}

func chooseBaseURL(kind, upstreamBase, ollamaBase string) string {
	if strings.EqualFold(strings.TrimSpace(kind), "ollama") {
		return ollamaBase
	}
	return upstreamBase
}

func chooseAPIKey(kind, upstreamKey, ollamaKey string) string {
	if strings.EqualFold(strings.TrimSpace(kind), "ollama") {
		return ollamaKey
	}
	return upstreamKey
}

func closersOf(closer io.Closer) []io.Closer {
	if closer == nil {
		return nil
	}
	return []io.Closer{closer}
}

func closeAll(closers []io.Closer) {
	for i := len(closers) - 1; i >= 0; i-- {
		_ = closers[i].Close()
	}
}

func parseCSV(raw string) []string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	values := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part != "" {
			values = append(values, part)
		}
	}
	return values
}

func parseModelMap(raw string) map[string]string {
	if strings.TrimSpace(raw) == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	mapping := make(map[string]string, len(parts))
	for _, part := range parts {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		left, right, ok := strings.Cut(part, "=")
		if !ok {
			mapping[part] = part
			continue
		}
		left = strings.TrimSpace(left)
		right = strings.TrimSpace(right)
		if left == "" || right == "" {
			log.Fatalf("invalid model mapping %q", part)
		}
		mapping[left] = right
	}
	return mapping
}

func envOr(key, fallback string) string {
	if value := strings.TrimSpace(os.Getenv(key)); value != "" {
		return value
	}
	return fallback
}

func defaultString(value, fallback string) string {
	if strings.TrimSpace(value) == "" {
		return fallback
	}
	return value
}
