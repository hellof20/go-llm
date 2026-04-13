package llm

import (
	"context"
	"fmt"
	"log/slog"
	"sync"

)

// Provider name constants.
const (
	ProviderGemini       = "gemini"
	ProviderGeminiVertex = "gemini-vertex"
	ProviderQwen         = "qwen"
	ProviderClaude       = "claude"
	ProviderClaudeVertex = "claude-vertex"
	ProviderKimiBailian  = "kimi-bailian"
)

// Config holds configuration for a single LLM provider.
type Config struct {
	Project    string
	Location   string
	APIKey     string
	BaseURL    string
	RetryTimes int
}

// discardHandler is a slog.Handler that discards all log records.
type discardHandler struct{}

func (discardHandler) Enabled(context.Context, slog.Level) bool  { return false }
func (discardHandler) Handle(context.Context, slog.Record) error { return nil }
func (d discardHandler) WithAttrs([]slog.Attr) slog.Handler      { return d }
func (d discardHandler) WithGroup(string) slog.Handler            { return d }

// ProviderFactory creates a Provider from config.
type ProviderFactory func(cfg Config) (Provider, error)

var (
	factoryMu  sync.RWMutex
	factories  = make(map[string]ProviderFactory)
)

// RegisterFactory registers a named provider factory.
// It is safe for concurrent use.
func RegisterFactory(name string, factory ProviderFactory) {
	factoryMu.Lock()
	defer factoryMu.Unlock()
	factories[name] = factory
}

// NewProvider creates a Provider using a registered factory.
func NewProvider(name string, cfg Config) (Provider, error) {
	factoryMu.RLock()
	f, ok := factories[name]
	factoryMu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("unsupported llm provider: %s", name)
	}
	return f(cfg)
}
