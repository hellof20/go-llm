package llm

import (
	"fmt"
	"sync"
)

// Provider name constants.
const (
	ProviderGemini = "gemini"
	ProviderQwen   = "qwen"
	ProviderClaude = "claude"
)

// Config holds configuration for a single LLM provider.
type Config struct {
	Project    string
	Location   string
	APIKey     string
	BaseURL    string
	RetryTimes int
}

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
