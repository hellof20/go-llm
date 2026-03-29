package llm

import (
	"context"
	"fmt"
	"log/slog"
)

// ProviderRegistry holds multiple LLM providers and routes requests
// to the correct provider based on the ConversationRequest.Provider field.
// It implements the Provider interface.
type ProviderRegistry struct {
	providers   map[string]Provider
	defaultName string
}

// NewProviderRegistry creates a new provider registry with a default provider.
func NewProviderRegistry(defaultName string) *ProviderRegistry {
	return &ProviderRegistry{
		providers:   make(map[string]Provider),
		defaultName: defaultName,
	}
}

// Register adds a provider to the registry.
func (r *ProviderRegistry) Register(name string, provider Provider) {
	r.providers[name] = provider
}

// Get returns a provider by name.
func (r *ProviderRegistry) Get(name string) (Provider, bool) {
	p, ok := r.providers[name]
	return p, ok
}

// SetDefault sets the default provider name.
func (r *ProviderRegistry) SetDefault(name string) {
	r.defaultName = name
}

// Default returns the default provider.
func (r *ProviderRegistry) Default() Provider {
	return r.providers[r.defaultName]
}

// resolve selects the provider to use for a request.
func (r *ProviderRegistry) resolve(ctx context.Context, providerName string) (Provider, error) {
	provider := r.providers[r.defaultName]
	if providerName != "" {
		if p, ok := r.providers[providerName]; ok {
			provider = p
		} else {
			slog.WarnContext(ctx, "requested provider not found, using default",
				"requested", providerName,
				"default", r.defaultName,
			)
		}
	}
	if provider == nil {
		return nil, fmt.Errorf("no provider available (requested=%s, default=%s)", providerName, r.defaultName)
	}
	return provider, nil
}

// Chat routes the request to the appropriate provider based on req.Provider.
func (r *ProviderRegistry) Chat(ctx context.Context, req ConversationRequest) (*LLMResponse, error) {
	provider, err := r.resolve(ctx, req.Provider)
	if err != nil {
		return nil, err
	}
	return provider.Chat(ctx, req)
}

// ChatStream routes the streaming request to the appropriate provider.
func (r *ProviderRegistry) ChatStream(ctx context.Context, req ConversationRequest) (<-chan StreamEvent, error) {
	provider, err := r.resolve(ctx, req.Provider)
	if err != nil {
		return nil, err
	}
	return provider.ChatStream(ctx, req)
}
