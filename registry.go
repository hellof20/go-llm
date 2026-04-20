package llm

import (
	"context"
	"fmt"
	"log/slog"
)

// ProviderRegistry holds multiple LLM providers and routes requests
// to the correct provider based on the ConversationRequest.Provider field.
// It implements the Provider interface.
//
// Providers can be registered in two ways:
//   - Register: pre-created provider instance, available immediately.
//   - RegisterConfig: lazy — provider is created on first use via the factory.
type ProviderRegistry struct {
	providers   map[string]Provider
	configs     map[string]Config
	defaultName string
}

// NewProviderRegistry creates a new provider registry with a default provider.
func NewProviderRegistry(defaultName string) *ProviderRegistry {
	return &ProviderRegistry{
		providers:   make(map[string]Provider),
		configs:     make(map[string]Config),
		defaultName: defaultName,
	}
}


// Register adds a pre-created provider to the registry.
func (r *ProviderRegistry) Register(name string, provider Provider) {
	r.providers[name] = provider
}

// RegisterConfig registers a provider config for lazy initialization.
// The provider will be created via the registered factory on first use.
func (r *ProviderRegistry) RegisterConfig(name string, cfg Config) {
	r.configs[name] = cfg
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
// If the provider is not yet instantiated but has a registered config,
// it will be created lazily via the factory.
func (r *ProviderRegistry) resolve(ctx context.Context, providerName string) (Provider, error) {
	name := providerName
	if name == "" {
		name = r.defaultName
	}

	provider, ok := r.providers[name]
	if !ok {
		// Try lazy creation from config
		if cfg, hasCfg := r.configs[name]; hasCfg {
			p, err := NewProvider(name, cfg)
			if err != nil {
				return nil, fmt.Errorf("lazy init provider %s: %w", name, err)
			}
			r.providers[name] = p
			provider = p
		} else if name != r.defaultName {
			slog.WarnContext(ctx, "requested provider not found, using default",
				"requested", name,
				"default", r.defaultName,
			)
			return r.resolve(ctx, "")
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

// GenerateImage routes the image generation request to the appropriate provider.
func (r *ProviderRegistry) GenerateImage(ctx context.Context, req ImageRequest) (*ImageResponse, error) {
	provider, err := r.resolve(ctx, req.Provider)
	if err != nil {
		return nil, err
	}
	imgProvider, ok := provider.(ImageProvider)
	if !ok {
		return nil, fmt.Errorf("provider %q does not support image generation", req.Provider)
	}
	return imgProvider.GenerateImage(ctx, req)
}
