package llm

import "context"

// Provider defines the interface for LLM providers.
type Provider interface {
	Chat(ctx context.Context, req ConversationRequest) (*LLMResponse, error)
	ChatStream(ctx context.Context, req ConversationRequest) (<-chan StreamEvent, error)
}
