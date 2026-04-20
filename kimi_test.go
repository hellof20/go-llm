package llm_test

import (
	"context"
	"testing"

	llm "github.com/hellof20/go-llm"
)

func TestKimiChat(t *testing.T) {
	apiKey := getEnv(t,"DASHSCOPE_API_KEY")
	provider, err := llm.NewProvider(llm.ProviderKimiBailian, llm.Config{APIKey: apiKey, RetryTimes: 3})
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	resp, err := provider.Chat(context.Background(), llm.ConversationRequest{
		Model:        "kimi-k2.5",
		SystemPrompt: "You are a helpful assistant. Answer concisely.",
		Messages: []llm.Message{
			{Role: "user", Content: "What is 1+1? Answer with just the number."},
		},
		Params: map[string]any{
			"max_tokens":     256,
			"temperature":    0.5,
			"top_p":          0.95,
			"thinking_level": "high",
		},
	})
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}

	if resp.Content == "" {
		t.Error("expected non-empty content")
	}
	if resp.TokenUsage.InputTokens == 0 {
		t.Error("expected non-zero input tokens")
	}
	if resp.TokenUsage.OutputTokens == 0 {
		t.Error("expected non-zero output tokens")
	}

	t.Logf("Content: %s", resp.Content)
	t.Logf("Tokens: input=%d output=%d thinking=%d cache_read=%d cache_write=%d finish=%s",
		resp.TokenUsage.InputTokens, resp.TokenUsage.OutputTokens, resp.TokenUsage.ThinkingTokens,
		resp.TokenUsage.CacheReadTokens, resp.TokenUsage.CacheWriteTokens, resp.FinishReason)
}

func TestKimiChatStream(t *testing.T) {
	apiKey := getEnv(t,"DASHSCOPE_API_KEY")
	provider, err := llm.NewProvider(llm.ProviderKimiBailian, llm.Config{APIKey: apiKey, RetryTimes: 3})
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	ch, err := provider.ChatStream(context.Background(), llm.ConversationRequest{
		Model: "kimi-k2.5",
		Messages: []llm.Message{
			{Role: "user", Content: "Count from 1 to 5."},
		},
		Params: map[string]any{
			"max_tokens":     256,
			"temperature":    0.5,
			"top_p":          0.95,
			"thinking_level": "high",
		},
	})
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}

	var content string
	var gotDone bool
	var usage *llm.TokenUsage
	for event := range ch {
		switch event.Type {
		case llm.EventTextDelta:
			content += event.Content
		case llm.EventDone:
			gotDone = true
			usage = event.Usage
		case llm.EventError:
			t.Fatalf("stream error: %v", event.Error)
		}
	}

	if content == "" {
		t.Error("expected non-empty streamed content")
	}
	if !gotDone {
		t.Error("expected Done event")
	}
	if usage == nil || usage.InputTokens == 0 {
		t.Error("expected token usage in Done event")
	}

	t.Logf("Content: %s", content)
	t.Logf("Tokens: input=%d output=%d thinking=%d cache_read=%d cache_write=%d",
		usage.InputTokens, usage.OutputTokens, usage.ThinkingTokens,
		usage.CacheReadTokens, usage.CacheWriteTokens)
}
