package llm_test

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"

	llm "github.com/hellof20/go-llm"
)

func TestQwenChat(t *testing.T) {
	apiKey := getEnv(t,"DASHSCOPE_API_KEY")
	provider, err := llm.NewOpenAICompat(apiKey, "", 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	resp, err := provider.Chat(context.Background(), llm.ConversationRequest{
		Model:        "qwen3.5-plus",
		SystemPrompt: "You are a helpful assistant. Answer concisely.",
		Messages: []llm.Message{
			{Role: "user", Content: "What is 1+1? Answer with just the number."},
		},
		Params: map[string]any{
			"max_tokens":  256,
			"temperature": 0.0,
			"top_p":       0.95,
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

func TestQwenChatStream(t *testing.T) {
	apiKey := getEnv(t,"DASHSCOPE_API_KEY")
	provider, err := llm.NewOpenAICompat(apiKey, "", 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	ch, err := provider.ChatStream(context.Background(), llm.ConversationRequest{
		Model: "qwen3.5-plus",
		Messages: []llm.Message{
			{Role: "user", Content: "Count from 1 to 5."},
		},
		Params: map[string]any{
			"max_tokens":  256,
			"temperature": 0.5,
			"top_p":       0.95,
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

func TestQwenChatCache(t *testing.T) {
	apiKey := getEnv(t,"DASHSCOPE_API_KEY")
	provider, err := llm.NewOpenAICompat(apiKey, "", 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	longContent := strings.Repeat(fmt.Sprintf("You are an expert assistant created at timestamp %d with deep knowledge in software engineering, cloud computing, distributed systems, and artificial intelligence. ", time.Now().UnixNano()), 150)
	req := llm.ConversationRequest{
		Model:        "qwen3.5-plus",
		SystemPrompt: longContent + "\nAlways answer concisely.",
		Messages:     []llm.Message{{Role: "user", Content: "What is 1+1? Answer with just the number."}},
		Params:       map[string]any{"max_tokens": 256, "temperature": 0.0, "enable_cache": true},
	}

	resp1, err := provider.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("request 1 error: %v", err)
	}
	t.Logf("Request 1: input=%d output=%d thinking=%d cache_read=%d cache_write=%d",
		resp1.TokenUsage.InputTokens, resp1.TokenUsage.OutputTokens, resp1.TokenUsage.ThinkingTokens,
		resp1.TokenUsage.CacheReadTokens, resp1.TokenUsage.CacheWriteTokens)

	time.Sleep(2 * time.Second)

	resp2, err := provider.Chat(context.Background(), req)
	if err != nil {
		t.Fatalf("request 2 error: %v", err)
	}
	t.Logf("Request 2: input=%d output=%d thinking=%d cache_read=%d cache_write=%d",
		resp2.TokenUsage.InputTokens, resp2.TokenUsage.OutputTokens, resp2.TokenUsage.ThinkingTokens,
		resp2.TokenUsage.CacheReadTokens, resp2.TokenUsage.CacheWriteTokens)

	if resp1.TokenUsage.CacheWriteTokens == 0 {
		t.Error("expected cache write tokens in request 1")
	}
	if resp2.TokenUsage.CacheReadTokens == 0 {
		t.Error("expected cache read tokens in request 2")
	}
}
