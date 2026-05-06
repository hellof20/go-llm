package llm_test

import (
	"context"
	"testing"

	llm "github.com/hellof20/go-llm"
)

func TestVertexOpenAIChat(t *testing.T) {
	project := getEnv(t, "GCP_PROJECT")
	provider, err := llm.NewVertexOpenAI(project, getEnv(t, "GCP_LOCATION", "us-central1"), 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	resp, err := provider.Chat(context.Background(), llm.ConversationRequest{
		Model:        "google/gemma-4-26b-a4b-it-maas",
		SystemPrompt: "You are a helpful assistant. Answer concisely.",
		Messages: []llm.Message{
			{Role: "user", Content: "Summer travel plan to Paris"},
		},
		Params: map[string]any{
			"max_tokens":     8192,
			"temperature":    1.0,
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

	t.Logf("Content: %s", resp.Content)
	t.Logf("Thinking: %s", resp.Thinking)
	t.Logf("Tokens: input=%d output=%d thinking=%d finish=%s",
		resp.TokenUsage.InputTokens, resp.TokenUsage.OutputTokens,
		resp.TokenUsage.ThinkingTokens, resp.FinishReason)
}

func TestVertexOpenAIChatStream(t *testing.T) {
	project := getEnv(t, "GCP_PROJECT")
	provider, err := llm.NewVertexOpenAI(project, getEnv(t, "GCP_LOCATION", "us-central1"), 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	ch, err := provider.ChatStream(context.Background(), llm.ConversationRequest{
		Model: "google/gemma-4-26b-a4b-it-maas",
		Messages: []llm.Message{
			{Role: "user", Content: "Count from 1 to 5."},
		},
		Params: map[string]any{
			"max_tokens":     256,
			"temperature":    1.0,
			"top_p":          0.95,
			"thinking_level": "high",
		},
	})
	if err != nil {
		t.Fatalf("stream error: %v", err)
	}

	var content string
	var thinking string
	var gotDone bool
	var usage *llm.TokenUsage
	for event := range ch {
		switch event.Type {
		case llm.EventTextDelta:
			content += event.Content
		case llm.EventThinkingDelta:
			thinking += event.Content
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

	t.Logf("Content: %s", content)
	t.Logf("Thinking: %s", thinking)
	if usage != nil {
		t.Logf("Tokens: input=%d output=%d thinking=%d",
			usage.InputTokens, usage.OutputTokens, usage.ThinkingTokens)
	}
}
