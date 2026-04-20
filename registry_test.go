package llm_test

import (
	"context"
	"testing"

	llm "github.com/hellof20/go-llm"
)

func TestProviderRegistryChat(t *testing.T) {
	project := getEnv(t, "GCP_PROJECT")

	registry := llm.NewProviderRegistry("gemini-vertex")
	gemini, err := llm.NewGeminiVertex("", project, getEnv(t, "GCP_LOCATION", "us-east5"), 3)
	if err != nil {
		t.Fatalf("create gemini: %v", err)
	}
	registry.Register("gemini-vertex", gemini)

	resp, err := registry.Chat(context.Background(), llm.ConversationRequest{
		Model:    "gemini-2.5-flash",
		Messages: []llm.Message{{Role: "user", Content: "Say hello in one word."}},
		Params:   map[string]any{"max_output_tokens": 64},
	})
	if err != nil {
		t.Fatalf("registry chat error: %v", err)
	}

	if resp.Content == "" {
		t.Error("expected non-empty content")
	}
	t.Logf("Content: %s", resp.Content)
}

func TestProviderRegistryGenerateImage(t *testing.T) {
	project := getEnv(t, "GCP_PROJECT")

	registry := llm.NewProviderRegistry("gemini-vertex")
	gemini, err := llm.NewGeminiVertex("", project, getEnv(t, "GCP_LOCATION", "us-east5"), 3)
	if err != nil {
		t.Fatalf("create gemini: %v", err)
	}
	registry.Register("gemini-vertex", gemini)

	resp, err := registry.GenerateImage(context.Background(), llm.ImageRequest{
		Provider: "gemini-vertex",
		Model:    "gemini-2.0-flash-preview-image-generation",
		Prompt:   "A blue square",
	})
	if err != nil {
		t.Fatalf("registry image error: %v", err)
	}

	if len(resp.Images) == 0 {
		t.Fatal("expected at least one image")
	}
	t.Logf("Generated %d image(s)", len(resp.Images))
}

func TestProviderRegistryImageUnsupported(t *testing.T) {
	registry := llm.NewProviderRegistry("none")

	_, err := registry.GenerateImage(context.Background(), llm.ImageRequest{
		Provider: "nonexistent",
		Model:    "fake-model",
		Prompt:   "test",
	})
	if err == nil {
		t.Error("expected error for non-existent provider")
	}
	t.Logf("Expected error: %v", err)
}
