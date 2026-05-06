package llm_test

import (
	"context"
	"fmt"
	"os"
	"strings"
	"testing"

	llm "github.com/hellof20/go-llm"
)

func TestGeminiChat(t *testing.T) {
	project := getEnv(t, "GCP_PROJECT")
	provider, err := llm.NewGeminiVertex("", project, getEnv(t, "GCP_LOCATION", "us-east5"), 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	resp, err := provider.Chat(context.Background(), llm.ConversationRequest{
		Model:        "gemini-3-flash-preview",
		SystemPrompt: "You are a helpful assistant. Answer concisely.",
		Messages: []llm.Message{
			{Role: "user", Content: "你觉得什么情况下AI会毁灭人类"},
		},
		Params: map[string]any{
			"max_output_tokens": 8192,
			"temperature":       1.0,
			"top_p":             0.95,
			"thinking_level":    "medium",
			"media_resolution":  "low",
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
	if resp.FinishReason == "" {
		t.Error("expected non-empty finish reason")
	}

	t.Logf("Content: %s", resp.Content)
	t.Logf("Tokens: input=%d output=%d thinking=%d cache_read=%d cache_write=%d finish=%s",
		resp.TokenUsage.InputTokens, resp.TokenUsage.OutputTokens, resp.TokenUsage.ThinkingTokens,
		resp.TokenUsage.CacheReadTokens, resp.TokenUsage.CacheWriteTokens, resp.FinishReason)
}

func TestGeminiChatStream(t *testing.T) {
	project := getEnv(t, "GCP_PROJECT")
	provider, err := llm.NewGeminiVertex("", project, getEnv(t, "GCP_LOCATION", "us-east5"), 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	ch, err := provider.ChatStream(context.Background(), llm.ConversationRequest{
		Model: "gemini-3-flash-preview",
		Messages: []llm.Message{
			{Role: "user", Content: "Count from 1 to 5."},
		},
		Params: map[string]any{
			"max_output_tokens": 256,
			"temperature":       1.0,
			"top_p":             0.95,
			"thinking_level":    "low",
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

func TestGeminiGenerateImage(t *testing.T) {
	project := getEnv(t, "GCP_PROJECT")
	provider, err := llm.NewGeminiVertex("", project, getEnv(t, "GCP_LOCATION", "us-east5"), 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	resp, err := provider.GenerateImage(context.Background(), llm.ImageRequest{
		Model:  "gemini-3.1-flash-image-preview",
		Prompt: "上海未来一周的天气预报",
		Params: map[string]any{
			"temperature":       1.0,
			"top_p":             0.95,
			"max_output_tokens": 8192,
			"thinking_level":    "HIGH",
			"aspect_ratio":      "1:1",
			"image_size":        "1K",
			"output_mime_type":  "image/png",
			"person_generation": "ALLOW_ALL",
			"safety_settings": []llm.SafetySetting{
				{Category: "HARM_CATEGORY_HATE_SPEECH", Threshold: "OFF"},
				{Category: "HARM_CATEGORY_DANGEROUS_CONTENT", Threshold: "OFF"},
				{Category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", Threshold: "OFF"},
				{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "OFF"},
			},
		},
	})
	if err != nil {
		t.Fatalf("generate image error: %v", err)
	}

	t.Logf("Text: %q", resp.Text)
	t.Logf("Images count: %d", len(resp.Images))
	t.Logf("Tokens: input=%d output=%d thinking=%d",
		resp.TokenUsage.InputTokens, resp.TokenUsage.OutputTokens, resp.TokenUsage.ThinkingTokens)

	if len(resp.Images) == 0 {
		t.Fatal("expected at least one image")
	}
	for i, img := range resp.Images {
		if len(img.Data) == 0 {
			t.Errorf("image %d: expected non-empty data", i)
		}
		if img.MimeType == "" {
			t.Errorf("image %d: expected non-empty mime type", i)
		}
		ext := strings.TrimPrefix(img.MimeType, "image/")
		path := fmt.Sprintf("bin/test_image_%d.%s", i, ext)
		if err := os.WriteFile(path, img.Data, 0644); err != nil {
			t.Errorf("image %d: save error: %v", i, err)
		} else {
			t.Logf("Image %d: %s, %d bytes -> %s", i, img.MimeType, len(img.Data), path)
		}
	}

	t.Logf("Final tokens: input=%d output=%d thinking=%d cache_read=%d cache_write=%d",
		resp.TokenUsage.InputTokens, resp.TokenUsage.OutputTokens, resp.TokenUsage.ThinkingTokens,
		resp.TokenUsage.CacheReadTokens, resp.TokenUsage.CacheWriteTokens)
}

func TestGeminiComputerUse(t *testing.T) {
	project := getEnv(t, "GCP_PROJECT")
	provider, err := llm.NewGeminiVertex("", project, getEnv(t, "GCP_LOCATION", "us-east5"), 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	resp, err := provider.Chat(context.Background(), llm.ConversationRequest{
		Model: "gemini-3-flash-preview",
		Messages: []llm.Message{
			{Role: "user", Content: "Go to google.com and search for 'weather in New York'"},
		},
		Params: map[string]any{
			"computer_use":   true,
			"thinking_level": "low",
			"temperature":    1.0,
			"top_p":          0.95,
		},
	})
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}

	t.Logf("Content: %s", resp.Content)
	t.Logf("Thinking: %s", resp.Thinking)
	t.Logf("ToolCalls count: %d", len(resp.ToolCalls))
	t.Logf("FinishReason: %s", resp.FinishReason)
	t.Logf("Tokens: input=%d output=%d thinking=%d",
		resp.TokenUsage.InputTokens, resp.TokenUsage.OutputTokens, resp.TokenUsage.ThinkingTokens)

	if len(resp.ToolCalls) == 0 && resp.Content == "" {
		t.Error("expected tool calls or content in response")
	}

	for i, tc := range resp.ToolCalls {
		t.Logf("ToolCall[%d]: name=%s id=%s args=%v", i, tc.Name, tc.ID, tc.Args)
	}
}

func TestGeminiComputerUseWithExcluded(t *testing.T) {
	project := getEnv(t, "GCP_PROJECT")
	provider, err := llm.NewGeminiVertex("", project, getEnv(t, "GCP_LOCATION", "us-east5"), 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	resp, err := provider.Chat(context.Background(), llm.ConversationRequest{
		Model: "gemini-3-flash-preview",
		Messages: []llm.Message{
			{Role: "user", Content: "Click on the search button at the center of the screen"},
		},
		Params: map[string]any{
			"computer_use":          true,
			"computer_use_excluded": "drag_and_drop,hover_at",
			"thinking_level":        "low",
		},
	})
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}

	t.Logf("Content: %s", resp.Content)
	t.Logf("ToolCalls count: %d", len(resp.ToolCalls))
	for i, tc := range resp.ToolCalls {
		t.Logf("ToolCall[%d]: name=%s args=%v", i, tc.Name, tc.Args)
		if tc.Name == "drag_and_drop" || tc.Name == "hover_at" {
			t.Errorf("excluded function %s should not be called", tc.Name)
		}
	}
}

func TestGeminiComputerUseWithCustomTools(t *testing.T) {
	project := getEnv(t, "GCP_PROJECT")
	provider, err := llm.NewGeminiVertex("", project, getEnv(t, "GCP_LOCATION", "us-east5"), 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	resp, err := provider.Chat(context.Background(), llm.ConversationRequest{
		Model: "gemini-3-flash-preview",
		Messages: []llm.Message{
			{Role: "user", Content: "Open the weather app on my phone"},
		},
		Tools: []llm.ToolDefinition{
			{
				Name:        "open_app",
				Description: "Opens an app by name on the device",
				Parameters: map[string]any{
					"type": "object",
					"properties": map[string]any{
						"app_name": map[string]any{
							"type":        "string",
							"description": "Name of the app to open",
						},
					},
					"required": []string{"app_name"},
				},
			},
		},
		Params: map[string]any{
			"computer_use":          true,
			"computer_use_excluded": "open_web_browser,search,navigate,hover_at,scroll_document,go_forward,key_combination,drag_and_drop",
			"thinking_level":        "low",
		},
	})
	if err != nil {
		t.Fatalf("chat error: %v", err)
	}

	t.Logf("Content: %s", resp.Content)
	t.Logf("ToolCalls count: %d", len(resp.ToolCalls))
	for i, tc := range resp.ToolCalls {
		t.Logf("ToolCall[%d]: name=%s args=%v", i, tc.Name, tc.Args)
	}
}
func TestGeminiEditImage(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long-running image edit test")
	}

	project := getEnv(t, "GCP_PROJECT")
	provider, err := llm.NewGeminiVertex("", project, getEnv(t, "GCP_LOCATION", "us-east5"), 3)
	if err != nil {
		t.Fatalf("create provider: %v", err)
	}

	genResp, err := provider.GenerateImage(context.Background(), llm.ImageRequest{
		Model:  "gemini-3.1-flash-image-preview",
		Prompt: "A red apple on a white table",
		Params: map[string]any{
			"aspect_ratio": "1:1",
			"image_size":   "1K",
		},
	})
	if err != nil {
		t.Fatalf("generate source image error: %v", err)
	}
	if len(genResp.Images) == 0 {
		t.Fatal("expected at least one source image")
	}
	srcExt := strings.TrimPrefix(genResp.Images[0].MimeType, "image/")
	srcPath := fmt.Sprintf("bin/test_edit_source.%s", srcExt)
	if err := os.WriteFile(srcPath, genResp.Images[0].Data, 0644); err != nil {
		t.Fatalf("save source image error: %v", err)
	}
	t.Logf("Source image: %s, %d bytes -> %s", genResp.Images[0].MimeType, len(genResp.Images[0].Data), srcPath)

	editResp, err := provider.GenerateImage(context.Background(), llm.ImageRequest{
		Model:  "gemini-3.1-flash-image-preview",
		Prompt: "Change the apple color from red to green",
		Images: []llm.Image{
			{
				MimeType: genResp.Images[0].MimeType,
				Data:     genResp.Images[0].Data,
			},
		},
		Params: map[string]any{
			"aspect_ratio": "1:1",
			"image_size":   "1K",
		},
	})
	if err != nil {
		t.Fatalf("edit image error: %v", err)
	}

	t.Logf("Text: %q", editResp.Text)
	t.Logf("Images count: %d", len(editResp.Images))

	if len(editResp.Images) == 0 {
		t.Fatal("expected at least one edited image")
	}
	for i, img := range editResp.Images {
		if len(img.Data) == 0 {
			t.Errorf("image %d: expected non-empty data", i)
		}
		ext := strings.TrimPrefix(img.MimeType, "image/")
		path := fmt.Sprintf("bin/test_edit_image_%d.%s", i, ext)
		if err := os.WriteFile(path, img.Data, 0644); err != nil {
			t.Errorf("image %d: save error: %v", i, err)
		} else {
			t.Logf("Image %d: %s, %d bytes -> %s", i, img.MimeType, len(img.Data), path)
		}
	}

	t.Logf("Tokens: input=%d output=%d thinking=%d",
		editResp.TokenUsage.InputTokens, editResp.TokenUsage.OutputTokens, editResp.TokenUsage.ThinkingTokens)
}
