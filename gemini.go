package llm

import (
	"context"
	"fmt"
	"log/slog"
	"strings"
	"time"

	"github.com/google/uuid"
	"google.golang.org/genai"
)

func init() {
	RegisterFactory(ProviderGemini, func(cfg Config) (Provider, error) {
		return NewGemini(cfg.APIKey, cfg.RetryTimes)
	})
	RegisterFactory(ProviderGeminiVertex, func(cfg Config) (Provider, error) {
		return NewGeminiVertex(cfg.APIKey, cfg.Project, cfg.Location, cfg.RetryTimes)
	})
}

// GeminiProvider implements Provider using Google Gemini API.
type GeminiProvider struct {
	client     *genai.Client
	retryTimes int
}

// Client returns the underlying genai client for direct API access.
func (g *GeminiProvider) Client() *genai.Client {
	return g.client
}

// NewGemini creates a new Gemini provider using Google AI Studio (API Key).
func NewGemini(apiKey string, retryTimes int) (*GeminiProvider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("api key is required for gemini provider")
	}
	if retryTimes <= 0 {
		retryTimes = 3
	}

	client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		return nil, fmt.Errorf("create google ai client: %w", err)
	}

	return &GeminiProvider{
		client:     client,
		retryTimes: retryTimes,

	}, nil
}

// NewGeminiVertex creates a new Gemini provider backed by Google Vertex AI.
// If apiKey is provided, it uses Express mode (no project/location needed).
// Otherwise, it uses GCP ADC authentication with project and location.
func NewGeminiVertex(apiKey, project, location string, retryTimes int) (*GeminiProvider, error) {
	if apiKey == "" && project == "" {
		return nil, fmt.Errorf("either api key or project is required for gemini-vertex provider")
	}
	if retryTimes <= 0 {
		retryTimes = 3
	}

	cfg := &genai.ClientConfig{
		Backend: genai.BackendVertexAI,
	}
	if apiKey != "" {
		cfg.APIKey = apiKey
	} else {
		cfg.Project = project
		cfg.Location = location
	}

	client, err := genai.NewClient(context.Background(), cfg)
	if err != nil {
		return nil, fmt.Errorf("create vertex ai client: %w", err)
	}

	return &GeminiProvider{
		client:     client,
		retryTimes: retryTimes,

	}, nil
}

// Chat sends a conversation request and returns a complete response.
func (g *GeminiProvider) Chat(ctx context.Context, req ConversationRequest) (*LLMResponse, error) {
	contents := g.buildContents(req.Messages)
	genConfig := g.buildConfig(req)

	resp, err := withRetry(ctx, g.retryTimes, func() (*genai.GenerateContentResponse, error) {
		r, err := g.client.Models.GenerateContent(ctx, req.Model, contents, genConfig)
		if err != nil {
			return nil, fmt.Errorf("gemini api call: %w", err)
		}
		return r, nil
	})
	if err != nil {
		return nil, err
	}

	llmResp := g.buildLLMResponse(resp, req.Model)

	slog.DebugContext(ctx, "gemini api response",
		"model", req.Model,
		"input_tokens", llmResp.TokenUsage.InputTokens,
		"output_tokens", llmResp.TokenUsage.OutputTokens,
		"thinking_tokens", llmResp.TokenUsage.ThinkingTokens,
		"cache_read_tokens", llmResp.TokenUsage.CacheReadTokens,
	)

	return llmResp, nil
}

// ChatStream sends a streaming conversation request with fine-grained events.
func (g *GeminiProvider) ChatStream(ctx context.Context, req ConversationRequest) (<-chan StreamEvent, error) {
	contents := g.buildContents(req.Messages)
	genConfig := g.buildConfig(req)

	ch := make(chan StreamEvent, 16)

	go func() {
		defer close(ch)

		iter := g.client.Models.GenerateContentStream(ctx, req.Model, contents, genConfig)
		var totalUsage TokenUsage
		var finishReason string

		// Track state for start/end events
		textStarted := false
		thinkingStarted := false
		toolCallsStarted := map[string]bool{} // by tool call ID

		for resp, err := range iter {
			if err != nil {
				ch <- StreamEvent{Type: EventError, Error: fmt.Errorf("gemini stream: %w", err)}
				return
			}

			if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
				for _, part := range resp.Candidates[0].Content.Parts {
					// Thinking content
					if part.Thought {
						if !thinkingStarted {
							ch <- StreamEvent{Type: EventThinkingStart}
							thinkingStarted = true
						}
						ch <- StreamEvent{Type: EventThinkingDelta, Content: part.Text}
						continue
					}

					// Text content
					if part.Text != "" {
						// End thinking if was active
						if thinkingStarted {
							ch <- StreamEvent{Type: EventThinkingEnd}
							thinkingStarted = false
						}
						if !textStarted {
							ch <- StreamEvent{Type: EventTextStart}
							textStarted = true
						}
						ch <- StreamEvent{Type: EventTextDelta, Content: part.Text}
					}

					// Function call
					if part.FunctionCall != nil {
						// End text if was active
						if textStarted {
							ch <- StreamEvent{Type: EventTextEnd}
							textStarted = false
						}

						tcID := fmt.Sprintf("call_%s", uuid.New().String())
						tc := ToolCall{
							ID:           tcID,
							Name:         part.FunctionCall.Name,
							Args:         part.FunctionCall.Args,
							ProviderData: part,
						}

						if !toolCallsStarted[tcID] {
							ch <- StreamEvent{
								Type:     EventToolCallStart,
								ToolCall: &ToolCall{ID: tcID, Name: part.FunctionCall.Name},
							}
							toolCallsStarted[tcID] = true
						}
						ch <- StreamEvent{Type: EventToolCallEnd, ToolCall: &tc}
					}
				}
				finishReason = string(resp.Candidates[0].FinishReason)
			}

			if resp.UsageMetadata != nil {
				totalUsage = TokenUsage{
					InputTokens:    int(resp.UsageMetadata.PromptTokenCount),
					OutputTokens:   int(resp.UsageMetadata.CandidatesTokenCount + resp.UsageMetadata.ThoughtsTokenCount),
					ThinkingTokens: int(resp.UsageMetadata.ThoughtsTokenCount),
					CacheReadTokens:   int(resp.UsageMetadata.CachedContentTokenCount),
				}
			}
		}

		// Close any open blocks
		if thinkingStarted {
			ch <- StreamEvent{Type: EventThinkingEnd}
		}
		if textStarted {
			ch <- StreamEvent{Type: EventTextEnd}
		}

		ch <- StreamEvent{
			Type:         EventDone,
			Usage:        &totalUsage,
			FinishReason: mapFinishReason(finishReason, nil),
		}
	}()

	return ch, nil
}

// buildContents converts Messages to Gemini content format.
func (g *GeminiProvider) buildContents(messages []Message) []*genai.Content {
	contents := make([]*genai.Content, 0, len(messages))

	for _, msg := range messages {
		// Tool results -> FunctionResponse parts (role=user in Gemini)
		if msg.Role == "tool" && len(msg.ToolResults) > 0 {
			var parts []*genai.Part
			for _, tr := range msg.ToolResults {
				parts = append(parts, genai.NewPartFromFunctionResponse(tr.Name, map[string]any{
					"result": tr.Content,
				}))
				for _, img := range tr.Images {
					parts = append(parts, &genai.Part{
						InlineData: &genai.Blob{
							MIMEType: img.MimeType,
							Data:     img.Data,
						},
					})
				}
			}
			contents = append(contents, &genai.Content{
				Role:  "user",
				Parts: parts,
			})
			continue
		}

		role := "user"
		if msg.Role == "model" || msg.Role == "assistant" {
			role = "model"
		}

		parts := make([]*genai.Part, 0, 1+len(msg.Images)+len(msg.Documents)+len(msg.ToolCalls))

		for _, img := range msg.Images {
			parts = append(parts, &genai.Part{
				InlineData: &genai.Blob{
					MIMEType: img.MimeType,
					Data:     img.Data,
				},
			})
		}

		for _, doc := range msg.Documents {
			parts = append(parts, &genai.Part{
				InlineData: &genai.Blob{
					MIMEType: doc.MimeType,
					Data:     doc.Data,
				},
			})
		}

		if msg.Content != "" {
			parts = append(parts, &genai.Part{Text: msg.Content})
		}

		// Tool calls -> FunctionCall parts (for replaying model history)
		for _, tc := range msg.ToolCalls {
			if rawPart, ok := tc.ProviderData.(*genai.Part); ok {
				parts = append(parts, rawPart)
			} else {
				parts = append(parts, genai.NewPartFromFunctionCall(tc.Name, tc.Args))
			}
		}

		if len(parts) > 0 {
			contents = append(contents, &genai.Content{
				Role:  role,
				Parts: parts,
			})
		}
	}

	return contents
}

// buildConfig creates a GenerateContentConfig from the request.
func (g *GeminiProvider) buildConfig(req ConversationRequest) *genai.GenerateContentConfig {
	temperature := float32(req.ParamFloat64("temperature", 0))
	maxTokens := int32(req.ParamInt("max_output_tokens", 8192))

	topP := float32(req.ParamFloat64("top_p", 0))

	cfg := &genai.GenerateContentConfig{
		Temperature:     &temperature,
		TopP:            &topP,
		MaxOutputTokens: maxTokens,
		SafetySettings: []*genai.SafetySetting{
			{Category: "HARM_CATEGORY_HATE_SPEECH", Threshold: "OFF"},
			{Category: "HARM_CATEGORY_DANGEROUS_CONTENT", Threshold: "OFF"},
			{Category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", Threshold: "OFF"},
			{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "OFF"},
		},
	}

	if req.SystemPrompt != "" {
		cfg.SystemInstruction = &genai.Content{
			Parts: []*genai.Part{{Text: req.SystemPrompt}},
		}
	}

	// Gemini 3 models cannot fully disable thinking.
	// "disabled"/"none" clamps to the model's minimum level:
	//   pro → LOW, flash/flash-lite → MINIMAL.
	thinkingLevel := req.ParamString("thinking_level", "")
	if thinkingLevel != "" {
		tc := &genai.ThinkingConfig{}
		isProModel := strings.Contains(req.Model, "-pro")
		switch strings.ToLower(thinkingLevel) {
		case "disabled", "none":
			if isProModel {
				tc.ThinkingLevel = genai.ThinkingLevelLow
			} else {
				tc.ThinkingLevel = genai.ThinkingLevelMinimal
			}
		case "minimal":
			if isProModel {
				tc.ThinkingLevel = genai.ThinkingLevelLow
			} else {
				tc.ThinkingLevel = genai.ThinkingLevelMinimal
			}
		case "low":
			tc.ThinkingLevel = genai.ThinkingLevelLow
		case "medium":
			tc.ThinkingLevel = genai.ThinkingLevelMedium
		case "high", "xhigh", "max":
			tc.ThinkingLevel = genai.ThinkingLevelHigh
		default:
			tc.ThinkingLevel = genai.ThinkingLevelHigh
		}
		cfg.ThinkingConfig = tc
	}

	mediaRes := req.ParamString("media_resolution", "")
	if mediaRes != "" {
		switch strings.ToLower(mediaRes) {
		case "low":
			cfg.MediaResolution = genai.MediaResolutionLow
		case "medium":
			cfg.MediaResolution = genai.MediaResolutionMedium
		case "high":
			cfg.MediaResolution = genai.MediaResolutionHigh
		case "ultra_high":
			cfg.MediaResolution = "MEDIA_RESOLUTION_ULTRA_HIGH"
		default:
			cfg.MediaResolution = genai.MediaResolutionMedium
		}
	}

	if len(req.Tools) > 0 {
		cfg.Tools = []*genai.Tool{{
			FunctionDeclarations: convertToolDefs(req.Tools),
		}}
		mode := genai.FunctionCallingConfigModeAuto
		if req.ForceTool {
			mode = genai.FunctionCallingConfigModeAny
		}
		cfg.ToolConfig = &genai.ToolConfig{
			FunctionCallingConfig: &genai.FunctionCallingConfig{
				Mode: mode,
			},
		}
	}

	return cfg
}

// extractUsage extracts token usage from the response.
func (g *GeminiProvider) extractUsage(resp *genai.GenerateContentResponse) TokenUsage {
	if resp.UsageMetadata == nil {
		return TokenUsage{}
	}
	return TokenUsage{
		InputTokens:    int(resp.UsageMetadata.PromptTokenCount),
		OutputTokens:   int(resp.UsageMetadata.CandidatesTokenCount + resp.UsageMetadata.ThoughtsTokenCount),
		ThinkingTokens: int(resp.UsageMetadata.ThoughtsTokenCount),
		CacheReadTokens:   int(resp.UsageMetadata.CachedContentTokenCount),
	}
}

// CreateCachedContent creates a prompt cache for the given model and system instruction.
// Returns the cache resource name used for subsequent ChatWithCache calls.
func (g *GeminiProvider) CreateCachedContent(ctx context.Context, model, displayName, systemInstruction string, ttl time.Duration) (string, error) {
	config := &genai.CreateCachedContentConfig{
		DisplayName: displayName,
		Contents: []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{{Text: systemInstruction}},
			},
		},
		TTL: ttl,
	}

	cached, err := g.client.Caches.Create(ctx, model, config)
	if err != nil {
		return "", fmt.Errorf("create cached content: %w", err)
	}

	slog.DebugContext(ctx, "created cached content",
		"name", cached.Name,
		"display_name", displayName,
		"model", model,
	)
	return cached.Name, nil
}

// UpdateCachedContent refreshes the TTL of an existing cache.
func (g *GeminiProvider) UpdateCachedContent(ctx context.Context, cacheName string, ttl time.Duration) error {
	_, err := g.client.Caches.Update(ctx, cacheName, &genai.UpdateCachedContentConfig{
		TTL: ttl,
	})
	if err != nil {
		return fmt.Errorf("update cached content %s: %w", cacheName, err)
	}
	return nil
}

// DeleteCachedContent removes a cache.
func (g *GeminiProvider) DeleteCachedContent(ctx context.Context, cacheName string) error {
	_, err := g.client.Caches.Delete(ctx, cacheName, nil)
	if err != nil {
		return fmt.Errorf("delete cached content %s: %w", cacheName, err)
	}
	return nil
}

// ChatWithCache sends a conversation request using a cached system prompt.
func (g *GeminiProvider) ChatWithCache(ctx context.Context, cacheName string, req ConversationRequest) (*LLMResponse, error) {
	contents := g.buildContents(req.Messages)
	genConfig := g.buildConfig(req)
	genConfig.CachedContent = cacheName
	genConfig.SystemInstruction = nil // system instruction is in the cache

	resp, err := withRetry(ctx, g.retryTimes, func() (*genai.GenerateContentResponse, error) {
		r, err := g.client.Models.GenerateContent(ctx, req.Model, contents, genConfig)
		if err != nil {
			return nil, fmt.Errorf("gemini cached api call: %w", err)
		}
		return r, nil
	})
	if err != nil {
		return nil, err
	}

	llmResp := g.buildLLMResponse(resp, req.Model)

	slog.DebugContext(ctx, "gemini cached api response",
		"model", req.Model,
		"cache", cacheName,
		"input_tokens", llmResp.TokenUsage.InputTokens,
		"output_tokens", llmResp.TokenUsage.OutputTokens,
		"cache_read_tokens", llmResp.TokenUsage.CacheReadTokens,
	)

	return llmResp, nil
}

// GenerateImage generates images using a Gemini image model.
func (g *GeminiProvider) GenerateImage(ctx context.Context, req ImageRequest) (*ImageResponse, error) {
	var parts []*genai.Part

	// Add input images for editing scenarios
	for _, img := range req.Images {
		parts = append(parts, &genai.Part{
			InlineData: &genai.Blob{
				MIMEType: img.MimeType,
				Data:     img.Data,
			},
		})
	}

	parts = append(parts, &genai.Part{Text: req.Prompt})

	contents := []*genai.Content{
		{Role: "user", Parts: parts},
	}

	cfg := &genai.GenerateContentConfig{
		ResponseModalities: []string{
			string(genai.ModalityText),
			string(genai.ModalityImage),
		},
		Temperature: req.ParamFloat32("temperature"),
		TopP:        req.ParamFloat32("top_p"),
		TopK:        req.ParamFloat32("top_k"),
	}

	if v := req.ParamInt32("max_output_tokens"); v != nil {
		cfg.MaxOutputTokens = *v
	}

	if aspectRatio := req.ParamString("aspect_ratio", ""); aspectRatio != "" {
		if cfg.ImageConfig == nil {
			cfg.ImageConfig = &genai.ImageConfig{}
		}
		cfg.ImageConfig.AspectRatio = aspectRatio
	}
	if imageSize := req.ParamString("image_size", ""); imageSize != "" {
		if cfg.ImageConfig == nil {
			cfg.ImageConfig = &genai.ImageConfig{}
		}
		cfg.ImageConfig.ImageSize = imageSize
	}
	if outputMIMEType := req.ParamString("output_mime_type", ""); outputMIMEType != "" {
		if cfg.ImageConfig == nil {
			cfg.ImageConfig = &genai.ImageConfig{}
		}
		cfg.ImageConfig.OutputMIMEType = outputMIMEType
	}
	if personGen := req.ParamString("person_generation", ""); personGen != "" {
		if cfg.ImageConfig == nil {
			cfg.ImageConfig = &genai.ImageConfig{}
		}
		cfg.ImageConfig.PersonGeneration = personGen
	}
	if compressionQuality := req.ParamInt32("output_compression_quality"); compressionQuality != nil {
		if cfg.ImageConfig == nil {
			cfg.ImageConfig = &genai.ImageConfig{}
		}
		cfg.ImageConfig.OutputCompressionQuality = compressionQuality
	}

	if thinkingLevel := req.ParamString("thinking_level", ""); thinkingLevel != "" {
		cfg.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingLevel: genai.ThinkingLevel(thinkingLevel),
		}
	}
	if thinkingBudget := req.ParamInt32("thinking_budget"); thinkingBudget != nil {
		if cfg.ThinkingConfig == nil {
			cfg.ThinkingConfig = &genai.ThinkingConfig{}
		}
		cfg.ThinkingConfig.ThinkingBudget = thinkingBudget
	}

	if safetySettings, ok := req.Params["safety_settings"]; ok {
		if settings, ok := safetySettings.([]SafetySetting); ok {
			for _, s := range settings {
				cfg.SafetySettings = append(cfg.SafetySettings, &genai.SafetySetting{
					Category:  genai.HarmCategory(s.Category),
					Threshold: genai.HarmBlockThreshold(s.Threshold),
				})
			}
		}
	}

	resp, err := withRetry(ctx, g.retryTimes, func() (*genai.GenerateContentResponse, error) {
		r, err := g.client.Models.GenerateContent(ctx, req.Model, contents, cfg)
		if err != nil {
			return nil, fmt.Errorf("gemini image api call: %w", err)
		}
		return r, nil
	})
	if err != nil {
		return nil, err
	}

	imgResp := &ImageResponse{
		ModelUsed: req.Model,
	}

	imgResp.TokenUsage = g.extractUsage(resp)

	for _, candidate := range resp.Candidates {
		if candidate.Content == nil {
			continue
		}
		for _, part := range candidate.Content.Parts {
			if part.Thought {
				continue
			}
			if part.Text != "" {
				if imgResp.Text != "" {
					imgResp.Text += "\n"
				}
				imgResp.Text += part.Text
			}
			if part.InlineData != nil {
				imgResp.Images = append(imgResp.Images, GeneratedImage{
					Data:     part.InlineData.Data,
					MimeType: part.InlineData.MIMEType,
				})
			}
		}
	}

	slog.DebugContext(ctx, "gemini image api response",
		"model", req.Model,
		"images_generated", len(imgResp.Images),
		"input_tokens", imgResp.TokenUsage.InputTokens,
		"output_tokens", imgResp.TokenUsage.OutputTokens,
	)

	return imgResp, nil
}

// buildLLMResponse extracts text and tool calls from a Gemini response.
func (g *GeminiProvider) buildLLMResponse(resp *genai.GenerateContentResponse, model string) *LLMResponse {
	finishReason := ""
	if len(resp.Candidates) > 0 {
		finishReason = string(resp.Candidates[0].FinishReason)
	}

	llmResp := &LLMResponse{
		TokenUsage:   g.extractUsage(resp),
		FinishReason: mapFinishReason(finishReason, nil),
		ModelUsed:    model,
	}

	if resp.PromptFeedback != nil && resp.PromptFeedback.BlockReason != "" {
		llmResp.FinishReason = FinishSafety
	}

	if len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
		for _, part := range resp.Candidates[0].Content.Parts {
			if part.Thought && part.Text != "" {
				if llmResp.Thinking != "" {
					llmResp.Thinking += "\n"
				}
				llmResp.Thinking += part.Text
				continue
			}
			if part.Text != "" {
				llmResp.Content += part.Text
			}
			if part.FunctionCall != nil {
				llmResp.ToolCalls = append(llmResp.ToolCalls, ToolCall{
					ID:           fmt.Sprintf("call_%s", uuid.New().String()),
					Name:         part.FunctionCall.Name,
					Args:         part.FunctionCall.Args,
					ProviderData: part,
				})
			}
		}
	}

	return llmResp
}

// convertToolDefs converts ToolDefinition slice to Gemini FunctionDeclarations.
func convertToolDefs(tools []ToolDefinition) []*genai.FunctionDeclaration {
	decls := make([]*genai.FunctionDeclaration, len(tools))
	for i, t := range tools {
		decls[i] = &genai.FunctionDeclaration{
			Name:        t.Name,
			Description: t.Description,
		}
		if t.Parameters != nil {
			decls[i].Parameters = mapToSchema(t.Parameters)
		}
	}
	return decls
}

// mapToSchema converts a JSON Schema map[string]any to genai.Schema.
func mapToSchema(m map[string]any) *genai.Schema {
	s := &genai.Schema{}

	if t, ok := m["type"].(string); ok {
		s.Type = toGenaiType(t)
	}

	if desc, ok := m["description"].(string); ok {
		s.Description = desc
	}

	if enum, ok := m["enum"].([]string); ok {
		s.Enum = enum
	}

	if props, ok := m["properties"].(map[string]any); ok {
		s.Properties = make(map[string]*genai.Schema, len(props))
		for k, v := range props {
			if pm, ok := v.(map[string]any); ok {
				s.Properties[k] = mapToSchema(pm)
			}
		}
	}

	if req, ok := m["required"].([]string); ok {
		s.Required = req
	}

	return s
}

// toGenaiType maps JSON Schema type strings to genai.Type constants.
func toGenaiType(t string) genai.Type {
	types := map[string]genai.Type{
		"string":  genai.TypeString,
		"integer": genai.TypeInteger,
		"number":  genai.TypeNumber,
		"boolean": genai.TypeBoolean,
		"array":   genai.TypeArray,
		"object":  genai.TypeObject,
	}
	if gt, ok := types[t]; ok {
		return gt
	}
	return genai.TypeString
}
