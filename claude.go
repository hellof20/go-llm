package llm

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"log/slog"
	"strings"

	"github.com/anthropics/anthropic-sdk-go"

	"github.com/anthropics/anthropic-sdk-go/option"
	"github.com/anthropics/anthropic-sdk-go/packages/param"
	"github.com/anthropics/anthropic-sdk-go/vertex"
)

func init() {
	RegisterFactory(ProviderClaude, func(cfg Config) (Provider, error) {
		return NewClaude(cfg.APIKey, cfg.RetryTimes)
	})
	RegisterFactory(ProviderClaudeVertex, func(cfg Config) (Provider, error) {
		return NewClaudeVertex(cfg.Project, cfg.Location, cfg.RetryTimes)
	})
}

// ClaudeProvider implements Provider using the Anthropic Messages API.
type ClaudeProvider struct {
	client     *anthropic.Client
	retryTimes int
	log        *slog.Logger
}

// SetLogger sets the logger for the Claude provider.
func (p *ClaudeProvider) SetLogger(l *slog.Logger) { p.log = l }

// NewClaude creates a new Claude provider using the Anthropic API directly.
func NewClaude(apiKey string, retryTimes int) (*ClaudeProvider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("api key is required for claude provider")
	}
	if retryTimes <= 0 {
		retryTimes = 3
	}

	client := anthropic.NewClient(
		option.WithAPIKey(apiKey),
		option.WithMaxRetries(retryTimes),
	)

	return &ClaudeProvider{
		client:     &client,
		retryTimes: retryTimes,
		log:        slog.New(discardHandler{}),
	}, nil
}

// NewClaudeVertex creates a new Claude provider backed by Google Vertex AI.
func NewClaudeVertex(project, location string, retryTimes int) (*ClaudeProvider, error) {
	if project == "" {
		return nil, fmt.Errorf("project is required for claude-vertex provider")
	}
	if location == "" {
		location = "us-east5"
	}
	if retryTimes <= 0 {
		retryTimes = 3
	}

	client := anthropic.NewClient(
		vertex.WithGoogleAuth(context.Background(), location, project,
			"https://www.googleapis.com/auth/cloud-platform",
		),
		option.WithMaxRetries(retryTimes),
	)

	return &ClaudeProvider{
		client:     &client,
		retryTimes: retryTimes,
		log:        slog.New(discardHandler{}),
	}, nil
}

// Chat sends a conversation request and returns a complete response.
func (p *ClaudeProvider) Chat(ctx context.Context, req ConversationRequest) (*LLMResponse, error) {
	params := p.buildParams(req)

	resp, err := withRetry(ctx, p.retryTimes, func() (*anthropic.Message, error) {
		msg, err := p.client.Messages.New(ctx, params)
		if err != nil {
			return nil, fmt.Errorf("claude api call: %w", err)
		}
		return msg, nil
	})
	if err != nil {
		return nil, err
	}

	llmResp := p.parseMessage(resp, req.Model)

	p.log.DebugContext(ctx, "claude api response",
		"model", req.Model,
		"input_tokens", llmResp.TokenUsage.InputTokens,
		"output_tokens", llmResp.TokenUsage.OutputTokens,
		"thinking_tokens", llmResp.TokenUsage.ThinkingTokens,
		"cache_read_tokens", llmResp.TokenUsage.CacheReadTokens,
		"cache_write_tokens", llmResp.TokenUsage.CacheWriteTokens,
	)

	return llmResp, nil
}

// ChatStream sends a streaming conversation request with fine-grained events.
func (p *ClaudeProvider) ChatStream(ctx context.Context, req ConversationRequest) (<-chan StreamEvent, error) {
	params := p.buildParams(req)

	stream := p.client.Messages.NewStreaming(ctx, params)
	if err := stream.Err(); err != nil {
		stream.Close()
		return nil, fmt.Errorf("claude stream: %w", err)
	}

	ch := make(chan StreamEvent, 16)

	go func() {
		defer close(ch)
		defer stream.Close()

		var totalUsage TokenUsage
		var finishReason string

		// Track active content blocks by index
		type blockState struct {
			blockType string // "text", "thinking", "tool_use"
			id        string
			name      string
			input     string
		}
		blocks := map[int64]*blockState{}

		for stream.Next() {
			event := stream.Current()

			switch event.Type {
			case "content_block_start":
				cb := event.ContentBlock
				state := &blockState{blockType: string(cb.Type)}

				switch cb.Type {
				case "text":
					ch <- StreamEvent{Type: EventTextStart}
				case "thinking":
					ch <- StreamEvent{Type: EventThinkingStart}
				case "tool_use":
					state.id = cb.ID
					state.name = cb.Name
					ch <- StreamEvent{
						Type:     EventToolCallStart,
						ToolCall: &ToolCall{ID: cb.ID, Name: cb.Name},
					}
				}
				blocks[event.Index] = state

			case "content_block_delta":
				delta := event.Delta
				state := blocks[event.Index]

				if delta.Type == "text_delta" && delta.Text != "" {
					ch <- StreamEvent{Type: EventTextDelta, Content: delta.Text}
				}
				if delta.Type == "thinking_delta" && delta.Thinking != "" {
					ch <- StreamEvent{Type: EventThinkingDelta, Content: delta.Thinking}
				}
				if delta.Type == "input_json_delta" && delta.PartialJSON != "" {
					if state != nil {
						state.input += delta.PartialJSON
					}
					ch <- StreamEvent{Type: EventToolCallDelta, ToolCallArgs: delta.PartialJSON}
				}

			case "content_block_stop":
				state := blocks[event.Index]
				if state == nil {
					continue
				}
				switch state.blockType {
				case "text":
					ch <- StreamEvent{Type: EventTextEnd}
				case "thinking":
					ch <- StreamEvent{Type: EventThinkingEnd}
				case "tool_use":
					args := make(map[string]any)
					if state.input != "" {
						if err := json.Unmarshal([]byte(state.input), &args); err != nil {
							p.log.Warn("failed to parse streamed tool call input",
								"tool", state.name, "error", err)
							args["raw"] = state.input
						}
					}
					ch <- StreamEvent{
						Type: EventToolCallEnd,
						ToolCall: &ToolCall{
							ID:   state.id,
							Name: state.name,
							Args: args,
						},
					}
				}
				delete(blocks, event.Index)

			case "message_delta":
				if event.Delta.StopReason != "" {
					finishReason = mapFinishReason(string(event.Delta.StopReason), nil)
				}
				totalUsage.OutputTokens = int(event.Usage.OutputTokens)
				if event.Usage.CacheReadInputTokens > 0 {
					totalUsage.CacheReadTokens = int(event.Usage.CacheReadInputTokens)
				}
				if event.Usage.CacheCreationInputTokens > 0 {
					totalUsage.CacheWriteTokens = int(event.Usage.CacheCreationInputTokens)
				}
				p.log.DebugContext(ctx, "claude stream message_delta usage",
					"output_tokens", event.Usage.OutputTokens,
					"cache_read", event.Usage.CacheReadInputTokens,
					"cache_write", event.Usage.CacheCreationInputTokens,
				)

			case "message_start":
				msg := event.Message
				totalUsage.InputTokens = int(msg.Usage.InputTokens)
				totalUsage.CacheReadTokens = int(msg.Usage.CacheReadInputTokens)
				totalUsage.CacheWriteTokens = int(msg.Usage.CacheCreationInputTokens)
				p.log.DebugContext(ctx, "claude stream message_start usage",
					"input_tokens", msg.Usage.InputTokens,
					"cache_read", msg.Usage.CacheReadInputTokens,
					"cache_write", msg.Usage.CacheCreationInputTokens,
				)
			}
		}

		if err := stream.Err(); err != nil {
			ch <- StreamEvent{Type: EventError, Error: fmt.Errorf("claude stream: %w", err)}
			return
		}

		ch <- StreamEvent{
			Type:         EventDone,
			Usage:        &totalUsage,
			FinishReason: finishReason,
		}
	}()

	return ch, nil
}

// buildParams constructs the Anthropic MessageNewParams from internal request.
func (p *ClaudeProvider) buildParams(req ConversationRequest) anthropic.MessageNewParams {
	maxTokens := req.ParamInt("max_tokens", 8192)

	params := anthropic.MessageNewParams{
		Model:     anthropic.Model(req.Model),
		MaxTokens: int64(maxTokens),
		Messages:  p.buildMessages(req),
	}

	if req.SystemPrompt != "" {
		block := anthropic.TextBlockParam{
			Text:         req.SystemPrompt,
			CacheControl: anthropic.NewCacheControlEphemeralParam(),
		}
		params.System = []anthropic.TextBlockParam{block}
	}

	if temp := req.ParamFloat64("temperature", -1); temp >= 0 {
		params.Temperature = anthropic.Float(temp)
	}
	if topP := req.ParamFloat64("top_p", -1); topP >= 0 {
		params.TopP = anthropic.Float(topP)
	}
	if topK := req.ParamInt("top_k", 0); topK > 0 {
		params.TopK = anthropic.Int(int64(topK))
	}

	// Adaptive thinking: read unified thinking_level and map to Claude effort
	// "disabled"/"none" explicitly disables thinking (no thinking param sent).
	if effort := req.ParamString("thinking_level", ""); effort != "" && effort != "disabled" && effort != "none" {
		// Map agent-level thinking levels to Claude effort values
		// "max" effort is only supported on Opus 4.6
		isOpus46 := strings.Contains(req.Model, "opus-4-6") || strings.Contains(req.Model, "opus-4.6")
		var mapped anthropic.OutputConfigEffort
		switch strings.ToLower(effort) {
		case "low", "minimal":
			mapped = anthropic.OutputConfigEffortLow
		case "medium":
			mapped = anthropic.OutputConfigEffortMedium
		case "high":
			mapped = anthropic.OutputConfigEffortHigh
		case "xhigh", "max":
			if isOpus46 {
				mapped = anthropic.OutputConfigEffortMax
			} else {
				mapped = anthropic.OutputConfigEffortHigh
			}
		default:
			mapped = anthropic.OutputConfigEffortHigh
		}
		adaptive := anthropic.NewThinkingConfigAdaptiveParam()
		params.Thinking = anthropic.ThinkingConfigParamUnion{
			OfAdaptive: &adaptive,
		}
		params.OutputConfig = anthropic.OutputConfigParam{
			Effort: mapped,
		}
	}

	// Tools
	// Claude does not allow forced tool_choice when thinking is enabled.
	thinkingEnabled := params.Thinking.OfAdaptive != nil || params.Thinking.OfEnabled != nil
	if len(req.Tools) > 0 {
		params.Tools = convertToClaudeTools(req.Tools)
		if req.ForceTool && !thinkingEnabled {
			params.ToolChoice = anthropic.ToolChoiceUnionParam{
				OfAny: &anthropic.ToolChoiceAnyParam{},
			}
		} else {
			params.ToolChoice = anthropic.ToolChoiceUnionParam{
				OfAuto: &anthropic.ToolChoiceAutoParam{},
			}
		}
	}

	return params
}

// buildMessages converts internal messages to Anthropic MessageParam format.
func (p *ClaudeProvider) buildMessages(req ConversationRequest) []anthropic.MessageParam {
	result := make([]anthropic.MessageParam, 0, len(req.Messages))

	for _, msg := range req.Messages {
		role := msg.Role
		if role == "model" {
			role = "assistant"
		}

		// Tool results -> user message with tool_result blocks
		if role == "tool" && len(msg.ToolResults) > 0 {
			var blocks []anthropic.ContentBlockParamUnion
			for _, tr := range msg.ToolResults {
				blocks = append(blocks, anthropic.NewToolResultBlock(tr.ToolCallID, tr.Content, tr.IsError))
			}
			result = append(result, anthropic.NewUserMessage(blocks...))
			continue
		}

		var blocks []anthropic.ContentBlockParamUnion

		if msg.Content != "" {
			blocks = append(blocks, anthropic.NewTextBlock(msg.Content))
		}

		for _, img := range msg.Images {
			encoded := base64.StdEncoding.EncodeToString(img.Data)
			blocks = append(blocks, anthropic.NewImageBlockBase64(img.MimeType, encoded))
		}

		for _, doc := range msg.Documents {
			encoded := base64.StdEncoding.EncodeToString(doc.Data)
			blocks = append(blocks, anthropic.NewDocumentBlock(anthropic.Base64PDFSourceParam{
				Data: encoded,
			}))
		}

		// Assistant tool calls
		if len(msg.ToolCalls) > 0 {
			for _, tc := range msg.ToolCalls {
				inputJSON, _ := json.Marshal(tc.Args)
				blocks = append(blocks, anthropic.NewToolUseBlock(tc.ID, json.RawMessage(inputJSON), tc.Name))
			}
		}

		if len(blocks) == 0 {
			blocks = append(blocks, anthropic.NewTextBlock(""))
		}

		claudeRole := anthropic.MessageParamRoleUser
		if role == "assistant" {
			claudeRole = anthropic.MessageParamRoleAssistant
		}

		result = append(result, anthropic.MessageParam{
			Role:    claudeRole,
			Content: blocks,
		})
	}

	// Add cache_control to the last non-assistant message's last content block.
	// This caches system prompt + tools + all messages up to this point.
	for i := len(result) - 1; i >= 0; i-- {
		if result[i].Role != anthropic.MessageParamRoleAssistant && len(result[i].Content) > 0 {
			lastBlock := &result[i].Content[len(result[i].Content)-1]
			if cc := lastBlock.GetCacheControl(); cc != nil {
				*cc = anthropic.NewCacheControlEphemeralParam()
			}
			break
		}
	}

	return result
}

// parseMessage converts an Anthropic Message response to internal LLMResponse.
func (p *ClaudeProvider) parseMessage(msg *anthropic.Message, model string) *LLMResponse {
	resp := &LLMResponse{
		FinishReason: mapFinishReason(string(msg.StopReason), nil),
		ModelUsed:    model,
	}

	for _, block := range msg.Content {
		switch block.Type {
		case "thinking":
			if resp.Thinking != "" {
				resp.Thinking += "\n"
			}
			resp.Thinking += block.Thinking
		case "text":
			if resp.Content != "" {
				resp.Content += "\n"
			}
			resp.Content += block.Text
		case "tool_use":
			args := make(map[string]any)
			if len(block.Input) > 0 {
				if err := json.Unmarshal(block.Input, &args); err != nil {
					p.log.Warn("failed to parse tool call input",
						"tool", block.Name, "error", err)
					args["raw"] = string(block.Input)
				}
			}
			resp.ToolCalls = append(resp.ToolCalls, ToolCall{
				ID:   block.ID,
				Name: block.Name,
				Args: args,
			})
		}
	}

	resp.TokenUsage = TokenUsage{
		InputTokens:         int(msg.Usage.InputTokens),
		OutputTokens:        int(msg.Usage.OutputTokens),
		CacheReadTokens:        int(msg.Usage.CacheReadInputTokens),
		CacheWriteTokens: int(msg.Usage.CacheCreationInputTokens),
	}

	return resp
}

// convertToClaudeTools converts internal ToolDefinition to Anthropic tool format.
func convertToClaudeTools(tools []ToolDefinition) []anthropic.ToolUnionParam {
	result := make([]anthropic.ToolUnionParam, 0, len(tools))
	for _, t := range tools {
		schema := anthropic.ToolInputSchemaParam{}
		if t.Parameters != nil {
			if props, ok := t.Parameters["properties"]; ok {
				schema.Properties = props
			}
			if req, ok := t.Parameters["required"]; ok {
				if reqSlice, ok := req.([]any); ok {
					for _, r := range reqSlice {
						if s, ok := r.(string); ok {
							schema.Required = append(schema.Required, s)
						}
					}
				}
			}
		}

		tool := anthropic.ToolUnionParamOfTool(schema, t.Name)
		if t.Description != "" {
			tool.OfTool.Description = param.Opt[string]{Value: t.Description}
		}
		result = append(result, tool)
	}
	return result
}
