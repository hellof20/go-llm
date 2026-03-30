package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"
)

func init() {
	RegisterFactory(ProviderQwen, func(cfg Config) (Provider, error) {
		return NewOpenAICompat(cfg.APIKey, cfg.BaseURL, cfg.RetryTimes)
	})
}

// OpenAICompatProvider implements Provider using OpenAI-compatible APIs (e.g. Qwen/DashScope).
type OpenAICompatProvider struct {
	apiKey     string
	baseURL    string
	retryTimes int
	httpClient *http.Client
	log        *slog.Logger
}

// SetLogger sets the logger for the OpenAI-compatible provider.
func (p *OpenAICompatProvider) SetLogger(l *slog.Logger) { p.log = l }

// NewOpenAICompat creates a new OpenAI-compatible provider.
func NewOpenAICompat(apiKey, baseURL string, retryTimes int) (*OpenAICompatProvider, error) {
	if apiKey == "" {
		return nil, fmt.Errorf("api key is required")
	}
	baseURL = strings.TrimRight(baseURL, "/")
	if baseURL == "" {
		baseURL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
	}
	if retryTimes <= 0 {
		retryTimes = 3
	}
	return &OpenAICompatProvider{
		apiKey:     apiKey,
		baseURL:    baseURL,
		retryTimes: retryTimes,
		httpClient: &http.Client{Timeout: 120 * time.Second},
		log:        slog.New(discardHandler{}),
	}, nil
}

// Chat sends a conversation request and returns a complete response.
func (p *OpenAICompatProvider) Chat(ctx context.Context, req ConversationRequest) (*LLMResponse, error) {
	body := p.buildRequestBody(req, false)

	respBody, err := withRetry(ctx, p.retryTimes, func() ([]byte, error) {
		b, err := p.doRequest(ctx, body)
		if err != nil {
			return nil, fmt.Errorf("openai compat api call: %w", err)
		}
		return b, nil
	})
	if err != nil {
		return nil, err
	}

	llmResp, err := p.parseResponse(respBody, req.Model)
	if err != nil {
		return nil, err
	}

	p.log.DebugContext(ctx, "openai compat api response",
		"model", req.Model,
		"input_tokens", llmResp.TokenUsage.InputTokens,
		"output_tokens", llmResp.TokenUsage.OutputTokens,
		"thinking_tokens", llmResp.TokenUsage.ThinkingTokens,
		"cache_read_tokens", llmResp.TokenUsage.CacheReadTokens,
	)

	return llmResp, nil
}

// ChatStream sends a streaming conversation request with fine-grained events.
func (p *OpenAICompatProvider) ChatStream(ctx context.Context, req ConversationRequest) (<-chan StreamEvent, error) {
	body := p.buildRequestBody(req, true)

	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(httpReq)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		respBody, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("api request failed: status %d, body: %s", resp.StatusCode, string(respBody))
	}

	ch := make(chan StreamEvent, 16)

	go func() {
		defer close(ch)
		defer resp.Body.Close()

		var totalUsage TokenUsage
		var finishReason string

		// Track state for start/end events
		textStarted := false
		thinkingStarted := false

		// Accumulate tool call deltas: index -> (id, name, arguments builder)
		type tcAccum struct {
			id      string
			name    string
			args    strings.Builder
			started bool
		}
		var toolCallAccums []tcAccum

		scanner := bufio.NewScanner(resp.Body)

		for scanner.Scan() {
			line := scanner.Text()
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := strings.TrimPrefix(line, "data: ")
			if data == "[DONE]" {
				break
			}

			var chunk oaiStreamChunk
			if err := json.Unmarshal([]byte(data), &chunk); err != nil {
				ch <- StreamEvent{Type: EventError, Error: fmt.Errorf("parse stream chunk: %w", err)}
				return
			}

			if len(chunk.Choices) > 0 {
				choice := chunk.Choices[0]
				delta := choice.Delta

				// Reasoning/thinking content
				if delta.ReasoningContent != "" {
					if !thinkingStarted {
						ch <- StreamEvent{Type: EventThinkingStart}
						thinkingStarted = true
					}
					ch <- StreamEvent{Type: EventThinkingDelta, Content: delta.ReasoningContent}
				}

				// Text content
				if delta.Content != "" {
					// End thinking if was active
					if thinkingStarted {
						ch <- StreamEvent{Type: EventThinkingEnd}
						thinkingStarted = false
					}
					if !textStarted {
						ch <- StreamEvent{Type: EventTextStart}
						textStarted = true
					}
					ch <- StreamEvent{Type: EventTextDelta, Content: delta.Content}
				}

				// Accumulate tool call deltas
				for _, tc := range delta.ToolCalls {
					// End text if was active
					if textStarted && tc.Index == 0 {
						ch <- StreamEvent{Type: EventTextEnd}
						textStarted = false
					}

					for tc.Index >= len(toolCallAccums) {
						toolCallAccums = append(toolCallAccums, tcAccum{})
					}
					acc := &toolCallAccums[tc.Index]
					if tc.ID != "" {
						acc.id = tc.ID
					}
					if tc.Function != nil {
						if tc.Function.Name != "" {
							acc.name = tc.Function.Name
						}
						acc.args.WriteString(tc.Function.Arguments)
					}

					// Emit start event on first encounter
					if !acc.started && (acc.id != "" || acc.name != "") {
						ch <- StreamEvent{
							Type:     EventToolCallStart,
							ToolCall: &ToolCall{ID: acc.id, Name: acc.name},
						}
						acc.started = true
					}

					// Emit delta for argument fragments
					if tc.Function != nil && tc.Function.Arguments != "" {
						ch <- StreamEvent{Type: EventToolCallDelta, ToolCallArgs: tc.Function.Arguments}
					}
				}

				if choice.FinishReason != nil {
					finishReason = *choice.FinishReason
				}
			}

			if chunk.Usage != nil {
				totalUsage = extractTokenUsage(chunk.Usage)
			}
		}

		if err := scanner.Err(); err != nil {
			ch <- StreamEvent{Type: EventError, Error: fmt.Errorf("read stream: %w", err)}
			return
		}

		// Close any open blocks
		if thinkingStarted {
			ch <- StreamEvent{Type: EventThinkingEnd}
		}
		if textStarted {
			ch <- StreamEvent{Type: EventTextEnd}
		}

		// Emit tool call end events with complete args
		for _, acc := range toolCallAccums {
			args := make(map[string]any)
			raw := acc.args.String()
			if raw != "" {
				if err := json.Unmarshal([]byte(raw), &args); err != nil {
					p.log.Warn("failed to parse streamed tool call arguments",
						"tool", acc.name, "error", err)
					args["raw"] = raw
				}
			}
			ch <- StreamEvent{
				Type: EventToolCallEnd,
				ToolCall: &ToolCall{
					ID:   acc.id,
					Name: acc.name,
					Args: args,
				},
			}
		}

		ch <- StreamEvent{
			Type:         EventDone,
			Usage:        &totalUsage,
			FinishReason: mapFinishReason(finishReason, nil),
		}
	}()

	return ch, nil
}

// buildRequestBody constructs the OpenAI-compatible request body.
func (p *OpenAICompatProvider) buildRequestBody(req ConversationRequest, stream bool) map[string]any {
	p.log.DebugContext(context.Background(), "openai compat request params", "params", req.Params)
	messages := p.buildMessages(req)

	body := map[string]any{
		"model":    req.Model,
		"messages": messages,
	}

	if maxTokens := req.ParamInt("max_tokens", 0); maxTokens > 0 {
		body["max_tokens"] = maxTokens
	}
	if temperature := req.ParamFloat64("temperature", 0); temperature > 0 {
		body["temperature"] = temperature
	}
	if topP := req.ParamFloat64("top_p", 0); topP > 0 {
		body["top_p"] = topP
	}

	if len(req.Tools) > 0 {
		body["tools"] = convertToOAITools(req.Tools)
		if req.ForceTool {
			body["tool_choice"] = "required"
		} else {
			body["tool_choice"] = "auto"
		}
	}

	if req.ParamString("thinking_level", "") != "" {
		body["enable_thinking"] = true
	}

	if req.ParamBool("vl_high_resolution_images", false) {
		body["vl_high_resolution_images"] = true
	}

	if stream {
		body["stream"] = true
		body["stream_options"] = map[string]any{"include_usage": true}
	}

	return body
}

// mapRole converts internal role names to OpenAI-compatible role names.
func mapRole(role string) string {
	if role == "model" {
		return "assistant"
	}
	return role
}

// buildMessages converts internal messages to OpenAI-compatible format.
func (p *OpenAICompatProvider) buildMessages(req ConversationRequest) []map[string]any {
	result := make([]map[string]any, 0, len(req.Messages)+1)

	if req.SystemPrompt != "" {
		result = append(result, map[string]any{
			"role":    "system",
			"content": req.SystemPrompt,
		})
	}

	// Vision parameter applied per image part
	maxPixels := req.ParamInt("max_pixels", 0)

	for _, msg := range req.Messages {
		role := mapRole(msg.Role)

		// Tool results -> individual role=tool messages
		if role == "tool" && len(msg.ToolResults) > 0 {
			for _, tr := range msg.ToolResults {
				if len(tr.Images) > 0 {
					parts := make([]map[string]any, 0, 1+len(tr.Images))
					parts = append(parts, map[string]any{
						"type": "text",
						"text": tr.Content,
					})
					for _, img := range tr.Images {
						dataURI := "data:" + img.MimeType + ";base64," + base64.StdEncoding.EncodeToString(img.Data)
						parts = append(parts, map[string]any{
							"type": "image_url",
							"image_url": map[string]any{
								"url": dataURI,
							},
						})
					}
					result = append(result, map[string]any{
						"role":         "tool",
						"tool_call_id": tr.ToolCallID,
						"content":      parts,
					})
				} else {
					result = append(result, map[string]any{
						"role":         "tool",
						"tool_call_id": tr.ToolCallID,
						"content":      tr.Content,
					})
				}
			}
			continue
		}

		// User messages with images or documents -> multipart content
		if len(msg.Images) > 0 || len(msg.Documents) > 0 {
			parts := make([]map[string]any, 0, len(msg.Images)+len(msg.Documents)+1)
			if msg.Content != "" {
				parts = append(parts, map[string]any{
					"type": "text",
					"text": msg.Content,
				})
			}
			for _, img := range msg.Images {
				dataURI := "data:" + img.MimeType + ";base64," + base64.StdEncoding.EncodeToString(img.Data)
				imgPart := map[string]any{
					"type": "image_url",
					"image_url": map[string]any{
						"url": dataURI,
					},
				}
				if maxPixels > 0 {
					imgPart["max_pixels"] = maxPixels
				}
				parts = append(parts, imgPart)
			}
			for _, doc := range msg.Documents {
				dataURI := "data:" + doc.MimeType + ";base64," + base64.StdEncoding.EncodeToString(doc.Data)
				parts = append(parts, map[string]any{
					"type": "file",
					"file": map[string]any{
						"filename":  doc.Filename,
						"file_data": dataURI,
					},
				})
			}
			result = append(result, map[string]any{
				"role":    role,
				"content": parts,
			})
			continue
		}

		m := map[string]any{
			"role":    role,
			"content": msg.Content,
		}

		// Assistant messages with tool calls
		if len(msg.ToolCalls) > 0 {
			tcs := make([]map[string]any, 0, len(msg.ToolCalls))
			for _, tc := range msg.ToolCalls {
				argsJSON, _ := json.Marshal(tc.Args)
				tcs = append(tcs, map[string]any{
					"id":   tc.ID,
					"type": "function",
					"function": map[string]any{
						"name":      tc.Name,
						"arguments": string(argsJSON),
					},
				})
			}
			m["tool_calls"] = tcs
		}

		result = append(result, m)
	}

	return result
}

// doRequest sends an HTTP request and returns the response body.
func (p *OpenAICompatProvider) doRequest(ctx context.Context, body map[string]any) ([]byte, error) {
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+p.apiKey)

	resp, err := p.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("send request: %w", err)
	}
	defer resp.Body.Close()

	respBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("api request failed: status %d, body: %s", resp.StatusCode, string(respBody))
	}

	return respBody, nil
}

// parseResponse parses a non-streaming OpenAI-compatible response.
func (p *OpenAICompatProvider) parseResponse(body []byte, model string) (*LLMResponse, error) {
	var apiResp oaiChatResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return nil, fmt.Errorf("unmarshal response: %w", err)
	}

	if len(apiResp.Choices) == 0 {
		return &LLMResponse{
			FinishReason: FinishStop,
			ModelUsed:    model,
		}, nil
	}

	choice := apiResp.Choices[0]
	llmResp := &LLMResponse{
		Content:      choice.Message.Content,
		Thinking:     choice.Message.ReasoningContent,
		FinishReason: mapFinishReason(choice.FinishReason, nil),
		ModelUsed:    model,
	}

	// Parse tool calls
	for _, tc := range choice.Message.ToolCalls {
		if tc.Function == nil {
			continue
		}
		args := make(map[string]any)
		if tc.Function.Arguments != "" {
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				p.log.Warn("failed to parse tool call arguments",
					"tool", tc.Function.Name,
					"error", err,
				)
				args["raw"] = tc.Function.Arguments
			}
		}
		llmResp.ToolCalls = append(llmResp.ToolCalls, ToolCall{
			ID:   tc.ID,
			Name: tc.Function.Name,
			Args: args,
		})
	}

	if apiResp.Usage != nil {
		llmResp.TokenUsage = extractTokenUsage(apiResp.Usage)
	}

	return llmResp, nil
}

// extractTokenUsage maps OpenAI-compatible usage to internal TokenUsage.
func extractTokenUsage(u *oaiUsage) TokenUsage {
	reasoningTokens := 0
	if u.CompletionTokensDetails != nil {
		reasoningTokens = u.CompletionTokensDetails.ReasoningTokens
	}
	cachedTokens := 0
	if u.PromptTokensDetails != nil {
		cachedTokens = u.PromptTokensDetails.CacheReadTokens
	}

	outputTokens := u.CompletionTokens - reasoningTokens
	if outputTokens < 0 {
		outputTokens = u.CompletionTokens
	}

	return TokenUsage{
		InputTokens:    u.PromptTokens,
		OutputTokens:   outputTokens,
		ThinkingTokens: reasoningTokens,
		CacheReadTokens:   cachedTokens,
	}
}

// convertToOAITools converts internal ToolDefinition to OpenAI tool format.
func convertToOAITools(tools []ToolDefinition) []map[string]any {
	result := make([]map[string]any, 0, len(tools))
	for _, t := range tools {
		fn := map[string]any{
			"name":        t.Name,
			"description": t.Description,
		}
		if t.Parameters != nil {
			fn["parameters"] = t.Parameters
		}
		result = append(result, map[string]any{
			"type":     "function",
			"function": fn,
		})
	}
	return result
}

// --- Wire types for OpenAI-compatible API ---

type oaiChatResponse struct {
	Choices []oaiChoice `json:"choices"`
	Usage   *oaiUsage   `json:"usage"`
}

type oaiChoice struct {
	Message      oaiMessage `json:"message"`
	FinishReason string     `json:"finish_reason"`
}

type oaiMessage struct {
	Content          string        `json:"content"`
	ReasoningContent string        `json:"reasoning_content,omitempty"`
	ToolCalls        []oaiToolCall `json:"tool_calls,omitempty"`
}

type oaiToolCall struct {
	ID       string       `json:"id"`
	Type     string       `json:"type"`
	Function *oaiFunction `json:"function"`
}

type oaiFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type oaiUsage struct {
	PromptTokens            int                      `json:"prompt_tokens"`
	CompletionTokens        int                      `json:"completion_tokens"`
	TotalTokens             int                      `json:"total_tokens"`
	CompletionTokensDetails *oaiCompletionTokensInfo `json:"completion_tokens_details,omitempty"`
	PromptTokensDetails     *oaiPromptTokensInfo     `json:"prompt_tokens_details,omitempty"`
}

type oaiCompletionTokensInfo struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

type oaiPromptTokensInfo struct {
	CacheReadTokens int `json:"cache_read_tokens"`
}

type oaiStreamChunk struct {
	Choices []oaiStreamChoice `json:"choices"`
	Usage   *oaiUsage         `json:"usage,omitempty"`
}

type oaiStreamChoice struct {
	Delta        oaiStreamDelta `json:"delta"`
	FinishReason *string        `json:"finish_reason,omitempty"`
}

type oaiStreamDelta struct {
	Content          string              `json:"content"`
	ReasoningContent string              `json:"reasoning_content,omitempty"`
	ToolCalls        []oaiStreamToolCall `json:"tool_calls,omitempty"`
}

type oaiStreamToolCall struct {
	Index    int          `json:"index"`
	ID       string       `json:"id,omitempty"`
	Type     string       `json:"type,omitempty"`
	Function *oaiFunction `json:"function,omitempty"`
}
