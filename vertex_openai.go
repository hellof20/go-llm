package llm

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"strings"
	"time"

	"golang.org/x/oauth2"
	"golang.org/x/oauth2/google"
)

func init() {
	RegisterFactory(ProviderVertexOpenAI, func(cfg Config) (Provider, error) {
		return NewVertexOpenAI(cfg.Project, cfg.Location, cfg.RetryTimes)
	})
}

type VertexOpenAIProvider struct {
	baseURL     string
	tokenSource oauth2.TokenSource
	retryTimes  int
	httpClient  *http.Client
}

func NewVertexOpenAI(project, location string, retryTimes int) (*VertexOpenAIProvider, error) {
	if project == "" {
		return nil, fmt.Errorf("project is required for vertex-openai provider")
	}
	if location == "" {
		location = "us-central1"
	}
	if retryTimes <= 0 {
		retryTimes = 3
	}

	ts, err := google.DefaultTokenSource(context.Background(), "https://www.googleapis.com/auth/cloud-platform")
	if err != nil {
		return nil, fmt.Errorf("create token source: %w", err)
	}

	var baseURL string
	if location == "global" {
		baseURL = fmt.Sprintf("https://aiplatform.googleapis.com/v1/projects/%s/locations/global/endpoints/openapi", project)
	} else {
		baseURL = fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/endpoints/openapi", location, project, location)
	}

	return &VertexOpenAIProvider{
		baseURL:     baseURL,
		tokenSource: ts,
		retryTimes:  retryTimes,
		httpClient:  &http.Client{Timeout: 120 * time.Second},
	}, nil
}

func (p *VertexOpenAIProvider) Chat(ctx context.Context, req ConversationRequest) (*LLMResponse, error) {
	body := p.buildRequestBody(req, false)

	respBody, err := withRetry(ctx, p.retryTimes, func() ([]byte, error) {
		b, err := p.doRequest(ctx, body)
		if err != nil {
			return nil, fmt.Errorf("vertex openai api call: %w", err)
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

	slog.DebugContext(ctx, "vertex openai api response",
		"model", req.Model,
		"input_tokens", llmResp.TokenUsage.InputTokens,
		"output_tokens", llmResp.TokenUsage.OutputTokens,
		"thinking_tokens", llmResp.TokenUsage.ThinkingTokens,
	)

	return llmResp, nil
}

func (p *VertexOpenAIProvider) ChatStream(ctx context.Context, req ConversationRequest) (<-chan StreamEvent, error) {
	body := p.buildRequestBody(req, true)

	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	token, err := p.tokenSource.Token()
	if err != nil {
		return nil, fmt.Errorf("get access token: %w", err)
	}

	httpReq, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	httpReq.Header.Set("Content-Type", "application/json")
	httpReq.Header.Set("Authorization", "Bearer "+token.AccessToken)

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

		textStarted := false
		thinkingStarted := false

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

				if delta.ReasoningContent != "" {
					if !thinkingStarted {
						ch <- StreamEvent{Type: EventThinkingStart}
						thinkingStarted = true
					}
					ch <- StreamEvent{Type: EventThinkingDelta, Content: delta.ReasoningContent}
				}

				if delta.Content != "" {
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

				for _, tc := range delta.ToolCalls {
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

					if !acc.started && (acc.id != "" || acc.name != "") {
						ch <- StreamEvent{
							Type:     EventToolCallStart,
							ToolCall: &ToolCall{ID: acc.id, Name: acc.name},
						}
						acc.started = true
					}

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

		if thinkingStarted {
			ch <- StreamEvent{Type: EventThinkingEnd}
		}
		if textStarted {
			ch <- StreamEvent{Type: EventTextEnd}
		}

		for _, acc := range toolCallAccums {
			args := make(map[string]any)
			raw := acc.args.String()
			if raw != "" {
				if err := json.Unmarshal([]byte(raw), &args); err != nil {
					slog.Warn("failed to parse streamed tool call arguments",
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

func (p *VertexOpenAIProvider) buildRequestBody(req ConversationRequest, stream bool) map[string]any {
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

	if tl := req.ParamString("thinking_level", ""); tl != "" && tl != "disabled" && tl != "none" {
		body["chat_template_kwargs"] = map[string]any{"enable_thinking": true}
	}

	if stream {
		body["stream"] = true
		body["stream_options"] = map[string]any{"include_usage": true}
	}

	return body
}

func (p *VertexOpenAIProvider) buildMessages(req ConversationRequest) []map[string]any {
	result := make([]map[string]any, 0, len(req.Messages)+1)

	if req.SystemPrompt != "" {
		result = append(result, map[string]any{
			"role":    "system",
			"content": req.SystemPrompt,
		})
	}

	for _, msg := range req.Messages {
		role := mapRole(msg.Role)

		if role == "tool" && len(msg.ToolResults) > 0 {
			for _, tr := range msg.ToolResults {
				result = append(result, map[string]any{
					"role":         "tool",
					"tool_call_id": tr.ToolCallID,
					"content":      tr.Content,
				})
			}
			continue
		}

		m := map[string]any{
			"role":    role,
			"content": msg.Content,
		}

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

func (p *VertexOpenAIProvider) doRequest(ctx context.Context, body map[string]any) ([]byte, error) {
	jsonData, err := json.Marshal(body)
	if err != nil {
		return nil, fmt.Errorf("marshal request: %w", err)
	}

	token, err := p.tokenSource.Token()
	if err != nil {
		return nil, fmt.Errorf("get access token: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", p.baseURL+"/chat/completions", bytes.NewReader(jsonData))
	if err != nil {
		return nil, fmt.Errorf("create request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+token.AccessToken)

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

func (p *VertexOpenAIProvider) parseResponse(body []byte, model string) (*LLMResponse, error) {
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

	for _, tc := range choice.Message.ToolCalls {
		if tc.Function == nil {
			continue
		}
		args := make(map[string]any)
		if tc.Function.Arguments != "" {
			if err := json.Unmarshal([]byte(tc.Function.Arguments), &args); err != nil {
				slog.Warn("failed to parse tool call arguments",
					"tool", tc.Function.Name, "error", err)
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
