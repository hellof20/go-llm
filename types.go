package llm

import "encoding/json"

// Event type constants for streaming responses.
const (
	EventTextStart     = "text_start"
	EventTextDelta     = "text_delta"
	EventTextEnd       = "text_end"
	EventThinkingStart = "thinking_start"
	EventThinkingDelta = "thinking_delta"
	EventThinkingEnd   = "thinking_end"
	EventToolCallStart = "toolcall_start"
	EventToolCallDelta = "toolcall_delta"
	EventToolCallEnd   = "toolcall_end"
	EventDone          = "done"
	EventError         = "error"
)

// Model bundles a model ID with its provider name, enabling provider routing
// via a single SetModel call.
type Model struct {
	ID       string `json:"id"`       // model ID, e.g. "claude-sonnet-4-6"
	Provider string `json:"provider"` // provider name, e.g. "claude-vertex"
}

// Standard finish reason constants.
const (
	FinishStop            = "stop"
	FinishLength          = "length"
	FinishToolCalls       = "tool_calls"
	FinishSafety          = "safety"
	FinishContextOverflow = "context_overflow"
)

// StreamEvent represents a single event in a streaming response.
type StreamEvent struct {
	Type         string      // event type constant
	Content      string      // text or thinking delta content
	ToolCall     *ToolCall   // toolcall_start: ID+Name; toolcall_end: complete with Args
	ToolCallArgs string      // toolcall_delta: partial JSON arguments
	FinishReason string      // done event
	Usage        *TokenUsage // done event
	Error        error       // error event
}

// TokenUsage holds token usage information from an LLM API call.
type TokenUsage struct {
	InputTokens         int `json:"inputTokens"`
	OutputTokens        int `json:"outputTokens"`
	ThinkingTokens      int `json:"thinkingTokens,omitempty"`
	CacheReadTokens        int `json:"cacheReadTokens,omitempty"`
	CacheWriteTokens int `json:"cacheWriteTokens,omitempty"`
}

// Add merges another TokenUsage into this one.
func (t *TokenUsage) Add(other *TokenUsage) {
	if other == nil {
		return
	}
	t.InputTokens += other.InputTokens
	t.OutputTokens += other.OutputTokens
	t.ThinkingTokens += other.ThinkingTokens
	t.CacheReadTokens += other.CacheReadTokens
	t.CacheWriteTokens += other.CacheWriteTokens
}

// Message represents a single message in a conversation.
type Message struct {
	Role        string       `json:"role"`
	Content     string       `json:"content"`
	Images      []Image      `json:"images,omitempty"`
	Documents   []Document   `json:"documents,omitempty"`
	ToolCalls   []ToolCall   `json:"toolCalls,omitempty"`
	ToolResults []ToolResult `json:"toolResults,omitempty"`
}

// Image represents an image attachment in a message.
type Image struct {
	MimeType string `json:"mimeType"`
	Data     []byte `json:"data"`
}

// Document represents a non-image file (PDF, etc.) for native LLM processing.
type Document struct {
	MimeType string `json:"mimeType"`
	Filename string `json:"filename"`
	Data     []byte `json:"data"`
}

// ToolCall represents a tool/function call requested by the LLM.
type ToolCall struct {
	ID           string         `json:"id"`
	Name         string         `json:"name"`
	Args         map[string]any `json:"args"`
	ProviderData any            `json:"-"` // opaque provider-specific data for multi-turn replay
}

// ToolResult contains the output of a tool execution.
type ToolResult struct {
	ToolCallID string         `json:"toolCallId"`
	Name       string         `json:"name"`
	Content    string         `json:"content"`
	IsError    bool           `json:"isError,omitempty"`
	Images     []Image        `json:"images,omitempty"`
	Extra      map[string]any `json:"extra,omitempty"`
}

// LLMResponse represents the unified response from an LLM provider.
type LLMResponse struct {
	Content      string     `json:"content"`
	Thinking     string     `json:"thinking,omitempty"`
	ToolCalls    []ToolCall `json:"toolCalls,omitempty"`
	TokenUsage   TokenUsage `json:"tokenUsage"`
	FinishReason string     `json:"finishReason"`
	ModelUsed    string     `json:"modelUsed,omitempty"`
}

// ToolDefinition describes a tool available for the LLM to call.
type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description"`
	Parameters  map[string]any `json:"parameters,omitempty"`
}

// SafetySetting configures content safety filtering for a specific harm category.
type SafetySetting struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// ImageRequest contains parameters for an image generation API call.
type ImageRequest struct {
	Provider string         `json:"provider,omitempty"`
	Model    string         `json:"model"`
	Prompt   string         `json:"prompt"`
	Images   []Image        `json:"images,omitempty"`   // input images for editing
	Params   map[string]any `json:"params,omitempty"`
}

// ParamString extracts a string value from Params by key, with a default.
func (r ImageRequest) ParamString(key string, defaultVal string) string {
	v, ok := r.Params[key]
	if !ok {
		return defaultVal
	}
	if s, ok := v.(string); ok {
		return s
	}
	return defaultVal
}

// ParamFloat32 extracts a float32 value from Params by key, returning nil if absent.
func (r ImageRequest) ParamFloat32(key string) *float32 {
	v, ok := r.Params[key]
	if !ok {
		return nil
	}
	var f float32
	switch n := v.(type) {
	case float64:
		f = float32(n)
	case float32:
		f = n
	case int:
		f = float32(n)
	case json.Number:
		ff, err := n.Float64()
		if err != nil {
			return nil
		}
		f = float32(ff)
	default:
		return nil
	}
	return &f
}

// ParamInt32 extracts an int32 value from Params by key, returning nil if absent.
func (r ImageRequest) ParamInt32(key string) *int32 {
	v, ok := r.Params[key]
	if !ok {
		return nil
	}
	var i int32
	switch n := v.(type) {
	case float64:
		i = int32(n)
	case int:
		i = int32(n)
	case json.Number:
		ii, err := n.Int64()
		if err != nil {
			return nil
		}
		i = int32(ii)
	default:
		return nil
	}
	return &i
}

// ParamBool extracts a bool value from Params by key, with a default.
func (r ImageRequest) ParamBool(key string, defaultVal bool) bool {
	v, ok := r.Params[key]
	if !ok {
		return defaultVal
	}
	if b, ok := v.(bool); ok {
		return b
	}
	return defaultVal
}

// ImageResponse represents the result of an image generation request.
type ImageResponse struct {
	Images     []GeneratedImage `json:"images"`
	Text       string           `json:"text,omitempty"`
	TokenUsage TokenUsage       `json:"tokenUsage"`
	ModelUsed  string           `json:"modelUsed,omitempty"`
}

// GeneratedImage represents a single generated image.
type GeneratedImage struct {
	Data     []byte `json:"data"`
	MimeType string `json:"mimeType"`
}

// ConversationRequest contains parameters for a conversation API call.
type ConversationRequest struct {
	Messages              []Message        `json:"messages"`
	SystemPrompt          string           `json:"systemPrompt,omitempty"`
	CacheableSystemPrompt string           `json:"cacheableSystemPrompt,omitempty"` // static system prompt cached separately (Claude only)
	Model                 string           `json:"model"`
	Provider     string           `json:"provider,omitempty"` // provider name for routing
	Params       map[string]any   `json:"params,omitempty"`   // model parameters (temperature, max_tokens, etc.)
	Tools        []ToolDefinition `json:"tools,omitempty"`
	ForceTool    bool             `json:"forceTool,omitempty"` // force model to always call a tool (ANY mode)
}

// ParamFloat64 extracts a float64 value from Params by key, with a default.
func (r ConversationRequest) ParamFloat64(key string, defaultVal float64) float64 {
	v, ok := r.Params[key]
	if !ok {
		return defaultVal
	}
	switch n := v.(type) {
	case float64:
		return n
	case int:
		return float64(n)
	case json.Number:
		f, err := n.Float64()
		if err == nil {
			return f
		}
	}
	return defaultVal
}

// ParamInt extracts an int value from Params by key, with a default.
func (r ConversationRequest) ParamInt(key string, defaultVal int) int {
	v, ok := r.Params[key]
	if !ok {
		return defaultVal
	}
	switch n := v.(type) {
	case float64:
		return int(n)
	case int:
		return n
	case json.Number:
		i, err := n.Int64()
		if err == nil {
			return int(i)
		}
	}
	return defaultVal
}

// ParamString extracts a string value from Params by key, with a default.
func (r ConversationRequest) ParamString(key string, defaultVal string) string {
	v, ok := r.Params[key]
	if !ok {
		return defaultVal
	}
	if s, ok := v.(string); ok {
		return s
	}
	return defaultVal
}

// ParamBool extracts a bool value from Params by key, with a default.
func (r ConversationRequest) ParamBool(key string, defaultVal bool) bool {
	v, ok := r.Params[key]
	if !ok {
		return defaultVal
	}
	if b, ok := v.(bool); ok {
		return b
	}
	return defaultVal
}
