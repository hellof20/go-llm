package llm

import "strings"

// Collect reads all events from a StreamEvent channel and assembles them
// into a complete LLMResponse. This is a convenience wrapper for callers
// who don't need real-time streaming and just want the final result.
//
// It returns a non-nil error only if an EventError is received.
func Collect(ch <-chan StreamEvent) (*LLMResponse, error) {
	resp := &LLMResponse{}

	var textParts []string
	var thinkingParts []string
	var currentToolCall *ToolCall

	for event := range ch {
		switch event.Type {
		case EventTextDelta:
			textParts = append(textParts, event.Content)

		case EventThinkingDelta:
			thinkingParts = append(thinkingParts, event.Content)

		case EventToolCallStart:
			if event.ToolCall != nil {
				currentToolCall = &ToolCall{
					ID:   event.ToolCall.ID,
					Name: event.ToolCall.Name,
				}
			}

		case EventToolCallEnd:
			if event.ToolCall != nil {
				resp.ToolCalls = append(resp.ToolCalls, *event.ToolCall)
				currentToolCall = nil
			} else if currentToolCall != nil {
				resp.ToolCalls = append(resp.ToolCalls, *currentToolCall)
				currentToolCall = nil
			}

		case EventDone:
			resp.FinishReason = event.FinishReason
			if event.Usage != nil {
				resp.TokenUsage = *event.Usage
			}

		case EventError:
			return resp, event.Error
		}
	}

	resp.Content = strings.Join(textParts, "")
	resp.Thinking = strings.Join(thinkingParts, "")

	return resp, nil
}
