package llm

import (
	"regexp"
	"strings"
)

// overflowPatterns matches common context overflow error messages from various providers.
var overflowPatterns = []*regexp.Regexp{
	// Gemini / Vertex AI
	regexp.MustCompile(`(?i)exceeds? the maximum number of tokens`),
	regexp.MustCompile(`(?i)request too large`),
	regexp.MustCompile(`(?i)content too long`),
	regexp.MustCompile(`(?i)input token count.*exceeds the maximum`),

	// Claude / Anthropic
	regexp.MustCompile(`(?i)prompt is too long`),
	regexp.MustCompile(`(?i)exceeds? the maximum allowed`),

	// OpenAI / compatible
	regexp.MustCompile(`(?i)maximum context length`),
	regexp.MustCompile(`(?i)exceeds the context window`),
	regexp.MustCompile(`(?i)context_length_exceeded`),
	regexp.MustCompile(`(?i)too many tokens`),
	regexp.MustCompile(`(?i)token limit exceeded`),

	// Groq
	regexp.MustCompile(`(?i)reduce the length of the messages`),

	// Mistral
	regexp.MustCompile(`(?i)too large for model with \d+ maximum context length`),

	// Ollama / llama.cpp
	regexp.MustCompile(`(?i)prompt too long; exceeded (?:max )?context length`),
	regexp.MustCompile(`(?i)greater than the context length`),
	regexp.MustCompile(`(?i)exceeds the available context size`),

	// OpenRouter / generic
	regexp.MustCompile(`(?i)maximum context length is \d+ tokens`),
	regexp.MustCompile(`(?i)context[_ ]window[_ ]exceeds? limit`),
}

// IsContextOverflow checks whether an error indicates a context overflow condition.
func IsContextOverflow(err error) bool {
	if err == nil {
		return false
	}
	msg := err.Error()
	for _, p := range overflowPatterns {
		if p.MatchString(msg) {
			return true
		}
	}
	return false
}

// mapFinishReason normalizes a provider-specific finish reason to a standard constant.
// If the error indicates context overflow, it returns FinishContextOverflow.
func mapFinishReason(reason string, err error) string {
	if err != nil && IsContextOverflow(err) {
		return FinishContextOverflow
	}

	r := strings.ToLower(reason)
	switch {
	// stop
	case r == "stop" || r == "end_turn" || r == "stop_sequence":
		return FinishStop
	// length
	case r == "length" || r == "max_tokens":
		return FinishLength
	// tool calls
	case r == "tool_calls" || r == "tool_use":
		return FinishToolCalls
	// safety
	case r == "safety" || r == "recitation" || r == "blocklist" ||
		r == "prohibited_content" || r == "spii" ||
		r == "content_filter":
		return FinishSafety
	default:
		if reason == "" {
			return FinishStop
		}
		return reason
	}
}
