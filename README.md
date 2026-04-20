# go-llm

统一的 Go 语言 LLM 抽象层，通过一致的接口对接多个大语言模型供应商。支持流式输出、工具调用、提示缓存、多模态输入和扩展思考等能力。

## 支持的供应商

| 供应商 | 直连 API | Vertex AI |
|--------|---------|-----------|
| Anthropic Claude | `ProviderClaude` | `ProviderClaudeVertex` |
| Google Gemini | `ProviderGemini` | `ProviderGeminiVertex` |
| 通义千问 (Qwen) | `ProviderQwen` | - |
| Kimi (月之暗面) | `ProviderKimiBailian` | - |

## 安装

```bash
go get github.com/hellof20/go-llm
```

## 快速开始

### 基本对话

```go
package main

import (
    "context"
    "fmt"

    llm "github.com/hellof20/go-llm"
)

func main() {
    provider, err := llm.NewClaudeVertex("my-project", "us-east5", 3)
    if err != nil {
        panic(err)
    }

    req := llm.ConversationRequest{
        Model:        "claude-sonnet-4-20250514",
        SystemPrompt: "You are a helpful assistant.",
        Messages: []llm.Message{
            {Role: "user", Content: "Hello!"},
        },
        Params: map[string]any{
            "max_tokens": 1024,
        },
    }

    resp, err := provider.Chat(context.Background(), req)
    if err != nil {
        panic(err)
    }
    fmt.Println(resp.Content)
    fmt.Printf("Tokens: input=%d, output=%d\n", resp.Usage.InputTokens, resp.Usage.OutputTokens)
}
```

### 流式输出

```go
ch, err := provider.ChatStream(ctx, req)
if err != nil {
    panic(err)
}

for event := range ch {
    switch event.Type {
    case llm.EventTextDelta:
        fmt.Print(event.Content)
    case llm.EventThinkingDelta:
        fmt.Print(event.Content) // 扩展思考内容
    case llm.EventToolCallEnd:
        fmt.Printf("Tool call: %s(%s)\n", event.ToolCall.Name, event.ToolCall.Arguments)
    case llm.EventDone:
        fmt.Printf("\nTokens: %+v\n", event.Usage)
    case llm.EventError:
        fmt.Printf("Error: %v\n", event.Error)
    }
}
```

也可以使用 `Collect` 将流式结果收集为完整响应：

```go
ch, _ := provider.ChatStream(ctx, req)
resp, err := llm.Collect(ch)
```

### 多供应商路由

使用 `ProviderRegistry` 按请求路由到不同供应商：

```go
registry := llm.NewProviderRegistry()

claude, _ := llm.NewClaudeVertex("project", "us-east5", 3)
gemini, _ := llm.NewGeminiVertex("", "project", "us-central1", 3)

registry.Register("claude-vertex", claude)
registry.Register("gemini-vertex", gemini)
registry.SetDefault("claude-vertex")

req := llm.ConversationRequest{
    Provider: "gemini-vertex", // 指定路由到 Gemini
    Model:    "gemini-2.5-flash",
    Messages: []llm.Message{{Role: "user", Content: "Hi"}},
}
resp, err := registry.Chat(ctx, req)
```

## 创建供应商

```go
// Anthropic Claude - 直连 API
provider, err := llm.NewClaude(apiKey, retryTimes)

// Anthropic Claude - Vertex AI
provider, err := llm.NewClaudeVertex(project, location, retryTimes)

// Google Gemini - 直连 API
provider, err := llm.NewGemini(apiKey, retryTimes)

// Google Gemini - Vertex AI（使用 ADC 认证）
provider, err := llm.NewGeminiVertex("", project, location, retryTimes)

// Google Gemini - Vertex AI（使用 API Key，Express 模式）
provider, err := llm.NewGeminiVertex(apiKey, project, location, retryTimes)

// 通义千问
provider, err := llm.NewOpenAICompat(apiKey, "https://dashscope.aliyuncs.com/compatible-mode/v1", retryTimes)

// Kimi（百炼）
provider, err := llm.NewOpenAICompat(apiKey, "https://dashscope.aliyuncs.com/compatible-mode/v1", retryTimes)
```

## 核心类型

### ConversationRequest

```go
type ConversationRequest struct {
    Provider             string
    Model                string
    SystemPrompt         string
    CacheableSystemPrompt string  // 可缓存的系统提示（细粒度缓存控制）
    Messages             []Message
    Tools                []ToolDefinition
    ForceTool            bool
    Params               map[string]any
}
```

### 请求参数 (Params)

| 参数 | 类型 | 说明 |
|------|------|------|
| `max_tokens` / `max_output_tokens` | int | 最大输出 token 数 |
| `temperature` | float | 采样温度 |
| `top_p` | float | Nucleus 采样 |
| `top_k` | int | Top-K 采样 |
| `thinking_level` | string | 扩展思考级别：`low`/`medium`/`high`/`max`/`disabled` |
| `enable_thinking` | bool | 启用思考输出 |
| `enable_cache` | bool | 启用提示缓存（OpenAI 兼容供应商） |

### LLMResponse

```go
type LLMResponse struct {
    Content      string
    Thinking     string
    ToolCalls    []ToolCall
    Usage        TokenUsage
    FinishReason string
    ModelUsed    string
}
```

### TokenUsage

```go
type TokenUsage struct {
    InputTokens      int
    OutputTokens     int
    ThinkingTokens   int
    CacheReadTokens  int
    CacheWriteTokens int
}
```

## 功能特性

### 工具调用

```go
tools := []llm.ToolDefinition{
    {
        Name:        "get_weather",
        Description: "Get current weather for a location",
        Parameters: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "location": map[string]any{"type": "string"},
            },
            "required": []string{"location"},
        },
    },
}

req := llm.ConversationRequest{
    Model:    "claude-sonnet-4-20250514",
    Messages: []llm.Message{{Role: "user", Content: "What's the weather in Beijing?"}},
    Tools:    tools,
}
```

### 多模态输入

消息支持图片和文档附件：

```go
msg := llm.Message{
    Role:    "user",
    Content: "Describe this image",
    Images: []llm.ImageData{
        {MIMEType: "image/png", Base64Data: base64Str},
    },
    Documents: []llm.DocumentData{
        {MIMEType: "application/pdf", Base64Data: base64Str, Filename: "doc.pdf"},
    },
}
```

### 提示缓存

Claude 和 OpenAI 兼容供应商支持提示缓存以降低成本：

```go
req := llm.ConversationRequest{
    SystemPrompt:          "Brief instructions",
    CacheableSystemPrompt: "Long reference content that should be cached...",
    Params: map[string]any{
        "enable_cache": true,
    },
}
```

缓存命中信息通过 `TokenUsage.CacheReadTokens` 和 `CacheWriteTokens` 追踪。

### 上下文溢出检测

```go
resp, err := provider.Chat(ctx, req)
if llm.IsContextOverflow(err) {
    // 处理上下文窗口超出
}
```

### 日志注入

所有供应商支持通过 `SetLogger` 注入自定义 logger：

```go
logger := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: slog.LevelDebug}))
provider.(interface{ SetLogger(*slog.Logger) }).SetLogger(logger)
```

## License

MIT
