package llm

import (
	"context"
	"time"
)

// withRetry executes fn with linear backoff retry. It returns immediately
// if fn succeeds or the context is cancelled.
func withRetry[T any](ctx context.Context, maxAttempts int, fn func() (T, error)) (T, error) {
	retryDelay := time.Second
	var zero T
	var lastErr error

	for attempt := 1; attempt <= maxAttempts; attempt++ {
		result, err := fn()
		if err == nil {
			return result, nil
		}
		lastErr = err
		if ctx.Err() != nil {
			return zero, lastErr
		}
		if attempt < maxAttempts {
			time.Sleep(retryDelay)
			retryDelay = min(retryDelay+time.Second, 3*time.Second)
		}
	}

	return zero, lastErr
}
