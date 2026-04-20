package llm_test

import (
	"os"
	"testing"
)

func getEnv(t *testing.T, key string, defaultVal ...string) string {
	t.Helper()
	if v := os.Getenv(key); v != "" {
		return v
	}
	if len(defaultVal) > 0 {
		return defaultVal[0]
	}
	t.Skipf("%s not set", key)
	return ""
}
