//go:build unit

package service

import (
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestAccount_SimulateCache_Getters(t *testing.T) {
	tests := []struct {
		name             string
		extra            map[string]any
		enabled          bool
		minPct           float64
		maxPct           float64
		expectRateZero   bool
	}{
		{
			name:           "disabled when extra nil",
			extra:          nil,
			enabled:        false,
			expectRateZero: true,
		},
		{
			name:           "disabled when flag false",
			extra:          map[string]any{"simulate_cache_enabled": false},
			enabled:        false,
			expectRateZero: true,
		},
		{
			name:           "enabled with zero max returns zero rate",
			extra:          map[string]any{"simulate_cache_enabled": true, "simulate_cache_min_percent": float64(0), "simulate_cache_max_percent": float64(0)},
			enabled:        true,
			expectRateZero: true,
		},
		{
			name:   "enabled with 50-50 returns 0.5 rate",
			extra:  map[string]any{"simulate_cache_enabled": true, "simulate_cache_min_percent": float64(50), "simulate_cache_max_percent": float64(50)},
			enabled: true,
			minPct: 50,
			maxPct: 50,
		},
		{
			name:   "enabled with 20-80 returns rate in [0.2, 0.8]",
			extra:  map[string]any{"simulate_cache_enabled": true, "simulate_cache_min_percent": float64(20), "simulate_cache_max_percent": float64(80)},
			enabled: true,
			minPct: 20,
			maxPct: 80,
		},
		{
			name:   "min > max clamped to max",
			extra:  map[string]any{"simulate_cache_enabled": true, "simulate_cache_min_percent": float64(90), "simulate_cache_max_percent": float64(30)},
			enabled: true,
			minPct: 90,
			maxPct: 30,
		},
		{
			name:   "negative percent clamped to 0",
			extra:  map[string]any{"simulate_cache_enabled": true, "simulate_cache_min_percent": float64(-10), "simulate_cache_max_percent": float64(40)},
			enabled: true,
			minPct: 0,
			maxPct: 40,
		},
		{
			name:   "over 100 clamped to 100",
			extra:  map[string]any{"simulate_cache_enabled": true, "simulate_cache_min_percent": float64(120), "simulate_cache_max_percent": float64(150)},
			enabled: true,
			minPct: 100,
			maxPct: 100,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			a := &Account{Extra: tt.extra}
			if got := a.IsSimulateCacheEnabled(); got != tt.enabled {
				t.Errorf("IsSimulateCacheEnabled = %v, want %v", got, tt.enabled)
			}
			if !tt.enabled {
				return
			}
			minPct := a.GetSimulateCacheMinPercent()
			maxPct := a.GetSimulateCacheMaxPercent()
			if minPct != tt.minPct {
				t.Errorf("GetSimulateCacheMinPercent = %v, want %v", minPct, tt.minPct)
			}
			if maxPct != tt.maxPct {
				t.Errorf("GetSimulateCacheMaxPercent = %v, want %v", maxPct, tt.maxPct)
			}
			rate := a.ResolveSimulateCacheRate()
			if tt.expectRateZero {
				if rate != 0 {
					t.Errorf("ResolveSimulateCacheRate = %v, want 0", rate)
				}
				return
			}
			if rate <= 0 {
				t.Errorf("ResolveSimulateCacheRate = %v, want > 0", rate)
			}
			lo := tt.minPct / 100.0
			hi := tt.maxPct / 100.0
			// 当 min>max 时 ResolveSimulateCacheRate 内部把 min 钳为 max，所以上界取 max
			if tt.minPct > tt.maxPct {
				lo = tt.maxPct / 100.0
			}
			if rate < lo-1e-9 || rate > hi+1e-9 {
				t.Errorf("ResolveSimulateCacheRate = %v, want in [%v, %v]", rate, lo, hi)
			}
		})
	}
}

func TestAccount_ResolveSimulateCacheRate_Distribution(t *testing.T) {
	a := &Account{Extra: map[string]any{
		"simulate_cache_enabled":      true,
		"simulate_cache_min_percent":  float64(30),
		"simulate_cache_max_percent":  float64(70),
	}}
	seen := map[float64]int{}
	for i := 0; i < 200; i++ {
		r := a.ResolveSimulateCacheRate()
		if r < 0.3-1e-9 || r > 0.7+1e-9 {
			t.Fatalf("rate %v out of [0.3, 0.7]", r)
		}
		seen[r]++
	}
	if len(seen) < 2 {
		t.Errorf("expected multiple distinct rates, got %d", len(seen))
	}
}

func TestApplySimulateCacheToUsageObj(t *testing.T) {
	tests := []struct {
		name        string
		usage       map[string]any
		rate        float64
		wantChanged bool
		wantInput   float64
		wantCached  float64
	}{
		{
			name:        "nil usage no change",
			usage:       nil,
			rate:        0.5,
			wantChanged: false,
		},
		{
			name:        "zero rate no change",
			usage:       map[string]any{"input_tokens": float64(100), "cache_read_input_tokens": float64(0)},
			rate:        0,
			wantChanged: false,
		},
		{
			name:        "zero total no change",
			usage:       map[string]any{"input_tokens": float64(0), "cache_read_input_tokens": float64(0)},
			rate:        0.5,
			wantChanged: false,
		},
		{
			name:        "split 1000 at 0.5",
			usage:       map[string]any{"input_tokens": float64(1000), "cache_read_input_tokens": float64(0)},
			rate:        0.5,
			wantChanged: true,
			wantInput:   500,
			wantCached:  500,
		},
		{
			name:        "override existing cache_read",
			usage:       map[string]any{"input_tokens": float64(800), "cache_read_input_tokens": float64(200)},
			rate:        0.5,
			wantChanged: true,
			wantInput:   500,
			wantCached:  500,
		},
		{
			name:        "rate 0.3 on 1000",
			usage:       map[string]any{"input_tokens": float64(1000), "cache_read_input_tokens": float64(0)},
			rate:        0.3,
			wantChanged: true,
			wantInput:   700,
			wantCached:  300,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			changed := applySimulateCacheToUsageObj(tt.usage, tt.rate)
			if changed != tt.wantChanged {
				t.Errorf("changed = %v, want %v", changed, tt.wantChanged)
			}
			if !tt.wantChanged {
				return
			}
			if got := tt.usage["input_tokens"].(float64); got != tt.wantInput {
				t.Errorf("input_tokens = %v, want %v", got, tt.wantInput)
			}
			if got := tt.usage["cache_read_input_tokens"].(float64); got != tt.wantCached {
				t.Errorf("cache_read_input_tokens = %v, want %v", got, tt.wantCached)
			}
		})
	}
}

func TestApplySimulateCacheToOpenAIUsageKeepsInputTotal(t *testing.T) {
	usage := &OpenAIUsage{InputTokens: 1000, CacheReadInputTokens: 25, OutputTokens: 20}
	account := &Account{Extra: map[string]any{
		"simulate_cache_enabled":     true,
		"simulate_cache_min_percent": float64(40),
		"simulate_cache_max_percent": float64(40),
	}}

	if changed := applySimulateCacheToOpenAIUsage(usage, account); !changed {
		t.Fatal("expected simulated cache to change usage")
	}
	if usage.InputTokens != 1000 {
		t.Fatalf("InputTokens = %d, want unchanged 1000", usage.InputTokens)
	}
	if usage.CacheReadInputTokens != 400 {
		t.Fatalf("CacheReadInputTokens = %d, want 400", usage.CacheReadInputTokens)
	}
}

func TestApplySimulateCacheToOpenAISSEBodyUpdatesTerminalUsage(t *testing.T) {
	body := strings.Join([]string{
		`data: {"type":"response.in_progress","response":{"id":"resp_1"}}`,
		`data: {"type":"response.completed","response":{"id":"resp_1","usage":{"input_tokens":100,"output_tokens":7}}}`,
		`data: [DONE]`,
	}, "\n")

	updated, changed := applySimulateCacheToOpenAISSEBody(body, 0.25)
	if !changed {
		t.Fatal("expected SSE body to be updated")
	}
	_, terminal, ok := extractOpenAISSETerminalEvent(updated)
	if !ok {
		t.Fatal("expected terminal event after update")
	}
	if got := gjson.GetBytes(terminal, "response.usage.input_tokens").Int(); got != 100 {
		t.Fatalf("input_tokens = %d, want unchanged 100", got)
	}
	if got := gjson.GetBytes(terminal, "response.usage.cache_read_input_tokens").Int(); got != 25 {
		t.Fatalf("cache_read_input_tokens = %d, want 25", got)
	}
	if got := gjson.GetBytes(terminal, "response.usage.input_tokens_details.cached_tokens").Int(); got != 25 {
		t.Fatalf("input_tokens_details.cached_tokens = %d, want 25", got)
	}
}
