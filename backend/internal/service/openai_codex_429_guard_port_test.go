package service

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
)

func requireCodexGuardTestSlice(t *testing.T, value any) []any {
	t.Helper()
	result, ok := value.([]any)
	require.True(t, ok)
	return result
}

func requireCodexGuardTestMap(t *testing.T, value any) map[string]any {
	t.Helper()
	result, ok := value.(map[string]any)
	require.True(t, ok)
	return result
}

func requireCodexGuardTestString(t *testing.T, value any) string {
	t.Helper()
	result, ok := value.(string)
	require.True(t, ok)
	return result
}

func TestApplyCodexOAuthTransform_AppendsSyntheticAgentContextPair(t *testing.T) {
	reqBody := map[string]any{
		"model": "gpt-5.4",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "continue the task"},
		},
		"tools": []any{
			map[string]any{"type": "function", "name": "shell", "parameters": map[string]any{"type": "object"}},
		},
	}

	applyCodexOAuthTransformWithOptions(reqBody, codexOAuthTransformOptions{Codex429GuardEnabled: true})

	input, ok := reqBody["input"].([]any)
	require.True(t, ok)
	require.Len(t, input, 3)
	call, ok := input[1].(map[string]any)
	require.True(t, ok)
	output, ok := input[2].(map[string]any)
	require.True(t, ok)
	require.Equal(t, "custom_tool_call", call["type"])
	require.Equal(t, codexSyntheticAgentContextToolName, call["name"])
	require.Equal(t, codexSyntheticAgentContextInput, call["input"])
	require.Equal(t, "custom_tool_call_output", output["type"])
	require.Equal(t, []map[string]string{{"type": "input_text", "text": codexSyntheticAgentContextOutputText}}, output["output"])
	require.Equal(t, call["call_id"], output["call_id"])
	require.LessOrEqual(t, len(requireCodexGuardTestString(t, call["call_id"])), codexCallIDMaxLength)

	tools, ok := reqBody["tools"].([]any)
	require.True(t, ok)
	require.Len(t, tools, 1, "synthetic checkpoint is history-only and must not be declared as an available tool")
	require.Equal(t, "shell", requireCodexGuardTestMap(t, tools[0])["name"])
	require.False(t, HasFunctionCallOutput(reqBody), "synthetic output must not masquerade as a real continuation")
}

func TestAppendCodexSyntheticAgentContextPair_UsesRandomIDsAndIsIdempotent(t *testing.T) {
	build := func() map[string]any {
		return map[string]any{
			"input": []any{map[string]any{"type": "message", "role": "user", "content": "same history"}},
		}
	}
	first := build()
	second := build()
	require.True(t, appendCodexSyntheticAgentContextPair(first))
	require.True(t, appendCodexSyntheticAgentContextPair(second))

	firstInput := requireCodexGuardTestSlice(t, first["input"])
	secondInput := requireCodexGuardTestSlice(t, second["input"])
	firstID := requireCodexGuardTestMap(t, firstInput[1])["call_id"]
	require.NotEqual(t, firstID, requireCodexGuardTestMap(t, secondInput[1])["call_id"])
	require.Contains(t, requireCodexGuardTestString(t, firstID), "call_sub2api_overdraft_")
	require.False(t, appendCodexSyntheticAgentContextPair(first))
	require.Len(t, requireCodexGuardTestSlice(t, first["input"]), 3)
}

func TestSyntheticAgentContextPairRecognizesNormalizedFCID(t *testing.T) {
	callID := "fc_sub2api_overdraft_normalized"
	call := map[string]any{
		"type": "custom_tool_call", "call_id": callID,
		"name": codexSyntheticAgentContextToolName, "input": codexSyntheticAgentContextInput,
	}
	output := map[string]any{
		"type": "custom_tool_call_output", "call_id": callID,
		"output": []map[string]string{{"type": "input_text", "text": codexSyntheticAgentContextOutputText}},
	}
	require.True(t, isCodexSyntheticAgentContextCall(call))
	require.True(t, isCodexSyntheticAgentContextOutput(output))
	require.True(t, isCodexSyntheticAgentContextCallID(callID))
	require.False(t, appendCodexSyntheticAgentContextPair(map[string]any{
		"input": []any{map[string]any{"type": "message", "role": "user"}, call, output},
	}))
}

func TestAppendCodexSyntheticAgentContextPair_DoesNotReinjectEarlierHistoryPair(t *testing.T) {
	reqBody := map[string]any{
		"input": []any{map[string]any{"type": "message", "role": "user", "content": "first"}},
	}
	require.True(t, appendCodexSyntheticAgentContextPair(reqBody))
	items := requireCodexGuardTestSlice(t, reqBody["input"])
	items = append(items, map[string]any{"type": "message", "role": "user", "content": "follow-up"})
	reqBody["input"] = items

	require.False(t, appendCodexSyntheticAgentContextPair(reqBody))
	require.Len(t, requireCodexGuardTestSlice(t, reqBody["input"]), 4)
}

func TestAppendCodexSyntheticAgentContextPairToBodyMatchesReferenceGuards(t *testing.T) {
	body := []byte(`{"model":"gpt-5.4","input":[{"type":"message","role":"user","content":"hello"}],"metadata":{"keep":true}}`)
	updated, changed, err := appendCodexSyntheticAgentContextPairToBody(body)
	require.NoError(t, err)
	require.True(t, changed)
	require.Contains(t, string(updated), `"type":"custom_tool_call"`)
	require.Contains(t, string(updated), `call_sub2api_overdraft_`)
	require.Contains(t, string(updated), `"keep":true`)

	assistantTail := []byte(`{"input":[{"type":"message","role":"assistant","content":"answer"}]}`)
	unchanged, changed, err := appendCodexSyntheticAgentContextPairToBody(assistantTail)
	require.NoError(t, err)
	require.True(t, changed)
	require.NotEqual(t, string(assistantTail), string(unchanged))
	require.Contains(t, string(unchanged), `"type":"custom_tool_call"`)

	invalid := []byte(`{"input":[}`)
	unchanged, changed, err = appendCodexSyntheticAgentContextPairToBody(invalid)
	require.NoError(t, err)
	require.False(t, changed)
	require.Equal(t, string(invalid), string(unchanged))
}

func TestAppendCodexSyntheticAgentContextPairToBodyNormalizesInputShapes(t *testing.T) {
	tests := []struct {
		name  string
		input string
	}{
		{name: "array", input: `[{"type":"message","role":"user","content":"hello"}]`},
		{name: "object", input: `{"type":"message","role":"user","content":"hello"}`},
		{name: "string", input: `"hello"`},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			body := []byte(`{"model":"gpt-5.4","input":` + tt.input + `}`)
			updated, changed, err := appendCodexSyntheticAgentContextPairToBody(body)
			require.NoError(t, err)
			require.True(t, changed)

			var document struct {
				Input []map[string]any `json:"input"`
			}
			require.NoError(t, json.Unmarshal(updated, &document))
			require.Len(t, document.Input, 3)
			require.Equal(t, "message", document.Input[0]["type"])
			require.Equal(t, "user", document.Input[0]["role"])
			require.Equal(t, "custom_tool_call", document.Input[1]["type"])
			require.Equal(t, "custom_tool_call_output", document.Input[2]["type"])
			require.Equal(t, document.Input[1]["call_id"], document.Input[2]["call_id"])
		})
	}
}

func TestApplyCodexOAuthTransform_SyntheticAgentContextPairSkipsToolHistoryAndCompact(t *testing.T) {
	for _, itemType := range []string{
		"function_call",
		"function_call_output",
		"local_shell_call",
		"custom_tool_call_output",
		"mcp_tool_call_output",
		"image_generation_call",
	} {
		t.Run(itemType, func(t *testing.T) {
			reqBody := map[string]any{
				"model": "gpt-5.4",
				"input": []any{map[string]any{"type": itemType, "call_id": "call_real", "name": "real_tool"}},
			}
			applyCodexOAuthTransformWithOptions(reqBody, codexOAuthTransformOptions{Codex429GuardEnabled: true})
			require.Len(t, requireCodexGuardTestSlice(t, reqBody["input"]), 1)
		})
	}

	compact := map[string]any{
		"model": "gpt-5.4",
		"input": []any{map[string]any{"type": "message", "role": "user", "content": "compact"}},
	}
	applyCodexOAuthTransformWithOptions(compact, codexOAuthTransformOptions{IsCompact: true, Codex429GuardEnabled: true})
	require.Len(t, requireCodexGuardTestSlice(t, compact["input"]), 1)
}

func TestApplyCodexOAuthTransform_SyntheticAgentContextPairSupportsStringAndObjectInput(t *testing.T) {
	for _, input := range []any{
		"hello",
		map[string]any{"type": "message", "role": "user", "content": "hello"},
	} {
		reqBody := map[string]any{"model": "gpt-5.4", "input": input}
		applyCodexOAuthTransformWithOptions(reqBody, codexOAuthTransformOptions{Codex429GuardEnabled: true})
		items := requireCodexGuardTestSlice(t, reqBody["input"])
		require.Len(t, items, 3)
		require.Equal(t, "custom_tool_call", requireCodexGuardTestMap(t, items[1])["type"])
		require.Equal(t, "custom_tool_call_output", requireCodexGuardTestMap(t, items[2])["type"])
	}
}

func TestApplyCodexOAuthTransform_SyntheticAgentContextPairAllowsOrdinaryMessageTails(t *testing.T) {
	reqBody := map[string]any{
		"model": "gpt-5.4",
		"input": []any{
			map[string]any{"type": "message", "role": "user", "content": "first"},
			map[string]any{"type": "message", "role": "assistant", "content": "last"},
		},
	}

	applyCodexOAuthTransformWithOptions(reqBody, codexOAuthTransformOptions{Codex429GuardEnabled: true})
	items := requireCodexGuardTestSlice(t, reqBody["input"])
	require.Len(t, items, 4)
	require.Equal(t, "custom_tool_call", requireCodexGuardTestMap(t, items[2])["type"])
	require.Equal(t, "custom_tool_call_output", requireCodexGuardTestMap(t, items[3])["type"])
}

func TestApplyCodexOAuthTransform_SyntheticAgentContextPairSkipsClaudeCodeBridge(t *testing.T) {
	reqBody := map[string]any{
		"model":            "gpt-5.4",
		"prompt_cache_key": "anthropic-metadata-claude-code-session",
		"input": []any{map[string]any{
			"type":    "message",
			"role":    "user",
			"content": openAICompatClaudeCodeTodoGuardMarker,
		}},
	}

	applyCodexOAuthTransformWithOptions(reqBody, codexOAuthTransformOptions{Codex429GuardEnabled: true})
	require.Len(t, requireCodexGuardTestSlice(t, reqBody["input"]), 1)
}

func TestApplyCodexOAuthTransform_Codex429GuardDisabledDoesNotAppend(t *testing.T) {
	reqBody := map[string]any{
		"model": "gpt-5.4",
		"input": []any{map[string]any{"type": "message", "role": "user", "content": "hello"}},
	}
	applyCodexOAuthTransformWithOptions(reqBody, codexOAuthTransformOptions{Codex429GuardEnabled: false})
	require.Len(t, requireCodexGuardTestSlice(t, reqBody["input"]), 1)
}

func TestAccountCodex429GuardEnabled(t *testing.T) {
	tests := []struct {
		name    string
		account *Account
		want    bool
	}{
		{name: "nil", account: nil, want: false},
		{name: "claude oauth", account: &Account{Platform: PlatformAnthropic, Type: AccountTypeOAuth}, want: false},
		{name: "openai api key", account: &Account{Platform: PlatformOpenAI, Type: AccountTypeAPIKey}, want: false},
		{name: "openai oauth missing setting is opt-in off", account: &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth}, want: false},
		{name: "openai oauth shadow excluded", account: &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth, ParentAccountID: int64PtrForCodexGuardTest(1)}, want: false},
		{name: "openai oauth spark dimension excluded", account: &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth, QuotaDimension: QuotaDimensionSpark}, want: false},
		{name: "enabled", account: &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Extra: map[string]any{OpenAICodex429GuardEnabledExtraKey: true}}, want: true},
		{name: "disabled", account: &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Extra: map[string]any{OpenAICodex429GuardEnabledExtraKey: false}}, want: false},
		{name: "string disabled", account: &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Extra: map[string]any{OpenAICodex429GuardEnabledExtraKey: "false"}}, want: false},
		{name: "invalid string fails closed", account: &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Extra: map[string]any{OpenAICodex429GuardEnabledExtraKey: "unexpected"}}, want: false},
		{name: "nil setting is opt-in off", account: &Account{Platform: PlatformOpenAI, Type: AccountTypeOAuth, Extra: map[string]any{OpenAICodex429GuardEnabledExtraKey: nil}}, want: false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.account.Codex429GuardEnabled(); got != tt.want {
				t.Fatalf("Codex429GuardEnabled() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestConfirmOpenAIOAuth429RequiresTwoObservations(t *testing.T) {
	svc := &RateLimitService{}
	accountID := int64(1001)
	now := time.Now()

	require.False(t, svc.confirmOpenAIOAuth429Context(context.Background(), accountID, now), "first 429 must not confirm cooldown")
	require.True(t, svc.confirmOpenAIOAuth429Context(context.Background(), accountID, now.Add(time.Second)), "second 429 within window confirms cooldown")
}

func TestConfirmOpenAIOAuth429WindowExpires(t *testing.T) {
	svc := &RateLimitService{}
	accountID := int64(1002)
	now := time.Now()

	require.False(t, svc.confirmOpenAIOAuth429Context(context.Background(), accountID, now))
	require.False(t, svc.confirmOpenAIOAuth429Context(context.Background(), accountID, now.Add(openAIOAuth429ConfirmationWindow+time.Minute)),
		"stale streak must not confirm cooldown after the window expires")
	require.False(t, svc.confirmOpenAIOAuth429Context(context.Background(), accountID, now.Add(openAIOAuth429ConfirmationWindow+2*time.Minute)))
}

func TestClearOpenAIOAuth429StreakResetsCount(t *testing.T) {
	svc := &RateLimitService{}
	accountID := int64(1003)
	now := time.Now()

	require.False(t, svc.confirmOpenAIOAuth429Context(context.Background(), accountID, now))
	svc.clearOpenAIOAuth429Streak(accountID)
	require.False(t, svc.confirmOpenAIOAuth429Context(context.Background(), accountID, now.Add(2*time.Second)),
		"a successful request resets the streak, so the next 429 is treated as the first")
}

func int64PtrForCodexGuardTest(value int64) *int64 { return &value }
