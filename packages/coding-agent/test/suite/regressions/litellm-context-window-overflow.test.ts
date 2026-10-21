import { fauxAssistantMessage } from "@earendil-works/pi-ai";
import { afterEach, describe, expect, it } from "vitest";
import { createHarness, type Harness } from "../harness.ts";

const LITELLM_OVERFLOW = `400: ${JSON.stringify({
	message:
		"litellm.ContextWindowExceededError: litellm.BadRequestError: litellm._pre_call_checks: Context Window exceeded for given call. No models have context window large enough for this call.\nModel=vertex_ai/gemini-4-argon, Max Input Tokens=1048576, Got=1050306\nmodel=gcp/google/gemini-4-argon. context_window_fallbacks=None. fallbacks=None.",
	type: "invalid_request_error",
	param: null,
	code: "400",
})}`;

describe("LiteLLM pre-call context overflow recovery", () => {
	const harnesses: Harness[] = [];

	afterEach(() => {
		while (harnesses.length > 0) harnesses.pop()?.cleanup();
	});

	it("compacts and retries a zero-usage rejection below the local token threshold", async () => {
		const harness = await createHarness({
			provider: "custom-gateway",
			models: [{ id: "gemini-4-argon", contextWindow: 1_048_576 }],
			settings: {
				compaction: { keepRecentTokens: 100 },
				retry: { enabled: true, maxRetries: 3, baseDelayMs: 1 },
			},
		});
		harnesses.push(harness);
		// Keep a completed recent turn so compaction can summarize older history.
		const timestamp = Date.now() - 10_000;
		const model = harness.getModel();
		for (const text of ["Old task notes. ".repeat(200), "Recent task notes. ".repeat(50)]) {
			harness.sessionManager.appendMessage({ role: "user", content: text, timestamp });
			harness.sessionManager.appendMessage({
				...fauxAssistantMessage("Task reviewed", { timestamp }),
				provider: model.provider,
				api: model.api,
				model: model.id,
			});
		}
		harness.session.agent.state.messages = harness.sessionManager.buildSessionContext().messages;
		harness.setResponses([
			() => {
				throw new Error(LITELLM_OVERFLOW);
			},
			// Recovery summarizes the older history and the split turn's prefix.
			fauxAssistantMessage("The old task notes have been reviewed."),
			fauxAssistantMessage("Continue with the recent task notes."),
			(context) => {
				expect(JSON.stringify(context)).not.toContain("Old task notes.");
				return fauxAssistantMessage("Continued after compaction");
			},
		]);

		await harness.session.prompt("Continue the task.");

		expect(harness.faux.state.callCount).toBe(4);
		expect(harness.eventsOfType("auto_retry_start")).toHaveLength(0);
		expect(harness.eventsOfType("compaction_start")).toEqual([{ type: "compaction_start", reason: "overflow" }]);
		expect(harness.eventsOfType("compaction_end").at(-1)).toMatchObject({
			reason: "overflow",
			aborted: false,
			willRetry: true,
		});
		expect(harness.session.getLastAssistantText()).toBe("Continued after compaction");
	});
});
