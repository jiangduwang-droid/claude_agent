import { useCallback, useRef } from "react";
import { useChatStore, type ChatMessage, type TokenUsage, type ChatStore } from "@/stores/chatStore";
import { API_BASE, getToken } from "@/lib/api";

const _batchBuffers: Record<string, { delta: string; rafId: number | null }> = {};

function batchAppend(
  convId: string,
  msgId: string,
  delta: string,
  streamId: string,
  appendFn: (id: string, msgId: string, delta: string) => void
) {
  if (!_batchBuffers[streamId]) {
    _batchBuffers[streamId] = { delta: "", rafId: null };
  }
  _batchBuffers[streamId].delta += delta;

  if (_batchBuffers[streamId].rafId === null) {
    _batchBuffers[streamId].rafId = requestAnimationFrame(() => {
      const batched = _batchBuffers[streamId]?.delta || "";
      _batchBuffers[streamId].delta = "";
      _batchBuffers[streamId].rafId = null;
      if (batched) {
        appendFn(convId, msgId, batched);
      }
    });
  }
}

function flushBatch(streamId: string, convId: string, msgId: string, appendFn: (id: string, msgId: string, delta: string) => void) {
  if (_batchBuffers[streamId]?.delta) {
    appendFn(convId, msgId, _batchBuffers[streamId].delta);
    _batchBuffers[streamId].delta = "";
  }
  if (_batchBuffers[streamId]?.rafId !== null) {
    cancelAnimationFrame(_batchBuffers[streamId].rafId!);
    _batchBuffers[streamId].rafId = null;
  }
}

interface SSEEvent {
  type: string;
  [key: string]: unknown;
}

function parseSSEEvent(data: string): SSEEvent | null {
  try {
    return JSON.parse(data);
  } catch {
    return null;
  }
}

export function useChatStream() {
  const abortControllerRef = useRef<AbortController | null>(null);
  const store = useChatStore();

  const sendMessage = useCallback(
    async (
      conversationId: string,
      content: string,
      attachments?: File[]
    ) => {
      const streamId = `${conversationId}-${Date.now()}`;

      const {
        initConversation,
        addMessage,
        appendToMessage,
        updateMessageStatus,
        setStreaming,
        setUsage,
        setContextTokens,
        setCompressionPending,
        setThinkingStep,
        setStepState,
        addToolCall,
        updateToolCallResult,
        setPendingConfirmation,
      } = store;

      initConversation(conversationId);
      setStreaming(conversationId, true);

      const userMsgId = `user-${Date.now()}`;
      const assistantMsgId = `assistant-${Date.now()}`;

      const userMessage: ChatMessage = {
        id: userMsgId,
        role: "user",
        content,
        toolCalls: [],
        status: "complete",
        attachments: attachments?.map((f) => ({ name: f.name, content: "" })),
      };

      addMessage(conversationId, userMessage);

      const assistantMessage: ChatMessage = {
        id: assistantMsgId,
        role: "assistant",
        content: "",
        toolCalls: [],
        status: "streaming",
      };

      addMessage(conversationId, assistantMessage);

      abortControllerRef.current = new AbortController();

      let attachmentData: Array<{ name: string; content: string }> | undefined;
      if (attachments && attachments.length > 0) {
        attachmentData = await Promise.all(
          attachments.map(async (f) => ({
            name: f.name,
            content: await f.text(),
          }))
        );
      }

      try {
        const token = getToken();
        const response = await fetch(
          `${API_BASE}/chat/${conversationId}/stream`,
          {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              ...(token ? { Authorization: `Bearer ${token}` } : {}),
            },
            body: JSON.stringify({ content, attachments: attachmentData }),
            signal: abortControllerRef.current.signal,
          }
        );

        if (!response.ok) {
          const errData = await response.json().catch(() => ({ detail: "Stream failed" }));
          throw new Error(errData.detail || `HTTP ${response.status}`);
        }

        const reader = response.body?.getReader();
        const decoder = new TextDecoder();
        let buffer = "";

        if (!reader) throw new Error("No response body");

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          buffer += decoder.decode(value, { stream: true });
          const events = buffer.split("\n\n");
          buffer = events.pop() || "";

          for (const eventStr of events) {
            const lines = eventStr.trim().split("\n");
            const dataLine = lines.find((l) => l.startsWith("data: "));
            if (!dataLine) continue;

            const data = dataLine.slice(6);
            const event = parseSSEEvent(data);
            if (!event) continue;

            await handleSSEChunk(event, conversationId, assistantMsgId, streamId, store);
          }
        }

        flushBatch(streamId, conversationId, assistantMsgId, appendToMessage);
      } catch (err: unknown) {
        flushBatch(streamId, conversationId, assistantMsgId, appendToMessage);

        if (err instanceof Error && err.name === "AbortError") {
          updateMessageStatus(conversationId, assistantMsgId, "cancelled");
        } else {
          const errorMsg = err instanceof Error ? err.message : "Unknown error";
          appendToMessage(conversationId, assistantMsgId, `\n\n**Error**: ${errorMsg}`);
          updateMessageStatus(conversationId, assistantMsgId, "error");
        }
      } finally {
        setStreaming(conversationId, false);
        setThinkingStep(conversationId, undefined);
        abortControllerRef.current = null;
      }
    },
    [store]
  );

  const cancelStream = useCallback(async (conversationId: string) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    try {
      const token = getToken();
      await fetch(`${API_BASE}/chat/${conversationId}/cancel`, {
        method: "POST",
        headers: { ...(token ? { Authorization: `Bearer ${token}` } : {}) },
      });
    } catch {
      // Ignore cancel errors
    }
  }, []);

  const confirmTool = useCallback(
    async (conversationId: string, toolCallId: string, approved: boolean) => {
      try {
        const token = getToken();
        await fetch(`${API_BASE}/chat/${conversationId}/confirm`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            ...(token ? { Authorization: `Bearer ${token}` } : {}),
          },
          body: JSON.stringify({ tool_call_id: toolCallId, approved }),
        });
        store.setPendingConfirmation(conversationId, null);
      } catch (err) {
        console.error("Confirm failed:", err);
      }
    },
    [store]
  );

  return { sendMessage, cancelStream, confirmTool };
}

async function handleSSEChunk(
  event: SSEEvent,
  convId: string,
  msgId: string,
  streamId: string,
  store: ChatStore
) {
  const {
    appendToMessage,
    updateMessageStatus,
    setUsage,
    setContextTokens,
    setCompressionPending,
    setThinkingStep,
    setStepState,
    addToolCall,
    updateToolCallResult,
    setPendingConfirmation,
  } = store;

  switch (event.type) {
    case "text_delta":
      batchAppend(convId, msgId, event.content as string, streamId, appendToMessage);
      break;

    case "thinking":
      setThinkingStep(convId, event.step as string);
      break;

    case "start_step":
      setThinkingStep(convId, `第 ${event.step} 步推理中...`);
      setStepState(convId, event.step as number, (event.max_steps as number) || 25);
      break;

    case "finish_step":
      setThinkingStep(convId, undefined);
      break;

    case "tool_call_progress": {
      const toolId = event.tool_id as string || event.tool_call_id as string;
      if (toolId) {
        addToolCall(convId, msgId, {
          id: toolId,
          name: event.tool_name as string,
          args: event.args as Record<string, unknown>,
          status: "running",
        });
      }
      break;
    }

    case "tool_result": {
      const toolId = event.tool_call_id as string;
      if (toolId) {
        updateToolCallResult(
          convId,
          msgId,
          toolId,
          event.result as string || "",
          event.status === "error" ? "error" : "done"
        );
      }
      break;
    }

    case "done":
      updateMessageStatus(convId, msgId, "complete");
      if (event.usage) {
        setUsage(convId, event.usage as TokenUsage);
      }
      if (event.cumulative_usage) {
        setUsage(convId, event.cumulative_usage as TokenUsage);
      }
      if (event.context_tokens !== undefined) {
        setContextTokens(convId, event.context_tokens as number);
      }
      if (event.compression_pending !== undefined) {
        setCompressionPending(convId, event.compression_pending as boolean);
      }
      break;

    case "error":
      appendToMessage(convId, msgId, `\n\n**Error**: ${event.error || "请求出错"}`);
      updateMessageStatus(convId, msgId, "error");
      break;

    case "aborted":
      updateMessageStatus(convId, msgId, "cancelled");
      break;

    case "doom_loop_detected":
      appendToMessage(convId, msgId, `\n\n⚠️ 检测到死循环,工具 "${event.tool_name}" 被重复调用。`);
      break;

    case "step_limit_reached":
      appendToMessage(convId, msgId, "\n\n⚠️ 已达到最大步数限制。");
      break;

    case "tool_confirm_request":
      setPendingConfirmation(convId, {
        confirmationId: event.confirmation_id as string,
        toolName: event.tool_name as string,
        toolCallId: event.tool_call_id as string,
        args: event.args as Record<string, unknown>,
        profile: event.profile as string,
      });
      break;

    case "tool_confirm_result":
      setPendingConfirmation(convId, null);
      break;

    default:
      // Unknown event type, ignore
      break;
  }
}
