"use client";

import { useEffect, useRef } from "react";
import { useChatStore } from "@/stores/chatStore";
import { MessageBubble } from "./MessageBubble";
import { ChatInput } from "./ChatInput";
import { ToolCallCard } from "./ToolCallCard";
import { PendingConfirmation } from "./PendingConfirmation";
import { useChatStream } from "@/hooks/useChatStream";

export function ChatContainer({ conversationId }: { conversationId: string }) {
  const {
    initConversation,
    getConversationState,
    setCurrentConversation,
  } = useChatStore();
  const { sendMessage, cancelStream, confirmTool } = useChatStream();
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    initConversation(conversationId);
    setCurrentConversation(conversationId);
  }, [conversationId]);

  const state = getConversationState(conversationId);
  const { messages, isStreaming, contextTokens, stepState, pendingConfirmation, thinkingStep } =
    state as any;

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const tokenPercent = contextTokens
    ? Math.min(100, (contextTokens / 128000) * 100)
    : 0;
  const tokenColor = tokenPercent > 90 ? "#f85149" : tokenPercent > 70 ? "#d29922" : "#3fb950";

  const handleSend = (content: string, attachments?: File[]) => {
    sendMessage(conversationId, content, attachments);
  };

  return (
    <div className="flex flex-col h-full">
      <div className="flex-1 overflow-y-auto">
        <div className="max-w-3xl mx-auto px-4 py-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-[60vh] text-[#8b949e]">
              <div className="text-4xl mb-4">🤖</div>
              <div className="text-lg">开始一个新的对话</div>
              <div className="text-sm mt-2">输入你的问题或上传文件开始</div>
            </div>
          ) : (
            messages.map((msg: any) => (
              <MessageBubble key={msg.id} message={msg} />
            ))
          )}
          {isStreaming && thinkingStep && (
            <div className="flex items-center gap-2 text-[#8b949e] text-sm py-2">
              <div className="animate-pulse">●</div>
              {thinkingStep}
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      </div>

      {contextTokens !== undefined && contextTokens > 0 && (
        <div className="max-w-3xl mx-auto w-full px-4 pb-1">
          <div className="flex items-center gap-2">
            <div className="flex-1 h-1 bg-[#21262d] rounded-full overflow-hidden">
              <div
                className="h-full rounded-full transition-all"
                style={{ width: `${tokenPercent}%`, backgroundColor: tokenColor }}
              />
            </div>
            <span className="text-xs text-[#8b949e]">
              {contextTokens.toLocaleString()} / 128K tokens
            </span>
          </div>
        </div>
      )}

      {pendingConfirmation && (
        <PendingConfirmation
          conversationId={conversationId}
          confirmation={pendingConfirmation}
          onConfirm={(approved) =>
            confirmTool(conversationId, pendingConfirmation.toolCallId, approved)
          }
        />
      )}

      <ChatInput
        onSend={handleSend}
        disabled={isStreaming}
        onCancel={() => cancelStream(conversationId)}
        isStreaming={isStreaming}
      />
    </div>
  );
}
