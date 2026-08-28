"use client";

import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { ToolCallCard } from "./ToolCallCard";
import type { ChatMessage } from "@/stores/chatStore";

export function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} mb-4`}>
      <div className={`max-w-[85%] ${isUser ? "items-end" : "items-start"}`}>
        <div
          className={`rounded-2xl px-4 py-3 ${
            isUser
              ? "bg-[#1c2128] border border-[#30363d]"
              : "bg-[#0d1117]"
          }`}
        >
          {message.attachments && message.attachments.length > 0 && (
            <div className="flex flex-wrap gap-2 mb-2">
              {message.attachments.map((att, i) => (
                <div
                  key={i}
                  className="flex items-center gap-1.5 px-2 py-1 rounded-md bg-[#21262d] text-xs text-[#8b949e]"
                >
                  📎 {att.name}
                </div>
              ))}
            </div>
          )}

          <div className="markdown-body text-[#e6edf3] text-sm leading-relaxed">
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {message.content || ""}
            </ReactMarkdown>
          </div>

          {message.toolCalls && message.toolCalls.length > 0 && (
            <div className="mt-3 space-y-2">
              {message.toolCalls.map((tc) => (
                <ToolCallCard key={tc.id} toolCall={tc} />
              ))}
            </div>
          )}

          {message.status === "streaming" && (
            <div className="inline-block w-2 h-4 bg-[#58a6ff] animate-pulse ml-1" />
          )}
          {message.status === "error" && (
            <div className="mt-2 text-xs text-[#f85149]">消息出错</div>
          )}
          {message.status === "cancelled" && (
            <div className="mt-2 text-xs text-[#d29922]">已取消</div>
          )}
        </div>
      </div>
    </div>
  );
}
