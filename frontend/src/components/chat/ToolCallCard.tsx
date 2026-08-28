"use client";

import { useState } from "react";
import type { ToolCall } from "@/stores/chatStore";

export function ToolCallCard({ toolCall }: { toolCall: ToolCall }) {
  const [expanded, setExpanded] = useState(false);

  const statusIcon = {
    pending: "○",
    running: "⟳",
    done: "✓",
    error: "✗",
  }[toolCall.status];

  const statusColor = {
    pending: "#8b949e",
    running: "#58a6ff",
    done: "#3fb950",
    error: "#f85149",
  }[toolCall.status];

  const resultStr = toolCall.result
    ? typeof toolCall.result === "string"
      ? toolCall.result
      : JSON.stringify(toolCall.result, null, 2)
    : "";

  return (
    <div className="border border-[#30363d] rounded-lg overflow-hidden bg-[#161b22]">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center gap-2 px-3 py-2 hover:bg-[#1c2128] transition-colors"
      >
        <span style={{ color: statusColor }} className="text-sm">
          {toolCall.status === "running" ? (
            <span className="inline-block animate-spin">{statusIcon}</span>
          ) : (
            statusIcon
          )}
        </span>
        <span className="text-sm font-medium text-[#58a6ff]">{toolCall.name}</span>
        {toolCall.args && Object.keys(toolCall.args).length > 0 && (
          <span className="text-xs text-[#8b949e] truncate flex-1 text-left">
            {Object.entries(toolCall.args)
              .slice(0, 2)
              .map(([k, v]) => `${k}: ${typeof v === "string" ? v.slice(0, 50) : JSON.stringify(v).slice(0, 50)}`)
              .join(", ")}
          </span>
        )}
        {toolCall.duration_ms && (
          <span className="text-xs text-[#8b949e]">{toolCall.duration_ms}ms</span>
        )}
        <span className="text-xs text-[#8b949e]">{expanded ? "▲" : "▼"}</span>
      </button>

      {expanded && (
        <div className="border-t border-[#30363d] p-3 space-y-2">
          {toolCall.args && Object.keys(toolCall.args).length > 0 && (
            <div>
              <div className="text-xs text-[#8b949e] mb-1">参数</div>
              <pre className="text-xs text-[#e6edf3] bg-[#0d1117] p-2 rounded overflow-x-auto">
                {JSON.stringify(toolCall.args, null, 2)}
              </pre>
            </div>
          )}
          {resultStr && (
            <div>
              <div className="text-xs text-[#8b949e] mb-1">结果</div>
              <pre
                className={`text-xs p-2 rounded overflow-x-auto max-h-60 ${
                  toolCall.status === "error"
                    ? "text-[#f85149] bg-[#0d1117]"
                    : "text-[#3fb950] bg-[#0d1117]"
                }`}
              >
                {resultStr.slice(0, 2000)}
                {resultStr.length > 2000 ? "\n...(truncated)" : ""}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
