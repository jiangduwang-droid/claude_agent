"use client";

import type { PendingConfirmation } from "@/stores/chatStore";

interface Props {
  conversationId: string;
  confirmation: PendingConfirmation;
  onConfirm: (approved: boolean) => void;
}

export function PendingConfirmation({ conversationId, confirmation, onConfirm }: Props) {
  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 bg-black/50">
      <div className="bg-[#161b22] border border-[#30363d] rounded-xl p-6 max-w-md w-full mx-4">
        <h3 className="text-lg font-semibold text-[#e6edf3] mb-2">工具执行确认</h3>
        <p className="text-sm text-[#8b949e] mb-4">
          Agent 要执行 <span className="text-[#f85149] font-medium">{confirmation.toolName}</span>，
          是否允许？
        </p>
        {confirmation.args && Object.keys(confirmation.args).length > 0 && (
          <div className="mb-4">
            <div className="text-xs text-[#8b949e] mb-1">参数</div>
            <pre className="text-xs text-[#e6edf3] bg-[#0d1117] p-2 rounded overflow-x-auto max-h-40">
              {JSON.stringify(confirmation.args, null, 2)}
            </pre>
          </div>
        )}
        <div className="text-xs text-[#8b949e] mb-4">
          当前权限 profile: {confirmation.profile}
        </div>
        <div className="flex gap-3 justify-end">
          <button
            onClick={() => onConfirm(false)}
            className="px-4 py-2 rounded-lg bg-[#21262d] border border-[#30363d] text-[#e6edf3] text-sm hover:bg-[#1c2128]"
          >
            拒绝
          </button>
          <button
            onClick={() => onConfirm(true)}
            className="px-4 py-2 rounded-lg bg-[#238636] hover:bg-[#2ea043] text-white text-sm font-medium"
          >
            允许
          </button>
        </div>
      </div>
    </div>
  );
}
