"use client";

import { useState, useEffect } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { useConversationStore, type Conversation } from "@/stores/conversationStore";

export function Sidebar() {
  const router = useRouter();
  const { logout, user } = useAuthStore();
  const { conversations, fetchConversations, createConversation, deleteConversation } =
    useConversationStore();
  const [confirmDelete, setConfirmDelete] = useState<string | null>(null);

  const handleNewChat = async () => {
    const conv = await createConversation();
    router.push(`/chat/${conv.id}`);
  };

  const handleDelete = async (id: string) => {
    await deleteConversation(id);
    setConfirmDelete(null);
  };

  useEffect(() => {
    fetchConversations();
  }, []);

  return (
    <div className="flex flex-col h-full">
      <button
        onClick={handleNewChat}
        className="m-3 px-4 py-2.5 rounded-lg bg-[#238636] hover:bg-[#2ea043] text-white text-sm font-medium transition-colors"
      >
        + 新建对话
      </button>

      <div className="flex-1 overflow-y-auto px-2">
        {conversations.length === 0 ? (
          <div className="text-center text-[#8b949e] text-sm py-8">暂无对话</div>
        ) : (
          <div className="space-y-1">
            {conversations.map((conv: Conversation) => (
              <div
                key={conv.id}
                className="group flex items-center px-3 py-2 rounded-lg hover:bg-[#1c2128] cursor-pointer"
              >
                <Link
                  href={`/chat/${conv.id}`}
                  className="flex-1 text-sm text-[#e6edf3] truncate"
                >
                  {conv.title || "新对话"}
                </Link>
                {confirmDelete === conv.id ? (
                  <div className="flex gap-1">
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        handleDelete(conv.id);
                      }}
                      className="text-xs text-[#f85149] hover:underline"
                    >
                      确认
                    </button>
                    <button
                      onClick={(e) => {
                        e.preventDefault();
                        setConfirmDelete(null);
                      }}
                      className="text-xs text-[#8b949e] hover:underline"
                    >
                      取消
                    </button>
                  </div>
                ) : (
                  <button
                    onClick={(e) => {
                      e.preventDefault();
                      setConfirmDelete(conv.id);
                    }}
                    className="opacity-0 group-hover:opacity-100 text-[#f85149] text-xs px-1"
                  >
                    删除
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="border-t border-[#30363d] p-2 space-y-1">
        <Link
          href="/knowledge"
          className="block px-3 py-2 rounded-lg hover:bg-[#1c2128] text-sm text-[#8b949e]"
        >
          知识库
        </Link>
        <Link
          href="/skills"
          className="block px-3 py-2 rounded-lg hover:bg-[#1c2128] text-sm text-[#8b949e]"
        >
          技能
        </Link>
        <Link
          href="/settings"
          className="block px-3 py-2 rounded-lg hover:bg-[#1c2128] text-sm text-[#8b949e]"
        >
          设置
        </Link>
        {user?.role === "admin" && (
          <Link
            href="/admin"
            className="block px-3 py-2 rounded-lg hover:bg-[#1c2128] text-sm text-[#8b949e]"
          >
            管理后台
          </Link>
        )}
        <button
          onClick={() => {
            logout();
            router.push("/login");
          }}
          className="block w-full text-left px-3 py-2 rounded-lg hover:bg-[#1c2128] text-sm text-[#8b949e]"
        >
          退出
        </button>
      </div>
    </div>
  );
}
