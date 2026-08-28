"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useConversationStore } from "@/stores/conversationStore";
import { useAuthStore } from "@/stores/authStore";

export default function NewChatPage() {
  const router = useRouter();
  const { createConversation } = useConversationStore();
  const { isAuthenticated, checkAuth } = useAuthStore();

  useEffect(() => {
    checkAuth();
    const token = localStorage.getItem("token");
    if (!token) {
      router.push("/login");
      return;
    }
    createConversation().then((conv) => {
      router.replace(`/chat/${conv.id}`);
    });
  }, []);

  return (
    <div className="flex items-center justify-center min-h-screen bg-[#0d1117]">
      <div className="text-[#8b949e]">创建新对话...</div>
    </div>
  );
}
