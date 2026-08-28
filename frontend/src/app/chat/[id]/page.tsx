"use client";

import { useParams } from "next/navigation";
import { Shell } from "@/components/layout/Shell";
import { ChatContainer } from "@/components/chat/ChatContainer";

export default function ChatPage() {
  const params = useParams();
  const conversationId = params.id as string;

  return (
    <Shell>
      <ChatContainer conversationId={conversationId} />
    </Shell>
  );
}
