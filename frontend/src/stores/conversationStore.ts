import { create } from "zustand";
import { api } from "@/lib/api";

export interface Conversation {
  id: string;
  title: string | null;
  model: string | null;
  sandbox_id: string | null;
  created_at?: string;
}

interface ConversationStore {
  conversations: Conversation[];
  isLoading: boolean;

  fetchConversations: () => Promise<void>;
  createConversation: (title?: string, model?: string) => Promise<Conversation>;
  deleteConversation: (id: string) => Promise<void>;
  updateConversation: (id: string, title: string) => Promise<void>;
}

export const useConversationStore = create<ConversationStore>((set, get) => ({
  conversations: [],
  isLoading: false,

  fetchConversations: async () => {
    set({ isLoading: true });
    try {
      const conversations = await api.get<Conversation[]>("/conversations");
      set({ conversations, isLoading: false });
    } catch {
      set({ isLoading: false });
    }
  },

  createConversation: async (title?: string, model?: string) => {
    const conv = await api.post<Conversation>("/conversations", { title, model });
    set((state) => ({
      conversations: [conv, ...state.conversations],
    }));
    return conv;
  },

  deleteConversation: async (id: string) => {
    await api.delete(`/conversations/${id}`);
    set((state) => ({
      conversations: state.conversations.filter((c) => c.id !== id),
    }));
  },

  updateConversation: async (id: string, title: string) => {
    await api.put(`/conversations/${id}`, { title });
    set((state) => ({
      conversations: state.conversations.map((c) =>
        c.id === id ? { ...c, title } : c
      ),
    }));
  },
}));
