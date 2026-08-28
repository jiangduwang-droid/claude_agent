import { create } from "zustand";
import { useShallow } from "zustand/react/shallow";

export interface ToolCall {
  id: string;
  name: string;
  args?: Record<string, unknown>;
  result?: string;
  status: "pending" | "running" | "done" | "error";
  duration_ms?: number;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  toolCalls: ToolCall[];
  status: "complete" | "streaming" | "error" | "cancelled";
  attachments?: Array<{ name: string; content: string }>;
  createdAt?: string;
}

export interface TokenUsage {
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
}

export interface StepState {
  currentStep: number;
  maxSteps: number;
  thinking: boolean;
}

export interface PendingConfirmation {
  confirmationId: string;
  toolName: string;
  toolCallId: string;
  args: Record<string, unknown>;
  profile: string;
}

interface ConversationState {
  messages: ChatMessage[];
  isStreaming: boolean;
  cumulativeUsage: TokenUsage;
  contextTokens?: number;
  compressionState?: { pending: boolean };
  stepState?: StepState;
  pendingConfirmation?: PendingConfirmation | null;
  thinkingStep?: string;
}

export interface ChatStore {
  conversations: Record<string, ConversationState>;
  currentConversationId: string | null;

  setCurrentConversation: (id: string | null) => void;
  getConversationState: (id: string) => ConversationState;
  initConversation: (id: string) => void;
  setMessages: (id: string, messages: ChatMessage[]) => void;
  addMessage: (id: string, message: ChatMessage) => void;
  updateMessage: (id: string, messageId: string, updates: Partial<ChatMessage>) => void;
  appendToMessage: (id: string, messageId: string, delta: string) => void;
  updateMessageStatus: (id: string, messageId: string, status: ChatMessage["status"]) => void;
  setStreaming: (id: string, streaming: boolean) => void;
  setUsage: (id: string, usage: TokenUsage) => void;
  setContextTokens: (id: string, tokens: number) => void;
  setCompressionPending: (id: string, pending: boolean) => void;
  setThinkingStep: (id: string, step: string | undefined) => void;
  setStepState: (id: string, step: number, maxSteps: number) => void;
  addToolCall: (id: string, messageId: string, toolCall: ToolCall) => void;
  updateToolCallResult: (id: string, messageId: string, toolCallId: string, result: string, status: ToolCall["status"]) => void;
  setPendingConfirmation: (id: string, confirmation: PendingConfirmation | null) => void;
  resetConversation: (id: string) => void;
}

function createInitialState(): ConversationState {
  return {
    messages: [],
    isStreaming: false,
    cumulativeUsage: { prompt_tokens: 0, completion_tokens: 0, total_tokens: 0 },
    contextTokens: 0,
    pendingConfirmation: null,
  };
}

export const useChatStore = create<ChatStore>((set, get) => ({
  conversations: {},
  currentConversationId: null,

  setCurrentConversation: (id) => set({ currentConversationId: id }),

  getConversationState: (id) => {
    const state = get().conversations[id];
    return state || createInitialState();
  },

  initConversation: (id) => {
    set((state) => {
      if (state.conversations[id]) return {};
      return {
        conversations: {
          ...state.conversations,
          [id]: createInitialState(),
        },
      };
    });
  },

  setMessages: (id, messages) => {
    set((state) => ({
      conversations: {
        ...state.conversations,
        [id]: { ...state.conversations[id], messages } as ConversationState,
      },
    }));
  },

  addMessage: (id, message) => {
    set((state) => {
      const conv = state.conversations[id] || createInitialState();
      return {
        conversations: {
          ...state.conversations,
          [id]: { ...conv, messages: [...conv.messages, message] },
        },
      };
    });
  },

  updateMessage: (id, messageId, updates) => {
    set((state) => {
      const conv = state.conversations[id];
      if (!conv) return {};
      return {
        conversations: {
          ...state.conversations,
          [id]: {
            ...conv,
            messages: conv.messages.map((m) =>
              m.id === messageId ? { ...m, ...updates } : m
            ),
          },
        },
      };
    });
  },

  appendToMessage: (id, messageId, delta) => {
    set((state) => {
      const conv = state.conversations[id];
      if (!conv) return {};
      return {
        conversations: {
          ...state.conversations,
          [id]: {
            ...conv,
            messages: conv.messages.map((m) =>
              m.id === messageId ? { ...m, content: m.content + delta } : m
            ),
          },
        },
      };
    });
  },

  updateMessageStatus: (id, messageId, status) => {
    set((state) => {
      const conv = state.conversations[id];
      if (!conv) return {};
      return {
        conversations: {
          ...state.conversations,
          [id]: {
            ...conv,
            messages: conv.messages.map((m) =>
              m.id === messageId ? { ...m, status } : m
            ),
          },
        },
      };
    });
  },

  setStreaming: (id, streaming) => {
    set((state) => {
      const conv = state.conversations[id] || createInitialState();
      return {
        conversations: {
          ...state.conversations,
          [id]: { ...conv, isStreaming: streaming },
        },
      };
    });
  },

  setUsage: (id, usage) => {
    set((state) => {
      const conv = state.conversations[id] || createInitialState();
      return {
        conversations: {
          ...state.conversations,
          [id]: { ...conv, cumulativeUsage: usage },
        },
      };
    });
  },

  setContextTokens: (id, tokens) => {
    set((state) => {
      const conv = state.conversations[id] || createInitialState();
      return {
        conversations: {
          ...state.conversations,
          [id]: { ...conv, contextTokens: tokens },
        },
      };
    });
  },

  setCompressionPending: (id, pending) => {
    set((state) => {
      const conv = state.conversations[id] || createInitialState();
      return {
        conversations: {
          ...state.conversations,
          [id]: { ...conv, compressionState: { pending } },
        },
      };
    });
  },

  setThinkingStep: (id, step) => {
    set((state) => {
      const conv = state.conversations[id] || createInitialState();
      return {
        conversations: {
          ...state.conversations,
          [id]: { ...conv, thinkingStep: step },
        },
      };
    });
  },

  setStepState: (id, step, maxSteps) => {
    set((state) => {
      const conv = state.conversations[id] || createInitialState();
      return {
        conversations: {
          ...state.conversations,
          [id]: {
            ...conv,
            stepState: { currentStep: step, maxSteps, thinking: true },
          },
        },
      };
    });
  },

  addToolCall: (id, messageId, toolCall) => {
    set((state) => {
      const conv = state.conversations[id];
      if (!conv) return {};
      return {
        conversations: {
          ...state.conversations,
          [id]: {
            ...conv,
            messages: conv.messages.map((m) => {
              if (m.id !== messageId) return m;
              const exists = m.toolCalls.some((tc) => tc.id === toolCall.id);
              if (exists) return m;
              return { ...m, toolCalls: [...m.toolCalls, toolCall] };
            }),
          },
        },
      };
    });
  },

  updateToolCallResult: (id, messageId, toolCallId, result, status) => {
    set((state) => {
      const conv = state.conversations[id];
      if (!conv) return {};
      return {
        conversations: {
          ...state.conversations,
          [id]: {
            ...conv,
            messages: conv.messages.map((m) =>
              m.id === messageId
                ? {
                    ...m,
                    toolCalls: m.toolCalls.map((tc) =>
                      tc.id === toolCallId
                        ? { ...tc, result, status }
                        : tc
                    ),
                  }
                : m
            ),
          },
        },
      };
    });
  },

  setPendingConfirmation: (id, confirmation) => {
    set((state) => {
      const conv = state.conversations[id] || createInitialState();
      return {
        conversations: {
          ...state.conversations,
          [id]: { ...conv, pendingConfirmation: confirmation },
        },
      };
    });
  },

  resetConversation: (id) => {
    set((state) => {
      const newConvs = { ...state.conversations };
      delete newConvs[id];
      return { conversations: newConvs };
    });
  },
}));
