import { create } from "zustand";
import { api } from "@/lib/api";

export interface KnowledgeBase {
  id: string;
  name: string;
  description: string | null;
}

export interface Document {
  id: string;
  filename: string;
  file_type: string;
  chunk_count?: number;
}

export interface SearchResult {
  content: string;
  filename: string;
  score: number;
  source: string;
}

interface KnowledgeStore {
  knowledgeBases: KnowledgeBase[];
  currentKB: { kb: KnowledgeBase; documents: Document[] } | null;
  searchResults: SearchResult[];
  isLoading: boolean;

  fetchKBs: () => Promise<void>;
  createKB: (name: string, description?: string) => Promise<void>;
  deleteKB: (id: string) => Promise<void>;
  fetchKB: (id: string) => Promise<void>;
  uploadDocument: (kbId: string, file: File) => Promise<void>;
  deleteDocument: (kbId: string, docId: string) => Promise<void>;
  search: (query: string, kbIds?: string[]) => Promise<void>;
}

export const useKnowledgeStore = create<KnowledgeStore>((set) => ({
  knowledgeBases: [],
  currentKB: null,
  searchResults: [],
  isLoading: false,

  fetchKBs: async () => {
    set({ isLoading: true });
    try {
      const kbs = await api.get<KnowledgeBase[]>("/knowledge");
      set({ knowledgeBases: kbs, isLoading: false });
    } catch {
      set({ isLoading: false });
    }
  },

  createKB: async (name: string, description?: string) => {
    const kb = await api.post<KnowledgeBase>("/knowledge", { name, description });
    set((state) => ({
      knowledgeBases: [kb, ...state.knowledgeBases],
    }));
  },

  deleteKB: async (id: string) => {
    await api.delete(`/knowledge/${id}`);
    set((state) => ({
      knowledgeBases: state.knowledgeBases.filter((kb) => kb.id !== id),
    }));
  },

  fetchKB: async (id: string) => {
    set({ isLoading: true });
    try {
      const data = await api.get<{ id: string; name: string; description: string; documents: Document[] }>(
        `/knowledge/${id}`
      );
      set({
        currentKB: {
          kb: { id: data.id, name: data.name, description: data.description },
          documents: data.documents,
        },
        isLoading: false,
      });
    } catch {
      set({ isLoading: false });
    }
  },

  uploadDocument: async (kbId: string, file: File) => {
    const formData = new FormData();
    formData.append("file", file);
    await api.upload(`/knowledge/${kbId}/documents`, formData);
  },

  deleteDocument: async (kbId: string, docId: string) => {
    await api.delete(`/knowledge/${kbId}/documents/${docId}`);
    set((state) => {
      if (!state.currentKB) return {};
      return {
        currentKB: {
          ...state.currentKB,
          documents: state.currentKB.documents.filter((d) => d.id !== docId),
        },
      };
    });
  },

  search: async (query: string, kbIds?: string[]) => {
    set({ isLoading: true });
    try {
      const result = await api.post<{ results: SearchResult[] }>("/knowledge/search", {
        query,
        kb_ids: kbIds,
      });
      set({ searchResults: result.results, isLoading: false });
    } catch {
      set({ isLoading: false });
    }
  },
}));
