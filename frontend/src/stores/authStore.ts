import { create } from "zustand";
import { api, setToken, clearToken, getToken } from "@/lib/api";

interface AuthUser {
  id: string;
  email: string;
  role: string;
}

interface AuthState {
  user: AuthUser | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  login: (email: string, password: string) => Promise<void>;
  register: (email: string, password: string) => Promise<void>;
  logout: () => void;
  checkAuth: () => void;
  fetchUser: () => Promise<void>;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  token: null,
  isAuthenticated: false,
  isLoading: false,

  login: async (email: string, password: string) => {
    const response = await api.post<{ access_token: string; user_id: string; email: string; role: string }>(
      "/auth/login",
      { email, password },
      { skipAuth: true }
    );
    setToken(response.access_token);
    set({
      token: response.access_token,
      user: { id: response.user_id, email: response.email, role: response.role },
      isAuthenticated: true,
    });
  },

  register: async (email: string, password: string) => {
    const response = await api.post<{ access_token: string; user_id: string; email: string; role: string }>(
      "/auth/register",
      { email, password },
      { skipAuth: true }
    );
    setToken(response.access_token);
    set({
      token: response.access_token,
      user: { id: response.user_id, email: response.email, role: response.role },
      isAuthenticated: true,
    });
  },

  logout: () => {
    clearToken();
    set({ user: null, token: null, isAuthenticated: false });
  },

  checkAuth: () => {
    const token = getToken();
    if (token) {
      set({ token, isAuthenticated: true });
    } else {
      set({ token: null, isAuthenticated: false });
    }
  },

  fetchUser: async () => {
    try {
      const user = await api.get<AuthUser>("/auth/me");
      set({ user, isAuthenticated: true });
    } catch {
      clearToken();
      set({ user: null, token: null, isAuthenticated: false });
    }
  },
}));
