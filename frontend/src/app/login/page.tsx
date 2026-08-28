"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";

export default function LoginPage() {
  const router = useRouter();
  const { login } = useAuthStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError("");
    setLoading(true);
    try {
      await login(email, password);
      router.push("/chat/new");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Login failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-[#0d1117]">
      <div className="w-full max-w-md mx-4">
        <h1 className="text-3xl font-bold text-[#e6edf3] text-center mb-2">Cloud Claude Agent</h1>
        <p className="text-[#8b949e] text-center mb-8">登录到你的账户</p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-sm text-[#8b949e] mb-1">邮箱</label>
            <input
              type="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              required
              className="w-full px-4 py-2.5 rounded-lg bg-[#161b22] border border-[#30363d] text-[#e6edf3] focus:outline-none focus:border-[#58a6ff]"
            />
          </div>
          <div>
            <label className="block text-sm text-[#8b949e] mb-1">密码</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              required
              className="w-full px-4 py-2.5 rounded-lg bg-[#161b22] border border-[#30363d] text-[#e6edf3] focus:outline-none focus:border-[#58a6ff]"
            />
          </div>
          {error && <div className="text-sm text-[#f85149]">{error}</div>}
          <button
            type="submit"
            disabled={loading}
            className="w-full py-2.5 rounded-lg bg-[#238636] hover:bg-[#2ea043] text-white font-medium disabled:opacity-50"
          >
            {loading ? "登录中..." : "登录"}
          </button>
        </form>
        <div className="text-center mt-4">
          <button
            onClick={() => router.push("/register")}
            className="text-sm text-[#58a6ff] hover:underline"
          >
            没有账户? 注册
          </button>
        </div>
      </div>
    </div>
  );
}
