"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";

export default function RegisterPage() {
  const router = useRouter();
  const { register } = useAuthStore();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  const passwordChecks = {
    length: password.length >= 8,
    upper: /[A-Z]/.test(password),
    lower: /[a-z]/.test(password),
    digit: /\d/.test(password),
  };

  const isValid = Object.values(passwordChecks).every(Boolean);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!isValid) {
      setError("密码不符合要求");
      return;
    }
    setError("");
    setLoading(true);
    try {
      await register(email, password);
      router.push("/chat/new");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Registration failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex items-center justify-center min-h-screen bg-[#0d1117]">
      <div className="w-full max-w-md mx-4">
        <h1 className="text-3xl font-bold text-[#e6edf3] text-center mb-2">Cloud Claude Agent</h1>
        <p className="text-[#8b949e] text-center mb-8">创建新账户</p>
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
            <div className="mt-2 space-y-1 text-xs">
              <div className={passwordChecks.length ? "text-[#3fb950]" : "text-[#8b949e]"}>
                {passwordChecks.length ? "✓" : "○"} 至少8个字符
              </div>
              <div className={passwordChecks.upper ? "text-[#3fb950]" : "text-[#8b949e]"}>
                {passwordChecks.upper ? "✓" : "○"} 至少1个大写字母
              </div>
              <div className={passwordChecks.lower ? "text-[#3fb950]" : "text-[#8b949e]"}>
                {passwordChecks.lower ? "✓" : "○"} 至少1个小写字母
              </div>
              <div className={passwordChecks.digit ? "text-[#3fb950]" : "text-[#8b949e]"}>
                {passwordChecks.digit ? "✓" : "○"} 至少1个数字
              </div>
            </div>
          </div>
          {error && <div className="text-sm text-[#f85149]">{error}</div>}
          <button
            type="submit"
            disabled={loading || !isValid}
            className="w-full py-2.5 rounded-lg bg-[#238636] hover:bg-[#2ea043] text-white font-medium disabled:opacity-50"
          >
            {loading ? "注册中..." : "注册"}
          </button>
        </form>
        <div className="text-center mt-4">
          <button
            onClick={() => router.push("/login")}
            className="text-sm text-[#58a6ff] hover:underline"
          >
            已有账户? 登录
          </button>
        </div>
      </div>
    </div>
  );
}
