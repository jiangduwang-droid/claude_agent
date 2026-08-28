"use client";

import { useState, useEffect } from "react";
import { Shell } from "@/components/layout/Shell";
import { api } from "@/lib/api";
import { useAuthStore } from "@/stores/authStore";

export default function SettingsPage() {
  const { user, fetchUser } = useAuthStore();
  const [provider, setProvider] = useState("anthropic");
  const [model, setModel] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [models, setModels] = useState<Record<string, string[]>>({});
  const [saved, setSaved] = useState(false);

  useEffect(() => {
    api.get("/settings/provider").then((data: any) => {
      setProvider(data.provider);
      setModel(data.model);
    }).catch(() => {});
    api.get("/settings/models").then((data: any) => {
      setModels(data.models);
    }).catch(() => {});
  }, []);

  const handleSave = async () => {
    await api.put("/settings/provider", { provider, model, api_key: apiKey || undefined });
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <Shell>
      <div className="max-w-2xl mx-auto px-6 py-8">
        <h1 className="text-2xl font-bold text-[#e6edf3] mb-6">设置</h1>

        <div className="space-y-6">
          <div>
            <h2 className="text-lg font-medium text-[#e6edf3] mb-3">账户信息</h2>
            <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-4 space-y-2">
              <div className="text-sm">
                <span className="text-[#8b949e]">邮箱:</span>{" "}
                <span className="text-[#e6edf3]">{user?.email}</span>
              </div>
              <div className="text-sm">
                <span className="text-[#8b949e]">角色:</span>{" "}
                <span className="text-[#e6edf3]">{user?.role}</span>
              </div>
            </div>
          </div>

          <div>
            <h2 className="text-lg font-medium text-[#e6edf3] mb-3">LLM 提供商</h2>
            <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-4 space-y-4">
              <div>
                <label className="block text-sm text-[#8b949e] mb-1">提供商</label>
                <select
                  value={provider}
                  onChange={(e) => {
                    setProvider(e.target.value);
                    const firstModel = models[e.target.value]?.[0];
                    if (firstModel) setModel(firstModel);
                  }}
                  className="w-full px-4 py-2 rounded-lg bg-[#0d1117] border border-[#30363d] text-[#e6edf3]"
                >
                  {Object.keys(models).map((p) => (
                    <option key={p} value={p}>{p}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm text-[#8b949e] mb-1">模型</label>
                <select
                  value={model}
                  onChange={(e) => setModel(e.target.value)}
                  className="w-full px-4 py-2 rounded-lg bg-[#0d1117] border border-[#30363d] text-[#e6edf3]"
                >
                  {(models[provider] || []).map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm text-[#8b949e] mb-1">
                  API Key (可选, 留空保持不变)
                </label>
                <input
                  type="password"
                  value={apiKey}
                  onChange={(e) => setApiKey(e.target.value)}
                  placeholder="sk-..."
                  className="w-full px-4 py-2 rounded-lg bg-[#0d1117] border border-[#30363d] text-[#e6edf3] focus:outline-none focus:border-[#58a6ff]"
                />
              </div>
              <button
                onClick={handleSave}
                className="px-4 py-2 rounded-lg bg-[#238636] hover:bg-[#2ea043] text-white text-sm font-medium"
              >
                {saved ? "已保存!" : "保存"}
              </button>
            </div>
          </div>
        </div>
      </div>
    </Shell>
  );
}
