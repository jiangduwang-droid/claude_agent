"use client";

import { useState, useEffect } from "react";
import { Shell } from "@/components/layout/Shell";
import { api } from "@/lib/api";

export default function SkillsPage() {
  const [skills, setSkills] = useState<any[]>([]);
  const [gitUrl, setGitUrl] = useState("");
  const [loading, setLoading] = useState(false);

  const fetchSkills = async () => {
    try {
      const data = await api.get<any[]>("/skills");
      setSkills(data);
    } catch {
      // ignore
    }
  };

  useEffect(() => {
    fetchSkills();
  }, []);

  const handleInstall = async () => {
    if (!gitUrl.trim()) return;
    setLoading(true);
    try {
      await api.post("/skills/install", { git_url: gitUrl });
      setGitUrl("");
      fetchSkills();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Install failed");
    } finally {
      setLoading(false);
    }
  };

  const handleRemove = async (name: string) => {
    try {
      await api.delete(`/skills/${name}`);
      fetchSkills();
    } catch (err) {
      alert(err instanceof Error ? err.message : "Remove failed");
    }
  };

  return (
    <Shell>
      <div className="max-w-2xl mx-auto px-6 py-8">
        <h1 className="text-2xl font-bold text-[#e6edf3] mb-6">技能管理</h1>

        <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-4 mb-6">
          <h2 className="text-lg font-medium text-[#e6edf3] mb-3">安装技能</h2>
          <div className="flex gap-2">
            <input
              type="text"
              value={gitUrl}
              onChange={(e) => setGitUrl(e.target.value)}
              placeholder="https://github.com/user/skill-repo"
              className="flex-1 px-4 py-2 rounded-lg bg-[#0d1117] border border-[#30363d] text-[#e6edf3] text-sm focus:outline-none focus:border-[#58a6ff]"
            />
            <button
              onClick={handleInstall}
              disabled={loading}
              className="px-4 py-2 rounded-lg bg-[#238636] hover:bg-[#2ea043] text-white text-sm font-medium disabled:opacity-50"
            >
              {loading ? "安装中..." : "安装"}
            </button>
          </div>
          <p className="text-xs text-[#8b949e] mt-2">
            支持: github.com, gitlab.com, bitbucket.org
          </p>
        </div>

        <div>
          <h2 className="text-lg font-medium text-[#e6edf3] mb-3">已安装技能</h2>
          {skills.length === 0 ? (
            <div className="text-center text-[#8b949e] py-8">暂无已安装技能</div>
          ) : (
            <div className="space-y-2">
              {skills.map((skill) => (
                <div
                  key={skill.name}
                  className="flex items-center justify-between bg-[#161b22] border border-[#30363d] rounded-lg p-4"
                >
                  <div>
                    <div className="text-sm font-medium text-[#e6edf3]">{skill.name}</div>
                    <div className="text-xs text-[#8b949e]">
                      {skill.format} · {skill.tools_count} tools
                    </div>
                    <div className="text-xs text-[#8b949e] mt-1">{skill.description}</div>
                  </div>
                  <button
                    onClick={() => handleRemove(skill.name)}
                    className="text-xs text-[#f85149] hover:underline"
                  >
                    卸载
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </Shell>
  );
}
