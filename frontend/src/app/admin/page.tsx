"use client";

import { useState, useEffect } from "react";
import { Shell } from "@/components/layout/Shell";
import { api } from "@/lib/api";

export default function AdminPage() {
  const [stats, setStats] = useState<any>(null);

  useEffect(() => {
    api.get("/admin/stats").then(setStats).catch(() => {});
  }, []);

  return (
    <Shell>
      <div className="max-w-2xl mx-auto px-6 py-8">
        <h1 className="text-2xl font-bold text-[#e6edf3] mb-6">管理后台</h1>
        {stats && (
          <div className="grid grid-cols-2 gap-4">
            <StatCard label="用户" value={stats.users} />
            <StatCard label="对话" value={stats.conversations} />
            <StatCard label="消息" value={stats.messages} />
            <StatCard label="知识库" value={stats.knowledge_bases} />
            <StatCard label="文档" value={stats.documents} />
            <StatCard label="活跃沙箱" value={stats.active_sandboxes} />
          </div>
        )}
      </div>
    </Shell>
  );
}

function StatCard({ label, value }: { label: string; value: number }) {
  return (
    <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-4 text-center">
      <div className="text-2xl font-bold text-[#58a6ff]">{value}</div>
      <div className="text-sm text-[#8b949e] mt-1">{label}</div>
    </div>
  );
}
