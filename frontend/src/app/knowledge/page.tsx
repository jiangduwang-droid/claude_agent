"use client";

import { useState, useEffect } from "react";
import { Shell } from "@/components/layout/Shell";
import { useKnowledgeStore } from "@/stores/knowledgeStore";

export default function KnowledgePage() {
  const {
    knowledgeBases,
    fetchKBs,
    createKB,
    deleteKB,
  } = useKnowledgeStore();
  const [showCreate, setShowCreate] = useState(false);
  const [newName, setNewName] = useState("");
  const [newDesc, setNewDesc] = useState("");

  useEffect(() => {
    fetchKBs();
  }, []);

  const handleCreate = async () => {
    if (!newName.trim()) return;
    await createKB(newName, newDesc);
    setNewName("");
    setNewDesc("");
    setShowCreate(false);
  };

  return (
    <Shell>
      <div className="max-w-2xl mx-auto px-6 py-8">
        <div className="flex items-center justify-between mb-6">
          <h1 className="text-2xl font-bold text-[#e6edf3]">知识库</h1>
          <button
            onClick={() => setShowCreate(!showCreate)}
            className="px-4 py-2 rounded-lg bg-[#238636] hover:bg-[#2ea043] text-white text-sm font-medium"
          >
            + 新建
          </button>
        </div>

        {showCreate && (
          <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-4 mb-4 space-y-3">
            <input
              type="text"
              value={newName}
              onChange={(e) => setNewName(e.target.value)}
              placeholder="知识库名称"
              className="w-full px-4 py-2 rounded-lg bg-[#0d1117] border border-[#30363d] text-[#e6edf3] text-sm focus:outline-none focus:border-[#58a6ff]"
            />
            <input
              type="text"
              value={newDesc}
              onChange={(e) => setNewDesc(e.target.value)}
              placeholder="描述 (可选)"
              className="w-full px-4 py-2 rounded-lg bg-[#0d1117] border border-[#30363d] text-[#e6edf3] text-sm focus:outline-none focus:border-[#58a6ff]"
            />
            <button
              onClick={handleCreate}
              className="px-4 py-2 rounded-lg bg-[#238636] hover:bg-[#2ea043] text-white text-sm font-medium"
            >
              创建
            </button>
          </div>
        )}

        {knowledgeBases.length === 0 ? (
          <div className="text-center text-[#8b949e] py-8">暂无知识库</div>
        ) : (
          <div className="space-y-2">
            {knowledgeBases.map((kb) => (
              <div
                key={kb.id}
                className="flex items-center justify-between bg-[#161b22] border border-[#30363d] rounded-lg p-4"
              >
                <div>
                  <div className="text-sm font-medium text-[#e6edf3]">{kb.name}</div>
                  {kb.description && (
                    <div className="text-xs text-[#8b949e] mt-1">{kb.description}</div>
                  )}
                </div>
                <div className="flex gap-2">
                  <a
                    href={`/knowledge/${kb.id}`}
                    className="text-xs text-[#58a6ff] hover:underline"
                  >
                    查看
                  </a>
                  <button
                    onClick={() => deleteKB(kb.id)}
                    className="text-xs text-[#f85149] hover:underline"
                  >
                    删除
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Shell>
  );
}
