"use client";

import { useState, useEffect } from "react";
import { useParams } from "next/navigation";
import { Shell } from "@/components/layout/Shell";
import { useKnowledgeStore } from "@/stores/knowledgeStore";

export default function KnowledgeDetailPage() {
  const params = useParams();
  const kbId = params.id as string;
  const {
    currentKB,
    fetchKB,
    uploadDocument,
    deleteDocument,
    searchResults,
    search,
    isLoading,
  } = useKnowledgeStore();
  const [searchQuery, setSearchQuery] = useState("");

  useEffect(() => {
    fetchKB(kbId);
  }, [kbId]);

  const handleUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files || e.target.files.length === 0) return;
    for (const file of Array.from(e.target.files)) {
      await uploadDocument(kbId, file);
    }
    fetchKB(kbId);
  };

  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    await search(searchQuery, [kbId]);
  };

  return (
    <Shell>
      <div className="max-w-2xl mx-auto px-6 py-8">
        {currentKB && (
          <>
            <h1 className="text-2xl font-bold text-[#e6edf3] mb-2">{currentKB.kb.name}</h1>
            {currentKB.kb.description && (
              <p className="text-sm text-[#8b949e] mb-6">{currentKB.kb.description}</p>
            )}
          </>
        )}

        <div className="bg-[#161b22] border border-[#30363d] rounded-lg p-4 mb-6">
          <div className="flex items-center gap-2 mb-3">
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSearch()}
              placeholder="搜索知识库..."
              className="flex-1 px-4 py-2 rounded-lg bg-[#0d1117] border border-[#30363d] text-[#e6edf3] text-sm focus:outline-none focus:border-[#58a6ff]"
            />
            <button
              onClick={handleSearch}
              disabled={isLoading}
              className="px-4 py-2 rounded-lg bg-[#238636] hover:bg-[#2ea043] text-white text-sm font-medium"
            >
              搜索
            </button>
          </div>
          {searchResults.length > 0 && (
            <div className="space-y-2 mt-4">
              {searchResults.map((r, i) => (
                <div key={i} className="bg-[#0d1117] border border-[#30363d] rounded-lg p-3">
                  <div className="text-xs text-[#8b949e] mb-1">{r.filename}</div>
                  <div className="text-sm text-[#e6edf3]">{r.content.slice(0, 300)}...</div>
                </div>
              ))}
            </div>
          )}
        </div>

        <div className="mb-4">
          <label className="inline-block px-4 py-2 rounded-lg bg-[#21262d] border border-[#30363d] text-[#e6edf3] text-sm cursor-pointer hover:border-[#58a6ff]">
            上传文档
            <input type="file" multiple onChange={handleUpload} className="hidden" />
          </label>
        </div>

        <div>
          <h2 className="text-lg font-medium text-[#e6edf3] mb-3">文档列表</h2>
          {currentKB && currentKB.documents.length > 0 ? (
            <div className="space-y-2">
              {currentKB.documents.map((doc) => (
                <div
                  key={doc.id}
                  className="flex items-center justify-between bg-[#161b22] border border-[#30363d] rounded-lg p-3"
                >
                  <div>
                    <div className="text-sm text-[#e6edf3]">{doc.filename}</div>
                    <div className="text-xs text-[#8b949e]">{doc.file_type}</div>
                  </div>
                  <button
                    onClick={() => deleteDocument(kbId, doc.id)}
                    className="text-xs text-[#f85149] hover:underline"
                  >
                    删除
                  </button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-center text-[#8b949e] py-8">暂无文档</div>
          )}
        </div>
      </div>
    </Shell>
  );
}
