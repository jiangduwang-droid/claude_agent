"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuthStore } from "@/stores/authStore";
import { useUIStore } from "@/stores/uiStore";
import { Sidebar } from "./Sidebar";

export function Shell({ children }: { children: React.ReactNode }) {
  const { checkAuth, fetchUser, isAuthenticated } = useAuthStore();
  const { sidebarOpen } = useUIStore();
  const router = useRouter();

  useEffect(() => {
    checkAuth();
    if (isAuthenticated) {
      fetchUser();
    } else {
      const token = localStorage.getItem("token");
      if (!token) {
        router.push("/login");
      } else {
        fetchUser();
      }
    }
  }, []);

  return (
    <div className="flex h-screen overflow-hidden bg-[#0d1117]">
      {sidebarOpen && (
        <div className="w-[256px] flex-shrink-0 border-r border-[#30363d] bg-[#161b22]">
          <Sidebar />
        </div>
      )}
      <main className="flex-1 overflow-hidden">{children}</main>
    </div>
  );
}
