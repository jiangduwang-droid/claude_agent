import type { NextConfig } from "next";

const API_BACKEND_URL = process.env.API_BACKEND_URL || "http://localhost:8000/api/v1";

const nextConfig: NextConfig = {
  output: "standalone",
  async rewrites() {
    return [
      {
        source: "/api/v1/:path*",
        destination: `${API_BACKEND_URL}/:path*`,
      },
    ];
  },
};

export default nextConfig;
