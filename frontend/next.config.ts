import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  reactStrictMode: false,
  generateBuildId: async () => {
    return 'build-id'
  }
};

export default nextConfig;
