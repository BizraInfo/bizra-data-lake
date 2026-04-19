import type { NextConfig } from "next";

// BIZRA Dema Console — forked from Z.ai prototype, stripped of vendor
// lock-in per cycle-7/prototype-adoption-adr-v1.md.
//
// Build discipline:
//   - typescript.ignoreBuildErrors REMOVED (was true upstream — fixed so
//     static type errors fail the build)
//   - reactStrictMode enabled to surface unsafe-lifecycle usage early
//   - allowedDevOrigins narrowed to loopback; gateway runs on :7421
const nextConfig: NextConfig = {
  output: "standalone",
  reactStrictMode: true,
  allowedDevOrigins: ["http://127.0.0.1", "http://localhost"],
};

export default nextConfig;
