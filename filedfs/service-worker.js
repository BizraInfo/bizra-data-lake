// filedfs/service-worker.js
// ============================================================
// BIZRA PWA Service Worker — network-first for API, cache-first for shell
// ============================================================

const CACHE_NAME = "bizra-v1";
const APP_SHELL = ["/", "/index.html", "/main.jsx", "/App.jsx", "/manifest.json"];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_NAME)
      .then((cache) => cache.addAll(APP_SHELL))
      .catch(() => Promise.resolve())
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((k) => k !== CACHE_NAME)
            .map((k) => caches.delete(k))
        )
      )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const { request } = event;
  if (request.method !== "GET") return;

  // Network-first for API/LLM calls — never cache dynamic responses
  if (
    request.url.includes("/v1/") ||
    request.url.includes("/api/") ||
    request.url.includes("localhost:11434") ||
    request.url.includes("localhost:1234")
  ) {
    return; // pass through to network
  }

  // Cache-first for static shell assets, network fallback
  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request)
        .then((response) => {
          const clone = response.clone();
          caches
            .open(CACHE_NAME)
            .then((cache) => cache.put(request, clone))
            .catch(() => {});
          return response;
        })
        .catch(() => caches.match("/index.html"));
    })
  );
});
