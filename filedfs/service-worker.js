// filedfs/service-worker.js
// ============================================================
// BIZRA PWA Service Worker — strict static-asset caching
// ============================================================

const CACHE_NAME = "bizra-v2";
const APP_SHELL = ["/", "/index.html"];
const CACHEABLE_ASSET_REGEX = /\.(?:js|css|png|jpe?g|svg|ico|woff2?|ttf|eot)$/i;

function isCacheableAssetRequest(request) {
  const url = new URL(request.url);
  if (url.origin !== self.location.origin) return false;
  if (url.search) return false;
  return CACHEABLE_ASSET_REGEX.test(url.pathname);
}

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

  if (request.mode === "navigate") {
    event.respondWith(fetch(request).catch(() => caches.match("/index.html")));
    return;
  }

  if (!isCacheableAssetRequest(request)) return;

  event.respondWith(
    caches.match(request).then((cached) => {
      if (cached) return cached;
      return fetch(request).then((response) => {
        if (!response.ok) return response;
        const clone = response.clone();
        caches
          .open(CACHE_NAME)
          .then((cache) => cache.put(request, clone))
          .catch(() => {});
        return response;
      });
    })
  );
});
