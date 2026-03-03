if ("serviceWorker" in navigator) {
  navigator.serviceWorker
    .register("/service-worker.js")
    .then((reg) => {
      console.log("[SW] registered:", reg.scope);
    })
    .catch((err) => {
      console.warn("[SW] registration failed:", err);
    });
}
