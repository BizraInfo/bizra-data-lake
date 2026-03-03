# Phase 56.04: Medium — Frontend Hardening + Operational Reproducibility

> Standing on Giants: OWASP (CSP, 2012) · Fielding (REST / cache semantics, 2000) · Docker Inc (image pinning best practices)

## F13: Service Worker Caches Dynamic Responses

### Current State

File: `filedfs/service-worker.js:34-54`

```javascript
self.addEventListener("fetch", (event) => {
    if (request.method !== "GET") return;
    // API/LLM calls excluded (good)
    // But ALL other GET responses are cached:
    event.respondWith(
        caches.match(request).then((cached) => {
            if (cached) return cached;
            return fetch(request).then((response) => {
                const clone = response.clone();
                // caches everything that falls through
```

While `/v1/`, `/api/`, and LLM endpoints are excluded, other dynamic content
(e.g., `/health`, manifest.json, any future API route not matching the exclusion
patterns) will be cached and served stale.

### Required Behavior

Use an explicit allowlist of cacheable paths rather than a denylist of dynamic ones.
Only cache static shell assets (JS, CSS, images, fonts).

### Pseudocode

```javascript
const CACHEABLE_EXTENSIONS = /\.(js|css|png|jpg|jpeg|svg|ico|woff2?|ttf|eot)$/;

self.addEventListener("fetch", (event) => {
    const { request } = event;
    if (request.method !== "GET") return;

    const url = new URL(request.url);

    // Only cache static assets by extension
    if (!CACHEABLE_EXTENSIONS.test(url.pathname)) return;

    // Also skip anything with query params (cache-busted assets)
    // ...existing cache-first logic for matching requests...
});
```

### Files Modified

| File | Change |
|------|--------|
| `filedfs/service-worker.js` | Switch from denylist to extension-based allowlist |

### TDD Anchors

```javascript
test("caches .js files", async () => {
    const response = await fetchThroughSW("/main.js");
    const cached = await caches.match("/main.js");
    expect(cached).toBeDefined();
});

test("does not cache /health", async () => {
    await fetchThroughSW("/health");
    const cached = await caches.match("/health");
    expect(cached).toBeUndefined();
});

test("does not cache manifest.json", async () => {
    await fetchThroughSW("/manifest.json");
    const cached = await caches.match("/manifest.json");
    expect(cached).toBeUndefined();
});
```

---

## F14: Missing Content Security Policy

### Current State

File: `filedfs/index.html:1-30`

No `Content-Security-Policy` meta tag or HTTP header. With localhost WebSocket
bridges (F2, F12), XSS in the frontend can connect to the bridge and issue
node commands.

### Required Behavior

Add a CSP meta tag restricting:
- `script-src` to `'self'` (no inline except the SW registration, which can be moved to a file)
- `connect-src` to `'self'` + known WS endpoints
- `style-src` to `'self'` + Google Fonts
- `font-src` to Google Fonts CDN
- `img-src` to `'self'` + `data:`

### Pseudocode

```html
<head>
    <meta charset="UTF-8" />
    <meta http-equiv="Content-Security-Policy" content="
        default-src 'self';
        script-src 'self';
        style-src 'self' https://fonts.googleapis.com 'unsafe-inline';
        font-src https://fonts.gstatic.com;
        connect-src 'self' ws://localhost:* http://localhost:*;
        img-src 'self' data:;
        object-src 'none';
        base-uri 'self';
        form-action 'self';
    " />
    ...
</head>
```

Move inline `<script>` (SW registration) to a separate file (`/sw-register.js`):

```html
<!-- Before: inline script -->
<!-- After: -->
<script src="/sw-register.js"></script>
```

### Files Modified

| File | Change |
|------|--------|
| `filedfs/index.html` | Add CSP meta tag, externalize inline script |
| `filedfs/sw-register.js` (new) | SW registration logic moved from inline |

### TDD Anchors

```javascript
test("index.html has Content-Security-Policy meta tag", async () => {
    const html = await fs.readFile("filedfs/index.html", "utf8");
    expect(html).toContain("Content-Security-Policy");
    expect(html).toContain("script-src 'self'");
    expect(html).toContain("object-src 'none'");
});

test("no inline scripts in index.html", async () => {
    const html = await fs.readFile("filedfs/index.html", "utf8");
    // Should not have <script>...</script> with inline code
    const inlineScripts = html.match(/<script(?! src)[^>]*>[^<]+<\/script>/g);
    expect(inlineScripts).toBeNull();
});
```

---

## F15: Mutable Image Tags + Broad RBAC

### Current State

File: `bizra-omega/docker-compose.yml:19`

```yaml
image: bizra-api:latest   # mutable tag — not reproducible
```

File: `deploy/k8s/base/rbac.yaml:17-20`

```yaml
rules:
  - apiGroups: [""]
    resources: ["configmaps", "secrets"]
    verbs: ["get", "list", "watch"]
```

The `secrets` resource with `list` and `watch` allows the service account to
enumerate ALL secrets in the namespace, not just the ones it needs.

### Required Behavior

1. Pin image tags to specific versions or digests
2. Restrict RBAC to only the specific secrets the app needs (by `resourceNames`)

### Pseudocode

```yaml
# docker-compose.yml:
services:
  bizra-api:
    image: bizra-api:2.0.0   # pinned version
    # Or with digest:
    # image: bizra-api@sha256:abc123...

# rbac.yaml — restrict secrets access:
rules:
  - apiGroups: [""]
    resources: ["configmaps"]
    verbs: ["get", "list", "watch"]
  - apiGroups: [""]
    resources: ["secrets"]
    resourceNames: ["bizra-secrets"]   # only the specific secret
    verbs: ["get"]                     # no list/watch
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get"]                     # removed list/watch (not needed)
  - apiGroups: [""]
    resources: ["services", "endpoints"]
    verbs: ["get", "list"]             # removed watch (not needed for discovery)
```

### Files Modified

| File | Change |
|------|--------|
| `bizra-omega/docker-compose.yml` | Pin image tags to workspace version (`2.0.0`) |
| `deploy/k8s/base/rbac.yaml` | Restrict secrets to `resourceNames`, reduce verbs |

### TDD Anchors

```python
# tests/integration/test_k8s_manifests.py

def test_no_latest_tags_in_compose():
    compose = yaml.safe_load(Path("bizra-omega/docker-compose.yml").read_text())
    for svc_name, svc in compose.get("services", {}).items():
        image = svc.get("image", "")
        assert ":latest" not in image, f"{svc_name} uses :latest tag"

def test_rbac_secrets_restricted_to_named():
    rbac = yaml.safe_load(Path("deploy/k8s/base/rbac.yaml").read_text())
    # Find the secrets rule
    for doc in yaml.safe_load_all(Path("deploy/k8s/base/rbac.yaml").read_text()):
        if doc and doc.get("kind") == "Role":
            for rule in doc.get("rules", []):
                if "secrets" in rule.get("resources", []):
                    assert "resourceNames" in rule, "secrets rule must use resourceNames"
                    assert "list" not in rule.get("verbs", []), "secrets should not have list"
                    assert "watch" not in rule.get("verbs", []), "secrets should not have watch"

def test_rbac_pods_no_list():
    """Pods rule should only have 'get', not 'list' or 'watch'."""
    ...
```

---

## Implementation Order

All medium findings can be patched independently. Suggested order:

1. **F15** (RBAC + tags) — pure config, zero runtime risk
2. **F14** (CSP) — HTML-only, plus one new JS file
3. **F13** (service worker) — narrow change, easy to test manually
