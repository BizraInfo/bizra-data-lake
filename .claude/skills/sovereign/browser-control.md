---
name: autonomous-browser-control
description: >
  Autonomous Browser Control (AI Cowork) — Full-spectrum autonomous web agent with Playwright.
  Navigate, extract, interact, fill forms, monitor pages, chain multi-step workflows across
  sites, handle authentication, dynamic content, and CAPTCHAs. Production-grade stealth with
  fingerprint randomization, human-like behavior simulation, proxy support, and rate limiting.
  Use when the user asks to browse a website autonomously, extract data from web pages,
  automate web tasks, fill forms, monitor sites for changes, interact with web applications,
  log into services, chain multi-site workflows, collect web data, take screenshots, capture
  page content, simulate user behavior, or perform any browser-based automation. Triggers on
  keywords: browse, automate, extract, monitor, web agent, fill form, login, navigate, click,
  screenshot, page data, site interaction, web workflow, scrape, crawl, bot, headless.
license: MIT
metadata:
  author: m.beshr
  version: '1.0'
  category: automation
---

# Autonomous Browser Control (AI Cowork)

Full autonomous web agent — navigate, extract, interact, and chain multi-step workflows across any website, powered by Playwright with production-grade stealth.

## When to Use This Skill

Use this skill when the user asks you to:

- **Browse or navigate** websites autonomously
- **Extract data** from web pages (tables, text, links, images, structured content)
- **Automate web tasks** — fill forms, click buttons, submit data, upload files
- **Monitor websites** for changes, new content, or price drops
- **Chain multi-step workflows** across multiple sites (e.g., search → compare → extract → report)
- **Log into services** and perform authenticated actions
- **Take screenshots** or capture visual state of pages
- **Interact with dynamic content** — SPAs, infinite scroll, lazy-loaded elements, modals
- **Simulate human behavior** on websites for testing or interaction

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│              AUTONOMOUS BROWSER AGENT                │
│                                                      │
│  ┌──────────┐  ┌──────────┐  ┌───────────────────┐  │
│  │ PLANNER  │→ │ EXECUTOR │→ │ RESULT PROCESSOR  │  │
│  │          │  │          │  │                    │  │
│  │ Decompose│  │ Playwright│  │ Parse + Structure │  │
│  │ tasks    │  │ actions  │  │ data              │  │
│  └──────────┘  └──────────┘  └───────────────────┘  │
│       ↑              │                    │          │
│       │         ┌────┴─────┐              │          │
│       │         │ STEALTH  │              ↓          │
│       │         │ LAYER    │      ┌──────────────┐   │
│       │         └──────────┘      │ OUTPUT       │   │
│       └───────── FEEDBACK ←───────│ JSON/CSV/Log │   │
│                  LOOP             └──────────────┘   │
└─────────────────────────────────────────────────────┘
```

The agent operates in 6 phases:
1. **Planning** — Decompose user intent into discrete browser actions
2. **Stealth Setup** — Configure anti-detection before any navigation
3. **Execution** — Perform browser actions with retry and error recovery
4. **Extraction** — Parse and structure collected data
5. **Validation** — Verify results, retry failed steps
6. **Reporting** — Deliver structured output with logs and screenshots

---

## Instructions

### Phase 1: Planning — Task Decomposition

Before launching the browser, decompose the user's request into an ordered action plan.

**Step 1 — Parse user intent**

Classify the request into one or more action types:

| Action Type | Description | Example |
|-------------|-------------|---------|
| `navigate` | Go to a URL or search for a site | "Go to GitHub" |
| `extract` | Pull data from a page | "Get all product prices" |
| `interact` | Click, type, select, scroll | "Fill out the contact form" |
| `authenticate` | Log into a service | "Log into my dashboard" |
| `monitor` | Watch for changes over time | "Alert me when price drops" |
| `capture` | Screenshot or save page state | "Screenshot the results page" |
| `chain` | Multi-step across pages/sites | "Search, compare, then buy" |

**Step 2 — Build action sequence**

Create an ordered list of atomic browser actions:

```
Action Plan: <task_description>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Step 1: [navigate] → https://example.com
Step 2: [interact] → Click "Login" button
Step 3: [interact] → Fill email field with <value>
Step 4: [interact] → Fill password field with <value>
Step 5: [interact] → Click "Submit"
Step 6: [extract]  → Get dashboard data table
Step 7: [capture]  → Screenshot final state
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Estimated steps: 7
Risk factors: Auth required, possible CAPTCHA
Stealth level: HIGH
```

**Step 3 — Present plan to user for confirmation before execution.**

---

### Phase 2: Stealth Setup — Anti-Detection Layer

**CRITICAL: Always configure stealth BEFORE the first navigation.**

#### 2A — Browser Context Configuration

```javascript
const { chromium } = require('playwright');

async function createStealthContext(options = {}) {
  const {
    proxy = null,
    locale = 'en-US',
    timezone = 'America/New_York',
    viewport = null,
    geolocation = null,
  } = options;

  // Randomize viewport from common resolutions
  const viewports = [
    { width: 1920, height: 1080 },
    { width: 1366, height: 768 },
    { width: 1536, height: 864 },
    { width: 1440, height: 900 },
    { width: 1280, height: 720 },
    { width: 2560, height: 1440 },
  ];
  const selectedViewport = viewport || viewports[Math.floor(Math.random() * viewports.length)];

  // Randomize user agent from real-world common agents
  const userAgents = [
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:123.0) Gecko/20100101 Firefox/123.0',
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.3 Safari/605.1.15',
    'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0',
  ];
  const selectedUA = userAgents[Math.floor(Math.random() * userAgents.length)];

  const launchOptions = {
    headless: true,
    args: [
      '--disable-blink-features=AutomationControlled',
      '--disable-features=IsolateOrigins,site-per-process',
      '--disable-dev-shm-usage',
      '--no-first-run',
      '--no-default-browser-check',
      '--disable-infobars',
      '--window-size=' + selectedViewport.width + ',' + selectedViewport.height,
    ],
  };

  if (proxy) {
    launchOptions.proxy = { server: proxy };
  }

  const browser = await chromium.launch(launchOptions);

  const contextOptions = {
    viewport: selectedViewport,
    userAgent: selectedUA,
    locale: locale,
    timezoneId: timezone,
    permissions: ['geolocation'],
    deviceScaleFactor: Math.random() > 0.5 ? 2 : 1,
    hasTouch: Math.random() > 0.7,
    colorScheme: Math.random() > 0.5 ? 'dark' : 'light',
    extraHTTPHeaders: {
      'Accept-Language': locale === 'en-US' ? 'en-US,en;q=0.9' : `${locale},en;q=0.9`,
      'Accept-Encoding': 'gzip, deflate, br',
      'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8',
      'Sec-Fetch-Dest': 'document',
      'Sec-Fetch-Mode': 'navigate',
      'Sec-Fetch-Site': 'none',
      'Sec-Fetch-User': '?1',
      'Upgrade-Insecure-Requests': '1',
    },
  };

  if (geolocation) {
    contextOptions.geolocation = geolocation;
  }

  const context = await browser.newContext(contextOptions);

  return { browser, context };
}
```

#### 2B — Navigator & WebDriver Override

Inject stealth scripts into every page before any content loads:

```javascript
async function applyStealthScripts(page) {
  await page.addInitScript(() => {
    // Override navigator.webdriver
    Object.defineProperty(navigator, 'webdriver', {
      get: () => undefined,
    });

    // Override navigator.plugins to look real
    Object.defineProperty(navigator, 'plugins', {
      get: () => {
        const plugins = [
          { name: 'Chrome PDF Plugin', filename: 'internal-pdf-viewer' },
          { name: 'Chrome PDF Viewer', filename: 'mhjfbmdgcfjbbpaeojofohoefgiehjai' },
          { name: 'Native Client', filename: 'internal-nacl-plugin' },
        ];
        const pluginArray = Object.create(PluginArray.prototype);
        plugins.forEach((p, i) => {
          const plugin = Object.create(Plugin.prototype);
          Object.defineProperties(plugin, {
            name: { value: p.name, enumerable: true },
            filename: { value: p.filename, enumerable: true },
            description: { value: p.name, enumerable: true },
            length: { value: 1, enumerable: true },
          });
          pluginArray[i] = plugin;
        });
        Object.defineProperty(pluginArray, 'length', { value: plugins.length });
        return pluginArray;
      },
    });

    // Override navigator.languages
    Object.defineProperty(navigator, 'languages', {
      get: () => ['en-US', 'en'],
    });

    // Override permissions query
    const originalQuery = window.navigator.permissions.query;
    window.navigator.permissions.query = (parameters) =>
      parameters.name === 'notifications'
        ? Promise.resolve({ state: Notification.permission })
        : originalQuery(parameters);

    // Patch chrome runtime
    window.chrome = {
      runtime: {
        onMessage: { addListener: () => {}, removeListener: () => {} },
        sendMessage: () => {},
        connect: () => {},
      },
    };

    // Override WebGL vendor/renderer
    const getParameter = WebGLRenderingContext.prototype.getParameter;
    WebGLRenderingContext.prototype.getParameter = function (param) {
      if (param === 37445) return 'Intel Inc.';
      if (param === 37446) return 'Intel Iris OpenGL Engine';
      return getParameter.call(this, param);
    };

    // Override canvas fingerprint with subtle noise
    const toBlob = HTMLCanvasElement.prototype.toBlob;
    HTMLCanvasElement.prototype.toBlob = function (callback, type, quality) {
      const context = this.getContext('2d');
      if (context) {
        const imageData = context.getImageData(0, 0, this.width, this.height);
        for (let i = 0; i < imageData.data.length; i += 4) {
          imageData.data[i] ^= 1; // subtle noise
        }
        context.putImageData(imageData, 0, 0);
      }
      return toBlob.call(this, callback, type, quality);
    };
  });
}
```

#### 2C — Human-Like Behavior Simulation

```javascript
// Randomized delay to simulate human reaction time
async function humanDelay(min = 300, max = 1500) {
  const delay = Math.floor(Math.random() * (max - min) + min);
  await new Promise(resolve => setTimeout(resolve, delay));
}

// Human-like typing with variable speed and occasional typos
async function humanType(page, selector, text, options = {}) {
  const { typoRate = 0.02, minDelay = 50, maxDelay = 200 } = options;

  await page.click(selector);
  await humanDelay(200, 500);

  for (let i = 0; i < text.length; i++) {
    // Occasional typo simulation
    if (Math.random() < typoRate && i > 0) {
      const typoChar = String.fromCharCode(text.charCodeAt(i) + (Math.random() > 0.5 ? 1 : -1));
      await page.keyboard.type(typoChar, { delay: Math.random() * 100 + 50 });
      await humanDelay(100, 300);
      await page.keyboard.press('Backspace');
      await humanDelay(50, 150);
    }

    const charDelay = Math.floor(Math.random() * (maxDelay - minDelay) + minDelay);
    await page.keyboard.type(text[i], { delay: charDelay });
  }
}

// Human-like mouse movement with bezier curves
async function humanClick(page, selector) {
  const element = await page.waitForSelector(selector, { timeout: 10000 });
  const box = await element.boundingBox();

  if (box) {
    // Move to a random point within the element, not dead center
    const x = box.x + box.width * (0.3 + Math.random() * 0.4);
    const y = box.y + box.height * (0.3 + Math.random() * 0.4);

    // Move mouse with slight deviation path
    await page.mouse.move(x, y, { steps: Math.floor(Math.random() * 10 + 5) });
    await humanDelay(50, 200);
    await page.mouse.click(x, y);
  } else {
    await element.click();
  }
}

// Simulate natural scrolling behavior
async function humanScroll(page, options = {}) {
  const { direction = 'down', distance = null, speed = 'normal' } = options;

  const scrollStep = speed === 'fast' ? 300 : speed === 'slow' ? 80 : 150;
  const totalDistance = distance || (Math.floor(Math.random() * 800) + 400);
  const steps = Math.ceil(totalDistance / scrollStep);

  for (let i = 0; i < steps; i++) {
    const delta = direction === 'down' ? scrollStep : -scrollStep;
    await page.mouse.wheel(0, delta);
    await humanDelay(30, 120);

    // Occasionally pause like a human reading
    if (Math.random() < 0.15) {
      await humanDelay(500, 2000);
    }
  }
}

// Random idle behavior between actions
async function humanIdle(page) {
  const actions = [
    async () => { /* just wait */ await humanDelay(1000, 3000); },
    async () => {
      // Random mouse wiggle
      const x = Math.floor(Math.random() * 800 + 100);
      const y = Math.floor(Math.random() * 500 + 100);
      await page.mouse.move(x, y, { steps: 3 });
    },
    async () => {
      // Small scroll
      await page.mouse.wheel(0, Math.random() > 0.5 ? 100 : -100);
    },
  ];
  const action = actions[Math.floor(Math.random() * actions.length)];
  await action();
}
```

#### 2D — Rate Limiting & Request Throttle

```javascript
class RateLimiter {
  constructor(options = {}) {
    this.minInterval = options.minInterval || 2000;  // ms between requests
    this.maxInterval = options.maxInterval || 5000;
    this.maxRequestsPerMinute = options.maxRequestsPerMinute || 20;
    this.requestLog = [];
    this.lastRequestTime = 0;
  }

  async throttle() {
    // Enforce minimum interval with randomization
    const now = Date.now();
    const elapsed = now - this.lastRequestTime;
    const interval = Math.floor(
      Math.random() * (this.maxInterval - this.minInterval) + this.minInterval
    );

    if (elapsed < interval) {
      await new Promise(resolve => setTimeout(resolve, interval - elapsed));
    }

    // Enforce per-minute limit
    const oneMinuteAgo = Date.now() - 60000;
    this.requestLog = this.requestLog.filter(t => t > oneMinuteAgo);

    if (this.requestLog.length >= this.maxRequestsPerMinute) {
      const waitUntil = this.requestLog[0] + 60000;
      const waitMs = waitUntil - Date.now();
      if (waitMs > 0) {
        await new Promise(resolve => setTimeout(resolve, waitMs));
      }
    }

    this.requestLog.push(Date.now());
    this.lastRequestTime = Date.now();
  }
}
```

---

### Phase 3: Execution Engine — Core Browser Actions

#### 3A — Navigation with Smart Waiting

```javascript
async function smartNavigate(page, url, options = {}) {
  const {
    waitUntil = 'domcontentloaded',
    timeout = 30000,
    waitForSelector = null,
    retries = 3,
    screenshotOnLoad = false,
    log = [],
  } = options;

  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      log.push({ action: 'navigate', url, attempt, timestamp: new Date().toISOString() });

      const response = await page.goto(url, { waitUntil, timeout });

      // Wait for dynamic content if needed
      if (waitForSelector) {
        await page.waitForSelector(waitForSelector, { timeout: 15000 });
      }

      // Wait for network idle as a secondary signal
      await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {});

      if (screenshotOnLoad) {
        const screenshotPath = `/home/user/workspace/_browser_screenshots/nav_${Date.now()}.png`;
        await page.screenshot({ path: screenshotPath, fullPage: false });
        log.push({ action: 'screenshot', path: screenshotPath });
      }

      log.push({
        action: 'navigate_success',
        status: response?.status(),
        url: page.url(),
      });

      return { success: true, status: response?.status(), url: page.url() };

    } catch (error) {
      log.push({ action: 'navigate_error', error: error.message, attempt });

      if (attempt < retries) {
        await humanDelay(2000, 5000);
      } else {
        return { success: false, error: error.message };
      }
    }
  }
}
```

#### 3B — Dynamic Content Handling

```javascript
// Handle infinite scroll pages
async function scrollToLoadAll(page, options = {}) {
  const {
    maxScrolls = 50,
    scrollDelay = 1500,
    checkSelector = null,
    stopWhenNoNewContent = true,
  } = options;

  let previousHeight = 0;
  let noChangeCount = 0;
  let totalScrolls = 0;

  while (totalScrolls < maxScrolls) {
    const currentHeight = await page.evaluate(() => document.body.scrollHeight);

    if (stopWhenNoNewContent && currentHeight === previousHeight) {
      noChangeCount++;
      if (noChangeCount >= 3) break; // No new content after 3 scrolls
    } else {
      noChangeCount = 0;
    }

    previousHeight = currentHeight;
    await page.evaluate(() => window.scrollTo(0, document.body.scrollHeight));
    await humanDelay(scrollDelay, scrollDelay + 1000);

    // Check for "Load More" buttons and click them
    const loadMoreBtn = await page.$('button:has-text("Load More"), button:has-text("Show More"), a:has-text("Load More")');
    if (loadMoreBtn) {
      await humanClick(page, 'button:has-text("Load More"), button:has-text("Show More")');
      await humanDelay(1000, 2000);
    }

    totalScrolls++;

    if (checkSelector) {
      const count = await page.$$eval(checkSelector, els => els.length);
      // Log progress for monitoring
    }
  }

  return { totalScrolls, finalHeight: previousHeight };
}

// Handle modals, popups, and cookie banners
async function dismissOverlays(page) {
  const overlaySelectors = [
    // Cookie consent
    'button:has-text("Accept")',
    'button:has-text("Accept All")',
    'button:has-text("Accept Cookies")',
    'button:has-text("I Agree")',
    '[id*="cookie"] button',
    '[class*="cookie"] button',
    '[class*="consent"] button',
    // Newsletter popups
    'button:has-text("No Thanks")',
    'button:has-text("Close")',
    'button[aria-label="Close"]',
    '[class*="modal"] button[class*="close"]',
    '[class*="popup"] button[class*="close"]',
    // Generic dismiss
    '.dismiss',
    '.close-button',
  ];

  for (const selector of overlaySelectors) {
    try {
      const el = await page.$(selector);
      if (el && await el.isVisible()) {
        await el.click();
        await humanDelay(300, 600);
      }
    } catch {
      // Selector not found, continue
    }
  }
}

// Wait for SPA navigation (React, Vue, Angular)
async function waitForSPANavigation(page, options = {}) {
  const { timeout = 10000 } = options;

  await Promise.race([
    page.waitForNavigation({ waitUntil: 'networkidle', timeout }),
    page.waitForTimeout(timeout),
    // Watch for DOM mutations that indicate SPA content loaded
    page.evaluate(() => {
      return new Promise(resolve => {
        const observer = new MutationObserver((mutations, obs) => {
          if (mutations.some(m => m.addedNodes.length > 0)) {
            obs.disconnect();
            resolve();
          }
        });
        observer.observe(document.body, { childList: true, subtree: true });
        setTimeout(() => { observer.disconnect(); resolve(); }, 5000);
      });
    }),
  ]).catch(() => {});
}
```

#### 3C — Form Interaction Engine

```javascript
async function fillForm(page, formData, options = {}) {
  const { humanLike = true, log = [] } = options;

  for (const field of formData) {
    const { selector, value, type = 'text' } = field;

    try {
      await page.waitForSelector(selector, { timeout: 10000 });

      switch (type) {
        case 'text':
        case 'email':
        case 'password':
        case 'search':
          if (humanLike) {
            await humanType(page, selector, value);
          } else {
            await page.fill(selector, value);
          }
          break;

        case 'select':
          await page.selectOption(selector, value);
          break;

        case 'checkbox':
          const isChecked = await page.isChecked(selector);
          if ((value && !isChecked) || (!value && isChecked)) {
            await humanClick(page, selector);
          }
          break;

        case 'radio':
          await humanClick(page, selector);
          break;

        case 'file':
          await page.setInputFiles(selector, value);
          break;

        case 'date':
          await page.fill(selector, value);
          break;

        case 'textarea':
          if (humanLike) {
            await humanType(page, selector, value, { minDelay: 30, maxDelay: 120 });
          } else {
            await page.fill(selector, value);
          }
          break;
      }

      log.push({ action: 'fill_field', selector, type, status: 'success' });
      await humanDelay(200, 800);

    } catch (error) {
      log.push({ action: 'fill_field', selector, type, status: 'error', error: error.message });
    }
  }
}
```

#### 3D — Authentication Handler

```javascript
async function handleAuthentication(page, credentials, options = {}) {
  const { log = [], site = '' } = options;

  // Common login form patterns
  const loginPatterns = [
    // Pattern 1: Email + Password on same page
    {
      detect: 'input[type="email"], input[name="email"], input[id*="email"]',
      fields: [
        { selector: 'input[type="email"], input[name="email"], input[id*="email"]', value: credentials.email, type: 'email' },
        { selector: 'input[type="password"], input[name="password"]', value: credentials.password, type: 'password' },
      ],
      submit: 'button[type="submit"], input[type="submit"], button:has-text("Sign in"), button:has-text("Log in")',
    },
    // Pattern 2: Username + Password
    {
      detect: 'input[name="username"], input[id*="username"], input[name="login"]',
      fields: [
        { selector: 'input[name="username"], input[id*="username"], input[name="login"]', value: credentials.username, type: 'text' },
        { selector: 'input[type="password"]', value: credentials.password, type: 'password' },
      ],
      submit: 'button[type="submit"], input[type="submit"]',
    },
    // Pattern 3: Multi-step (email first, then password)
    {
      detect: 'input[type="email"]:not(:has(~ input[type="password"]))',
      fields: [
        { selector: 'input[type="email"]', value: credentials.email, type: 'email' },
      ],
      submit: 'button[type="submit"], button:has-text("Next"), button:has-text("Continue")',
      // After submit, wait for password field to appear
      thenFields: [
        { selector: 'input[type="password"]', value: credentials.password, type: 'password' },
      ],
      thenSubmit: 'button[type="submit"], button:has-text("Sign in")',
    },
  ];

  for (const pattern of loginPatterns) {
    const detected = await page.$(pattern.detect);
    if (detected) {
      log.push({ action: 'auth_pattern_detected', pattern: pattern.detect });

      await fillForm(page, pattern.fields, { humanLike: true, log });
      await humanDelay(300, 800);
      await humanClick(page, pattern.submit);

      if (pattern.thenFields) {
        await humanDelay(1500, 3000);
        await fillForm(page, pattern.thenFields, { humanLike: true, log });
        await humanDelay(300, 800);
        await humanClick(page, pattern.thenSubmit);
      }

      await humanDelay(2000, 4000);
      log.push({ action: 'auth_submitted', url: page.url() });
      return { success: true };
    }
  }

  log.push({ action: 'auth_no_pattern_found' });
  return { success: false, error: 'No recognized login form found' };
}
```

---

### Phase 4: Data Extraction Engine

#### 4A — Structured Data Extraction

```javascript
async function extractStructuredData(page, schema, options = {}) {
  const { log = [] } = options;

  // Auto-detect data structures on page
  const extractors = {
    // Extract tables
    table: async (selector = 'table') => {
      return await page.$$eval(selector, tables => {
        return tables.map(table => {
          const headers = Array.from(table.querySelectorAll('thead th, tr:first-child th'))
            .map(th => th.textContent.trim());

          const rows = Array.from(table.querySelectorAll('tbody tr, tr:not(:first-child)'))
            .map(tr => {
              const cells = Array.from(tr.querySelectorAll('td'))
                .map(td => td.textContent.trim());
              if (headers.length) {
                const obj = {};
                headers.forEach((h, i) => obj[h] = cells[i] || '');
                return obj;
              }
              return cells;
            });

          return { headers, rows };
        });
      });
    },

    // Extract lists
    list: async (selector = 'ul, ol') => {
      return await page.$$eval(selector, lists => {
        return lists.map(list => ({
          type: list.tagName.toLowerCase(),
          items: Array.from(list.querySelectorAll('li'))
            .map(li => li.textContent.trim()),
        }));
      });
    },

    // Extract links
    links: async (selector = 'a[href]') => {
      return await page.$$eval(selector, anchors => {
        return anchors.map(a => ({
          text: a.textContent.trim(),
          href: a.href,
          title: a.title || null,
        })).filter(l => l.href && !l.href.startsWith('javascript:'));
      });
    },

    // Extract meta/SEO data
    metadata: async () => {
      return await page.evaluate(() => ({
        title: document.title,
        description: document.querySelector('meta[name="description"]')?.content || '',
        ogTitle: document.querySelector('meta[property="og:title"]')?.content || '',
        ogDescription: document.querySelector('meta[property="og:description"]')?.content || '',
        ogImage: document.querySelector('meta[property="og:image"]')?.content || '',
        canonical: document.querySelector('link[rel="canonical"]')?.href || '',
        h1: Array.from(document.querySelectorAll('h1')).map(h => h.textContent.trim()),
      }));
    },

    // Extract images
    images: async (selector = 'img') => {
      return await page.$$eval(selector, imgs => {
        return imgs.map(img => ({
          src: img.src,
          alt: img.alt || '',
          width: img.naturalWidth,
          height: img.naturalHeight,
        })).filter(i => i.src && !i.src.startsWith('data:'));
      });
    },

    // Custom CSS selector extraction
    custom: async (selector, attribute = 'textContent') => {
      return await page.$$eval(selector, (els, attr) => {
        return els.map(el => {
          if (attr === 'textContent') return el.textContent.trim();
          if (attr === 'innerHTML') return el.innerHTML;
          if (attr === 'outerHTML') return el.outerHTML;
          return el.getAttribute(attr) || '';
        });
      }, attribute);
    },
  };

  // If user provides a schema, use it to guide extraction
  if (schema) {
    const result = {};
    for (const [key, config] of Object.entries(schema)) {
      const { type, selector, attribute } = config;
      if (extractors[type]) {
        result[key] = await extractors[type](selector, attribute);
      }
    }
    log.push({ action: 'extract_structured', keys: Object.keys(result) });
    return result;
  }

  // Auto-extraction: detect what's on the page and extract everything
  const autoResult = {};
  autoResult.metadata = await extractors.metadata();
  autoResult.tables = await extractors.table();
  autoResult.links = await extractors.links();
  autoResult.images = await extractors.images();
  autoResult.lists = await extractors.list();

  log.push({ action: 'extract_auto', types: Object.keys(autoResult) });
  return autoResult;
}
```

#### 4B — Pagination Handler

```javascript
async function extractWithPagination(page, extractFn, options = {}) {
  const {
    maxPages = 20,
    nextSelector = 'a:has-text("Next"), button:has-text("Next"), [aria-label="Next"], .pagination .next',
    pageNumberSelector = null,
    log = [],
  } = options;

  const allData = [];
  let currentPage = 1;

  while (currentPage <= maxPages) {
    log.push({ action: 'extract_page', page: currentPage, url: page.url() });

    // Extract data from current page
    const pageData = await extractFn(page);
    allData.push({ page: currentPage, url: page.url(), data: pageData });

    // Look for next page
    const nextBtn = await page.$(nextSelector);
    if (!nextBtn) {
      log.push({ action: 'pagination_end', reason: 'no_next_button', totalPages: currentPage });
      break;
    }

    // Check if next button is disabled
    const isDisabled = await nextBtn.evaluate(el =>
      el.disabled || el.classList.contains('disabled') || el.getAttribute('aria-disabled') === 'true'
    );

    if (isDisabled) {
      log.push({ action: 'pagination_end', reason: 'next_disabled', totalPages: currentPage });
      break;
    }

    await humanClick(page, nextSelector);
    await humanDelay(1500, 3000);
    await page.waitForLoadState('networkidle', { timeout: 10000 }).catch(() => {});
    currentPage++;
  }

  return { pages: allData, totalPages: currentPage };
}
```

---

### Phase 5: Validation & Error Recovery

```javascript
class ExecutionValidator {
  constructor(log = []) {
    this.log = log;
    this.errors = [];
    this.retries = [];
  }

  // Verify a navigation succeeded
  async validateNavigation(page, expectedUrl, expectedContent = null) {
    const currentUrl = page.url();
    const urlMatch = currentUrl.includes(expectedUrl) || currentUrl === expectedUrl;

    let contentMatch = true;
    if (expectedContent) {
      const bodyText = await page.evaluate(() => document.body.innerText);
      contentMatch = bodyText.includes(expectedContent);
    }

    // Check for common error pages
    const isErrorPage = await page.evaluate(() => {
      const title = document.title.toLowerCase();
      const body = document.body.innerText.toLowerCase().slice(0, 500);
      return title.includes('404') || title.includes('error') || title.includes('not found')
        || body.includes('page not found') || body.includes('access denied')
        || body.includes('403 forbidden');
    });

    return { urlMatch, contentMatch, isErrorPage, currentUrl };
  }

  // Verify extracted data quality
  validateExtraction(data, expectations = {}) {
    const issues = [];

    if (expectations.minItems && Array.isArray(data) && data.length < expectations.minItems) {
      issues.push(`Expected at least ${expectations.minItems} items, got ${data.length}`);
    }

    if (expectations.requiredFields) {
      const items = Array.isArray(data) ? data : [data];
      for (const item of items) {
        for (const field of expectations.requiredFields) {
          if (!(field in item) || item[field] === '' || item[field] === null) {
            issues.push(`Missing required field: ${field}`);
          }
        }
      }
    }

    return { valid: issues.length === 0, issues };
  }

  // Retry a failed action with exponential backoff
  async retryAction(actionFn, options = {}) {
    const { maxRetries = 3, baseDelay = 1000, description = 'action' } = options;

    for (let attempt = 1; attempt <= maxRetries; attempt++) {
      try {
        const result = await actionFn();
        this.log.push({ action: 'retry_success', description, attempt });
        return result;
      } catch (error) {
        this.log.push({ action: 'retry_failed', description, attempt, error: error.message });

        if (attempt < maxRetries) {
          const delay = baseDelay * Math.pow(2, attempt - 1) + Math.random() * 1000;
          await new Promise(resolve => setTimeout(resolve, delay));
        } else {
          this.errors.push({ description, error: error.message, attempts: maxRetries });
          throw error;
        }
      }
    }
  }
}
```

---

### Phase 6: Output & Reporting

#### 6A — Structured Data Output

```javascript
const fs = require('fs');
const path = require('path');

async function saveResults(data, options = {}) {
  const {
    format = 'json',
    outputDir = '/home/user/workspace/_browser_output',
    filename = null,
    timestamp = true,
  } = options;

  fs.mkdirSync(outputDir, { recursive: true });

  const ts = timestamp ? `_${new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19)}` : '';
  const baseName = filename || 'browser_results';

  let outputPath;

  switch (format) {
    case 'json':
      outputPath = path.join(outputDir, `${baseName}${ts}.json`);
      fs.writeFileSync(outputPath, JSON.stringify(data, null, 2), 'utf-8');
      break;

    case 'csv':
      outputPath = path.join(outputDir, `${baseName}${ts}.csv`);
      const items = Array.isArray(data) ? data : [data];
      if (items.length > 0) {
        const headers = [...new Set(items.flatMap(item => Object.keys(item)))];
        const csvLines = [
          headers.join(','),
          ...items.map(item =>
            headers.map(h => {
              const val = String(item[h] || '').replace(/"/g, '""');
              return val.includes(',') || val.includes('"') || val.includes('\n')
                ? `"${val}"` : val;
            }).join(',')
          ),
        ];
        fs.writeFileSync(outputPath, csvLines.join('\n'), 'utf-8');
      }
      break;

    case 'markdown':
      outputPath = path.join(outputDir, `${baseName}${ts}.md`);
      // Convert data to markdown tables/lists
      let md = `# Browser Results\n\nGenerated: ${new Date().toISOString()}\n\n`;
      if (Array.isArray(data) && data.length > 0 && typeof data[0] === 'object') {
        const headers = Object.keys(data[0]);
        md += '| ' + headers.join(' | ') + ' |\n';
        md += '| ' + headers.map(() => '---').join(' | ') + ' |\n';
        data.forEach(row => {
          md += '| ' + headers.map(h => String(row[h] || '')).join(' | ') + ' |\n';
        });
      } else {
        md += JSON.stringify(data, null, 2);
      }
      fs.writeFileSync(outputPath, md, 'utf-8');
      break;
  }

  return outputPath;
}
```

#### 6B — Execution Log

```javascript
async function saveExecutionLog(log, options = {}) {
  const {
    outputDir = '/home/user/workspace/_browser_output',
    includeScreenshots = true,
  } = options;

  fs.mkdirSync(outputDir, { recursive: true });

  const logPath = path.join(outputDir, `execution_log_${Date.now()}.json`);
  const logData = {
    startTime: log[0]?.timestamp || new Date().toISOString(),
    endTime: new Date().toISOString(),
    totalActions: log.length,
    errors: log.filter(e => e.action?.includes('error')).length,
    retries: log.filter(e => e.action?.includes('retry')).length,
    actions: log,
  };

  if (includeScreenshots) {
    logData.screenshots = log
      .filter(e => e.action === 'screenshot')
      .map(e => e.path);
  }

  fs.writeFileSync(logPath, JSON.stringify(logData, null, 2), 'utf-8');
  return logPath;
}
```

---

## Decision Flowchart

```
User request
│
├─ "browse" / "go to" / "navigate" / "open"
│   → Phase 1 (Plan) → Phase 2 (Stealth) → Phase 3A (Navigate)
│   → Phase 4 (Extract if data requested) → Phase 6 (Report)
│
├─ "extract" / "get data" / "collect" / "pull from"
│   → Phase 1 → Phase 2 → Phase 3A (Navigate)
│   → Phase 4A (Structured Extraction)
│   → Phase 4B (Pagination if multi-page)
│   → Phase 5 (Validate) → Phase 6 (Save JSON/CSV)
│
├─ "fill form" / "submit" / "sign up" / "register"
│   → Phase 1 → Phase 2 → Phase 3A → Phase 3C (Form Fill)
│   → Phase 6 (Confirmation screenshot + log)
│
├─ "log in" / "authenticate" / "sign in"
│   → Phase 1 → Phase 2 → Phase 3A → Phase 3D (Auth Handler)
│   → Continue to next task step
│
├─ "monitor" / "watch" / "alert when" / "track changes"
│   → Phase 1 → Phase 2 → Phase 3A → Phase 4 (Extract baseline)
│   → Schedule periodic re-extraction → Compare → Alert on change
│
├─ "screenshot" / "capture" / "save page"
│   → Phase 2 → Phase 3A → page.screenshot()
│   → Phase 6 (Save image)
│
└─ Multi-step workflow (complex request spanning multiple sites)
    → Phase 1 (Decompose into sub-tasks per site)
    → For each sub-task: Phase 2 → 3 → 4 → 5
    → Phase 6 (Consolidated report)
```

---

## Examples

### Example 1: Extract product data from an e-commerce site

**User**: "Go to example-store.com and extract all product names, prices, and ratings"

**Agent workflow**:
1. Plan: navigate → dismiss overlays → extract table/cards → paginate → save
2. Stealth: randomized viewport, UA, human delays
3. Navigate to site, dismiss cookie banner
4. Detect product card structure (CSS selectors for name, price, rating)
5. Extract from all pages (pagination handler)
6. Validate: all products have name + price
7. Save to `browser_results.json` and `browser_results.csv`
8. Report: "Extracted 247 products across 12 pages"

### Example 2: Automate a form submission

**User**: "Fill out the contact form on example.com/contact with my details"

**Agent workflow**:
1. Plan: navigate → detect form → fill fields → submit → capture confirmation
2. Navigate to /contact
3. Auto-detect form fields (name, email, message, etc.)
4. Fill with human-like typing
5. Screenshot before submit → confirm with user
6. Submit → capture confirmation page
7. Report: "Form submitted successfully. Confirmation screenshot saved."

### Example 3: Monitor a page for changes

**User**: "Monitor example.com/pricing and tell me when prices change"

**Agent workflow**:
1. Navigate and extract current pricing data (baseline)
2. Save baseline to `_browser_output/baseline_pricing.json`
3. On schedule: re-extract, compare to baseline
4. If diff detected → alert user with old vs new values
5. Update baseline

### Example 4: Multi-site comparison workflow

**User**: "Compare pricing across site-a.com, site-b.com, and site-c.com"

**Agent workflow**:
1. Plan: 3 parallel sub-tasks (one per site)
2. For each site: stealth setup → navigate → find pricing → extract
3. Consolidate into comparison table
4. Save as JSON + CSV + markdown report
5. Report: side-by-side comparison with source URLs

### Example 5: Authenticated dashboard data extraction

**User**: "Log into my analytics dashboard and pull this week's metrics"

**Agent workflow**:
1. Plan: navigate to login → authenticate → navigate to dashboard → extract metrics
2. Stealth setup (match normal user behavior for the site)
3. Navigate to login page → detect form pattern → fill credentials
4. Handle 2FA if prompted (alert user for code)
5. Navigate to metrics page → extract tables/charts
6. Save structured data + screenshot of dashboard
7. Report: "Extracted 5 metric categories from dashboard"
