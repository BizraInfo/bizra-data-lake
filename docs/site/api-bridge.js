/**
 * BIZRA API Bridge — connects the frontend to live sovereign data.
 *
 * Fetches from:
 *   - Kernel API (localhost:9740) — health, missions, briefing
 *   - URP state (~/.bizra/urp/urp_state.json) — sea, membrane
 *   - Home Base (~/.bizra/home_base.json) — hardware, software, data
 *   - SEED ledger (~/.bizra/seed_ledger.jsonl) — balance, history
 *
 * For local cockpit: fetches directly from kernel API.
 * For public site: shows cached/static data with "live on your machine" CTA.
 */

const KERNEL_URL = 'http://127.0.0.1:9740';
const FALLBACK_STATS = {
    seed_balance: 0,
    total_missions: 0,
    total_nodes: 1,
    knowledge_entries: 0,
    receipts_recorded: 0,
    ihsan_avg: 0.95,
    gini: 0.0,
    uptime_s: 0,
    agents: 12,
    models: 0,
    cpu: 'Unknown',
    ram_gb: 0,
    gpu: 'Unknown',
    kernel_alive: false,
};

async function fetchLiveStats() {
    try {
        const resp = await fetch(`${KERNEL_URL}/api/live-stats`, { signal: AbortSignal.timeout(3000) });
        if (!resp.ok) return null;
        return await resp.json();
    } catch {
        return null;
    }
}

async function getBizraStats() {
    const stats = { ...FALLBACK_STATS };

    const live = await fetchLiveStats();
    if (live && live.kernel && live.kernel.alive) {
        stats.kernel_alive = true;
        stats.uptime_s = live.kernel.uptime_s || 0;
        stats.seed_balance = live.seed?.balance || 0;
        stats.total_missions = live.seed?.total_missions || 0;
        stats.knowledge_entries = live.urp?.knowledge_entries || 0;
        stats.receipts_recorded = live.urp?.receipts || 0;
        stats.cpu = live.hardware?.cpu || 'Unknown';
        stats.ram_gb = live.hardware?.ram_gb || 0;
        stats.gpu = live.hardware?.gpu || 'Unknown';
    }

    return stats;
}

// Auto-update DOM elements with data-bizra attributes
async function updateLiveData() {
    const stats = await getBizraStats();

    document.querySelectorAll('[data-bizra]').forEach(el => {
        const key = el.dataset.bizra;
        if (key in stats) {
            const value = stats[key];
            if (typeof value === 'number' && value > 1000) {
                el.textContent = value.toLocaleString();
            } else {
                el.textContent = value;
            }
        }
    });

    // Update status indicator
    const indicator = document.getElementById('kernel-status');
    if (indicator) {
        if (stats.kernel_alive) {
            indicator.textContent = '● SOVEREIGN';
            indicator.style.color = '#34D399';
        } else {
            indicator.textContent = '○ OFFLINE';
            indicator.style.color = '#6B7280';
        }
    }
}

// Poll every 30 seconds
function startLiveUpdates() {
    updateLiveData();
    setInterval(updateLiveData, 30000);
}

// Auto-start when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', startLiveUpdates);
} else {
    startLiveUpdates();
}

// Export for manual use
window.BIZRA = { getBizraStats, updateLiveData, KERNEL_URL };
