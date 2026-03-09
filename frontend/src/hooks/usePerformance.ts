/**
 * Web Vitals performance monitoring hook.
 * Reports FCP, LCP, CLS, FID, TTFB to console in dev, beacon in prod.
 */

import { useEffect } from 'react';

interface Metric {
  name: string;
  value: number;
  rating: 'good' | 'needs-improvement' | 'poor';
}

const THRESHOLDS: Record<string, [number, number]> = {
  FCP: [1800, 3000],
  LCP: [2500, 4000],
  FID: [100, 300],
  CLS: [0.1, 0.25],
  TTFB: [800, 1800],
};

function rate(name: string, value: number): Metric['rating'] {
  const t = THRESHOLDS[name];
  if (!t) return 'good';
  return value <= t[0] ? 'good' : value <= t[1] ? 'needs-improvement' : 'poor';
}

export function usePerformance(enabled = true): void {
  useEffect(() => {
    if (!enabled) return;

    import('web-vitals').then(({ onFCP, onLCP, onCLS, onFID, onTTFB }) => {
      const report = (metric: { name: string; value: number }) => {
        const m: Metric = {
          name: metric.name,
          value: Math.round(metric.value * 100) / 100,
          rating: rate(metric.name, metric.value),
        };

        if (import.meta.env.DEV) {
          const style = m.rating === 'good' ? 'color:#34D399' : m.rating === 'poor' ? 'color:#F87171' : 'color:#FBBF24';
          console.log(`%c[Vitals] ${m.name}: ${m.value} (${m.rating})`, style);
        }

        // Production: beacon to /v1/metrics/vitals
        if (import.meta.env.PROD && navigator.sendBeacon) {
          const payload = new Blob([JSON.stringify(m)], {
            type: 'application/json',
          });
          navigator.sendBeacon('/v1/metrics/vitals', payload);
        }
      };

      onFCP(report);
      onLCP(report);
      onCLS(report);
      onFID(report);
      onTTFB(report);
    }).catch(() => {
      // web-vitals not available — non-critical
    });
  }, [enabled]);
}
