import { vi } from 'vitest';
import '@testing-library/jest-dom/vitest';

// Mock web-vitals
vi.mock('web-vitals', () => ({
  onFCP: vi.fn(),
  onLCP: vi.fn(),
  onCLS: vi.fn(),
  onFID: vi.fn(),
  onTTFB: vi.fn(),
}));

// Stub HTMLCanvasElement.getContext for PrimordialBloom and other canvas components
vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
  setTransform: vi.fn(), clearRect: vi.fn(), fillRect: vi.fn(),
  beginPath: vi.fn(), fill: vi.fn(), stroke: vi.fn(),
  moveTo: vi.fn(), lineTo: vi.fn(), arc: vi.fn(),
  save: vi.fn(), restore: vi.fn(),
  lineWidth: 1, fillStyle: '', strokeStyle: '', globalCompositeOperation: 'source-over',
} as unknown as CanvasRenderingContext2D);

// Mock import.meta.env
Object.defineProperty(import.meta, 'env', {
  value: {
    DEV: true,
    PROD: false,
    VITE_API_URL: '',
  },
});
