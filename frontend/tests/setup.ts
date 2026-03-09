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

// Mock import.meta.env
Object.defineProperty(import.meta, 'env', {
  value: {
    DEV: true,
    PROD: false,
    VITE_API_URL: '',
  },
});
