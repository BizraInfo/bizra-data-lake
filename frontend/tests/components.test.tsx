import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { Reveal } from '../src/components/Reveal';
import { AstrolabeSVG } from '../src/components/AstrolabeSVG';
import { PrimordialBloom } from '../src/components/PrimordialBloom';

// eslint-disable-next-line @typescript-eslint/no-explicit-any
let getContextSpy: any;

beforeEach(() => {
  getContextSpy = vi.spyOn(HTMLCanvasElement.prototype, 'getContext').mockReturnValue({
    setTransform: vi.fn(),
    clearRect: vi.fn(),
    fillRect: vi.fn(),
    beginPath: vi.fn(),
    fill: vi.fn(),
    stroke: vi.fn(),
    moveTo: vi.fn(),
    lineTo: vi.fn(),
    arc: vi.fn(),
    save: vi.fn(),
    restore: vi.fn(),
    lineWidth: 1,
    fillStyle: '',
    strokeStyle: '',
    globalCompositeOperation: 'source-over',
  } as unknown as CanvasRenderingContext2D);
});

afterEach(() => {
  getContextSpy?.mockRestore();
  getContextSpy = undefined;
});

describe('Reveal', () => {
  it('renders children after delay', async () => {
    render(<Reveal delay={0}><span>visible</span></Reveal>);
    await waitFor(() => {
      const el = screen.getByText('visible');
      expect(el.parentElement).toHaveStyle({ opacity: '1' });
    });
  });

  it('starts hidden', () => {
    render(<Reveal delay={5000}><span>hidden</span></Reveal>);
    const el = screen.getByText('hidden');
    expect(el.parentElement).toHaveStyle({ opacity: '0' });
  });
});

describe('AstrolabeSVG', () => {
  it('renders SVG with correct dimensions', () => {
    const { container } = render(<AstrolabeSVG size={200} />);
    const svg = container.querySelector('svg');
    expect(svg).toBeTruthy();
    expect(svg?.getAttribute('width')).toBe('200');
    expect(svg?.getAttribute('height')).toBe('200');
  });

  it('renders agent nodes when provided', () => {
    const agents = [
      { color: '#60A5FA', booted: true },
      { color: '#34D399', booted: false },
      { color: '#F87171', booted: true },
    ];
    const { container } = render(<AstrolabeSVG size={180} agents={agents} active />);
    // 3 center circles for booted agents (inner dots)
    const circles = container.querySelectorAll('circle');
    // Outer ring (3) + agent circles (3 outer + 2 inner for booted) + center = 9
    expect(circles.length).toBeGreaterThanOrEqual(7);
  });

  it('renders connection lines when active', () => {
    const agents = [
      { color: '#60A5FA', booted: true },
      { color: '#34D399', booted: true },
    ];
    const { container } = render(<AstrolabeSVG size={100} agents={agents} active />);
    const lines = container.querySelectorAll('line');
    expect(lines.length).toBeGreaterThanOrEqual(1);
  });
});

describe('PrimordialBloom', () => {
  it('renders a canvas with the requested dimensions', () => {
    const { container } = render(<PrimordialBloom size={320} seed={7} />);
    const canvas = container.querySelector('canvas');
    expect(canvas).toBeTruthy();
    expect(canvas?.style.width).toBe('320px');
    expect(canvas?.style.height).toBe('320px');
  });
});
