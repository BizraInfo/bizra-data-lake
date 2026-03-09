import { describe, it, expect } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { Reveal } from '../src/components/Reveal';
import { AstrolabeSVG } from '../src/components/AstrolabeSVG';

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
