import { useRef, useEffect, CSSProperties } from 'react';

interface PrimordialBloomProps {
  size?: number;
  seed?: number;
  style?: CSSProperties;
}

const PHI = (1 + Math.sqrt(5)) / 2;
const GOLDEN_ANGLE = 2 * Math.PI / (PHI * PHI);

export function PrimordialBloom({ size = 300, seed = 42, style }: PrimordialBloomProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = size * dpr;
    canvas.height = size * dpr;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    const cx = size / 2;
    const cy = size / 2;
    const maxR = size * 0.42;
    const count = 200 + seed;

    ctx.clearRect(0, 0, size, size);

    for (let i = 0; i < count; i++) {
      const angle = i * GOLDEN_ANGLE;
      const r = maxR * Math.sqrt(i / count);
      const x = cx + r * Math.cos(angle);
      const y = cy + r * Math.sin(angle);
      const dotR = 0.8 + (i / count) * 1.5;
      const alpha = 0.15 + (1 - i / count) * 0.35;

      ctx.beginPath();
      ctx.arc(x, y, dotR, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(96, 165, 250, ${alpha})`;
      ctx.fill();
    }
  }, [size, seed]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: size, height: size, ...style }}
    />
  );
}

export default PrimordialBloom;
