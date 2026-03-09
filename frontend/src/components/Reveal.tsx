import { useEffect, useState } from 'react';
import type { CSSProperties, ReactNode } from 'react';

interface RevealProps {
  children: ReactNode;
  delay?: number;
  style?: CSSProperties;
}

export function Reveal({ children, delay: d = 0, style = {} }: RevealProps) {
  const [vis, setVis] = useState(false);

  useEffect(() => {
    const t = setTimeout(() => setVis(true), d);
    return () => clearTimeout(t);
  }, [d]);

  return (
    <div
      style={{
        opacity: vis ? 1 : 0,
        transform: vis ? 'translateY(0)' : 'translateY(12px)',
        transition:
          'opacity .7s cubic-bezier(.16,1,.3,1), transform .7s cubic-bezier(.16,1,.3,1)',
        ...style,
      }}
    >
      {children}
    </div>
  );
}
