interface AgentNode {
  color: string;
  booted?: boolean;
}

interface AstrolabeSVGProps {
  size?: number;
  agents?: AgentNode[];
  active?: boolean;
}

export function AstrolabeSVG({ size = 200, agents = [], active = false }: AstrolabeSVGProps) {
  const cx = size / 2;
  const cy = size / 2;
  const r = size * 0.38;

  const pts = agents.map((_, i) => {
    const angle = (i * 2 * Math.PI) / agents.length - Math.PI / 2;
    return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ overflow: 'visible' }}>
      {/* Outer ring */}
      <circle cx={cx} cy={cy} r={r + 12} fill="none" stroke="rgba(201,169,98,.08)" strokeWidth=".5" />
      <circle
        cx={cx} cy={cy} r={r}
        fill="none" stroke="rgba(201,169,98,.15)" strokeWidth="1"
        strokeDasharray="3 6"
        style={active ? { animation: 'spinSlow 60s linear infinite', transformOrigin: `${cx}px ${cy}px` } : {}}
      />
      <circle cx={cx} cy={cy} r={r - 12} fill="none" stroke="rgba(201,169,98,.06)" strokeWidth=".5" />

      {/* Connection lines */}
      {active &&
        pts.map((p, i) =>
          pts.slice(i + 1).map((q, j) => (
            <line key={`${i}-${j}`} x1={p.x} y1={p.y} x2={q.x} y2={q.y}
              stroke="rgba(201,169,98,.06)" strokeWidth=".5" />
          )),
        )}

      {/* Agent nodes */}
      {pts.map((p, i) => {
        const ag = agents[i];
        return (
          <g key={i}>
            <circle
              cx={p.x} cy={p.y} r={6}
              fill={ag?.booted ? `${ag.color}20` : 'transparent'}
              stroke={ag?.booted ? ag.color : 'rgba(255,255,255,.08)'}
              strokeWidth={ag?.booted ? 1.5 : 0.5}
              style={ag?.booted ? { filter: `drop-shadow(0 0 6px ${ag.color}40)` } : {}}
            />
            {ag?.booted && <circle cx={p.x} cy={p.y} r={2} fill={ag.color} />}
          </g>
        );
      })}

      {/* Center */}
      <circle cx={cx} cy={cy} r={4} fill="rgba(201,169,98,.15)" stroke="rgba(201,169,98,.3)" strokeWidth=".5" />
    </svg>
  );
}
