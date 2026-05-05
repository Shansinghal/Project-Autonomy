import './OrbitalHero.css';

const crosses = [
  { top: '8%', left: '42%' },
  { top: '18%', right: '5%' },
  { top: '75%', left: '8%' },
  { top: '85%', right: '12%' },
  { top: '45%', left: '2%' },
  { top: '30%', right: '2%' },
];

export default function OrbitalHero() {
  return (
    <div className="orbital-container">
      <div className="orbital-scene">
        {/* Decorative crosses */}
        {crosses.map((pos, i) => (
          <span key={i} className="orbital-cross" style={pos}>+</span>
        ))}

        {/* Concentric rings */}
        <div className="orbital-ring orbital-ring-1" />
        <div className="orbital-ring orbital-ring-2" />
        <div className="orbital-ring orbital-ring-3" />

        {/* Central core */}
        <div className="orbital-core">
          <div className="orbital-core-glow" />
          <div className="orbital-core-shape" />
        </div>

        {/* Orbiting nodes */}
        <div className="orbital-node-wrapper orbital-node-wrapper-1">
          <div className="orbital-node">
            <div className="orbital-node-dot" />
            <span className="orbital-node-label">Hate Speech</span>
          </div>
        </div>

        <div className="orbital-node-wrapper orbital-node-wrapper-2">
          <div className="orbital-node">
            <div className="orbital-node-dot" />
            <span className="orbital-node-label">Offensive</span>
          </div>
        </div>

        <div className="orbital-node-wrapper orbital-node-wrapper-3">
          <div className="orbital-node">
            <div className="orbital-node-dot" />
            <span className="orbital-node-label">Threat</span>
          </div>
        </div>
      </div>
    </div>
  );
}
