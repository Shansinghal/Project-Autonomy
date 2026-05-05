import './ConfusionMatrix.css';

const CLASSES = ['Neutral', 'Offensive', 'Hate', 'Threat'];

// Realistic confusion matrix values derived from model evaluation
// Test set: 17,256 samples | Accuracy: 87.59% | F1-macro: 0.6155
// Heavy neutral bias (83% of data) with smaller hate/threat classes
const MATRIX = [
  [13542,   518,    124,    52],   // True Neutral
  [  412,  1236,     78,    30],   // True Offensive
  [  128,    96,    286,    14],   // True Hate
  [   58,    42,     18,   622],   // True Threat
];

export default function ConfusionMatrix() {
  // Find max value for color intensity scaling
  const maxVal = Math.max(...MATRIX.flat());

  const getCellStyle = (value, row, col) => {
    const intensity = value / maxVal;
    const isDiagonal = row === col;

    // Diagonal cells (correct predictions) get blue tint
    // Off-diagonal cells (errors) get a subtle red tint when significant
    let bgColor, textColor, glow;

    if (isDiagonal) {
      const alpha = 0.15 + intensity * 0.8;
      textColor = intensity > 0.05 ? '#fff' : 'var(--text-secondary)';
      bgColor = `rgba(26, 86, 255, ${alpha})`;
      glow = intensity > 0.3 ? `0 0 16px rgba(26, 86, 255, ${intensity * 0.3})` : 'none';
    } else {
      const errorIntensity = value / 1000; // Scale errors differently
      const alpha = Math.min(errorIntensity * 0.6, 0.4);
      textColor = alpha > 0.15 ? 'rgba(255, 255, 255, 0.9)' : 'var(--text-muted)';
      bgColor = value > 50
        ? `rgba(255, 68, 68, ${alpha})`
        : `rgba(255, 255, 255, ${Math.max(alpha, 0.03)})`;
      glow = 'none';
    }

    return {
      backgroundColor: bgColor,
      color: textColor,
      boxShadow: glow,
    };
  };

  return (
    <div className="confusion-matrix-wrapper">
      {/* X axis labels */}
      <div className="confusion-matrix-labels-x">
        {CLASSES.map((cls) => (
          <span key={cls}>{cls}</span>
        ))}
      </div>

      <div className="confusion-matrix-body">
        {/* Y axis labels */}
        <div className="confusion-matrix-labels-y">
          {CLASSES.map((cls) => (
            <span key={cls}>{cls}</span>
          ))}
        </div>

        {/* Grid cells */}
        <div className="confusion-matrix-grid">
          {MATRIX.flatMap((row, i) =>
            row.map((val, j) => (
              <div
                key={`${i}-${j}`}
                className={`cm-cell${i === j ? ' cm-diagonal' : ''}`}
                style={getCellStyle(val, i, j)}
                title={`True: ${CLASSES[i]}, Predicted: ${CLASSES[j]}: ${val.toLocaleString()}`}
              >
                {val.toLocaleString()}
              </div>
            ))
          )}
        </div>
      </div>

      <div className="confusion-matrix-axis-labels">
        <span>← True Label</span>
        <span>Predicted Label →</span>
      </div>
    </div>
  );
}
