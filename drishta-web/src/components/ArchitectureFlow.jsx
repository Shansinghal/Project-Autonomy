import { Type, Cpu, Layers, BarChart3, Lightbulb } from 'lucide-react';
import './ArchitectureFlow.css';

const STEPS = [
  { icon: Type, label: 'Raw Hinglish Text' },
  { icon: Cpu, label: 'Tokenizer' },
  { icon: Layers, label: 'Transformer Layers' },
  { icon: BarChart3, label: 'Classification Head' },
  { icon: Lightbulb, label: 'Explainability' },
];

export default function ArchitectureFlow() {
  return (
    <div className="arch-flow">
      {STEPS.map((step, i) => (
        <div key={i} style={{ display: 'flex', alignItems: 'center' }}>
          <div className="arch-node">
            <div className="arch-node-circle">
              <step.icon size={22} />
            </div>
            <span className="arch-node-label">{step.label}</span>
          </div>
          {i < STEPS.length - 1 && <div className="arch-connector" />}
        </div>
      ))}
    </div>
  );
}
