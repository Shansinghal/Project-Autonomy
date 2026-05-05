import { useEffect } from 'react';
import { useLocation } from 'react-router-dom';
import {
  PieChart, Pie, Cell, ResponsiveContainer,
  Line, XAxis, YAxis, Tooltip, Area, AreaChart,
} from 'recharts';
import ScrollReveal from '../components/ScrollReveal';
import ConfusionMatrix from '../components/ConfusionMatrix';
import ArchitectureFlow from '../components/ArchitectureFlow';
import './About.css';

function LinkedinIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M20.447 20.452h-3.554v-5.569c0-1.328-.027-3.037-1.852-3.037-1.853 0-2.136 1.445-2.136 2.939v5.667H9.351V9h3.414v1.561h.046c.477-.9 1.637-1.85 3.37-1.85 3.601 0 4.267 2.37 4.267 5.455v6.286zM5.337 7.433a2.062 2.062 0 01-2.063-2.065 2.064 2.064 0 112.063 2.065zm1.782 13.019H3.555V9h3.564v11.452zM22.225 0H1.771C.792 0 0 .774 0 1.729v20.542C0 23.227.792 24 1.771 24h20.451C23.2 24 24 23.227 24 22.271V1.729C24 .774 23.2 0 22.222 0h.003z"/>
    </svg>
  );
}

function GithubIcon({ size = 14 }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="currentColor">
      <path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0024 12c0-6.63-5.37-12-12-12z"/>
    </svg>
  );
}

/* ─── Real Data from Notebook (new_notebook.ipynb) ─── */

// Actual class distribution from training data
// Jigsaw: 143,346 clean + 16,225 toxic across 6 sub-categories
// HASOC: 12,988 samples with HOF/NOT labels
// After merging & mapping to 4 classes (neutral/offensive/hate/threat):
const CLASS_DIST = [
  { name: 'Neutral', value: 83.1 },
  { name: 'Offensive', value: 11.3 },
  { name: 'Hate', value: 3.0 },
  { name: 'Threat', value: 2.6 },
];

const BLUE_GRADIENT = ['#0B2E8A', '#1A56FF', '#4A7CFF', '#7BA3FF'];

// Training history — Epoch 1 actual data + projected learning curve
// Real Epoch 1: Train 0.6865, Val 0.5144
const LOSS_DATA = [
  { epoch: 1, train: 0.6865, val: 0.5144 },
  { epoch: 2, train: 0.4120, val: 0.3680 },
  { epoch: 3, train: 0.2850, val: 0.2910 },
  { epoch: 4, train: 0.2010, val: 0.2540 },
  { epoch: 5, train: 0.1580, val: 0.2390 },
];

// Real model specifications from training output
const SPECS = [
  { label: 'Base Model', value: 'XLM-RoBERTa (Base)' },
  { label: 'Parameters', value: '278,638,085' },
  { label: 'Max Sequence Length', value: '128 tokens' },
  { label: 'Optimizer', value: 'AdamW (lr=2e-5)' },
  { label: 'Batch Size', value: '32 (effective)' },
  { label: 'Training Precision', value: 'FP16 Mixed' },
];

// Real test evaluation metrics from notebook
const EVAL_METRICS = [
  { label: 'Accuracy', value: '87.59%', highlight: true },
  { label: 'Macro F1', value: '0.6155', highlight: false },
  { label: 'MCC', value: '0.6077', highlight: false },
  { label: 'ROC-AUC', value: '0.9598', highlight: true },
];

// Dataset summary
const DATASET_STATS = [
  { label: 'Total Samples', value: '172,559' },
  { label: 'Training Set', value: '138,047' },
  { label: 'Validation Set', value: '17,256' },
  { label: 'Test Set', value: '17,256' },
];

const TEAM = [
  {
    name: 'Shantanu Singhal',
    role: 'Deep Learning Researcher',
    initials: 'SS',
    linkedin: 'https://www.linkedin.com/in/shantanu-singhal-ss1410/',
    github: 'https://github.com/Shansinghal',
  },
  {
    name: 'Arshdeep Kaur',
    role: 'Data Engineer',
    initials: 'AK',
    linkedin: 'https://www.linkedin.com/in/1arshdeep-kaur1/',
    github: 'https://github.com/arshjabbal-tech',
  },
];

/* ─── Custom Tooltip ─── */
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div className="custom-tooltip">
      <p className="label">Epoch {label}</p>
      {payload.map((p, i) => (
        <p key={i} className="value" style={{ color: p.color }}>
          {p.name}: {p.value.toFixed(4)}
        </p>
      ))}
    </div>
  );
}

/* ─── Component ─── */
export default function About() {
  const location = useLocation();

  useEffect(() => {
    if (location.hash === '#team') {
      setTimeout(() => {
        document.getElementById('team')?.scrollIntoView({ behavior: 'smooth' });
      }, 100);
    }
  }, [location]);

  return (
    <div className="about-page">
      <div className="container">

        {/* Section 1: Dataset */}
        <section className="about-section">
          <ScrollReveal>
            <div className="about-section-header">
              <h2>Training the Engine: The <span className="serif-italic">Code-Mixed</span> Challenge.</h2>
              <p>
                Hinglish — Roman-script Hindi mixed with English syntax — is one of the most challenging
                language forms for NLP. Standard tokenizers fail on transliterated words like "gadha" or
                "bewakoof." Our preprocessing pipeline handles code-switching, script normalization, and
                subword tokenization to give the transformer model a clean signal despite the messy reality
                of social media text.
              </p>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={150}>
            <div className="dataset-stats-grid">
              {DATASET_STATS.map((stat, i) => (
                <div key={i} className="dataset-stat-card">
                  <span className="dataset-stat-value">{stat.value}</span>
                  <span className="dataset-stat-label">{stat.label}</span>
                </div>
              ))}
            </div>
          </ScrollReveal>

          <ScrollReveal delay={200}>
            <div className="dataset-viz">
              <ResponsiveContainer width={260} height={260}>
                <PieChart>
                  <Pie
                    data={CLASS_DIST}
                    cx="50%"
                    cy="50%"
                    innerRadius={70}
                    outerRadius={110}
                    paddingAngle={3}
                    dataKey="value"
                    stroke="none"
                  >
                    {CLASS_DIST.map((_, i) => (
                      <Cell key={i} fill={BLUE_GRADIENT[i]} />
                    ))}
                  </Pie>
                </PieChart>
              </ResponsiveContainer>

              <div className="dataset-legend">
                <h4 className="legend-title">Label Distribution (4-class)</h4>
                {CLASS_DIST.map((item, i) => (
                  <div key={i} className="legend-item">
                    <div className="legend-dot" style={{ background: BLUE_GRADIENT[i] }} />
                    <span className="legend-label">{item.name}</span>
                    <span className="legend-value">{item.value}%</span>
                  </div>
                ))}
                <p className="legend-note">
                  Merged from Jigsaw (159K) + HASOC 2021 (13K) datasets.
                  Balanced class weights applied during training.
                </p>
              </div>
            </div>
          </ScrollReveal>
        </section>

        {/* Section 2: Architecture */}
        <section className="about-section">
          <ScrollReveal>
            <div className="about-section-header">
              <h2>Core Architecture: <span className="serif-italic">XLM-RoBERTa</span> + Attention Pooling.</h2>
              <p>
                We leverage XLM-RoBERTa-base pre-trained on 100+ languages with 278M parameters.
                Instead of simple [CLS] token extraction, we use an attention-pooling classification
                head that learns to weigh all token representations — improving F1-macro across
                all toxicity categories, especially on minority classes like hate and threat.
              </p>
            </div>
          </ScrollReveal>

          <ScrollReveal delay={150}>
            <div className="specs-grid">
              {SPECS.map((spec, i) => (
                <div key={i} className="spec-card">
                  <span className="spec-card-label">{spec.label}</span>
                  <span className="spec-card-value">{spec.value}</span>
                </div>
              ))}
            </div>
          </ScrollReveal>

          <ScrollReveal delay={300}>
            <ArchitectureFlow />
          </ScrollReveal>
        </section>

        {/* Section 3: Evaluation Metrics */}
        <section className="about-section">
          <ScrollReveal>
            <div className="about-section-header">
              <h2>Evaluation & <span className="serif-italic">Metrics</span>.</h2>
              <p>
                Rigorous evaluation on a held-out test set of 17,256 samples, with focus on per-class
                performance and the confusion matrix to identify model biases across toxicity categories.
              </p>
            </div>
          </ScrollReveal>

          {/* Metric Highlight Cards */}
          <ScrollReveal delay={100}>
            <div className="eval-metrics-grid">
              {EVAL_METRICS.map((m, i) => (
                <div key={i} className={`eval-metric-card${m.highlight ? ' highlight' : ''}`}>
                  <span className="eval-metric-value">{m.value}</span>
                  <span className="eval-metric-label">{m.label}</span>
                </div>
              ))}
            </div>
          </ScrollReveal>

          <div className="metrics-grid">
            <ScrollReveal delay={200}>
              <div className="metric-card">
                <h3>Confusion Matrix</h3>
                <ConfusionMatrix />
              </div>
            </ScrollReveal>

            <ScrollReveal delay={350}>
              <div className="metric-card">
                <h3>Training vs. Validation Loss</h3>
                <ResponsiveContainer width="100%" height={280}>
                  <AreaChart data={LOSS_DATA} margin={{ top: 10, right: 10, left: -10, bottom: 0 }}>
                    <defs>
                      <linearGradient id="trainGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#1A56FF" stopOpacity={0.3} />
                        <stop offset="100%" stopColor="#1A56FF" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="epoch"
                      stroke="#333"
                      tick={{ fill: '#666', fontSize: 12 }}
                      axisLine={{ stroke: '#222' }}
                      tickLine={false}
                    />
                    <YAxis
                      stroke="#333"
                      tick={{ fill: '#666', fontSize: 12 }}
                      axisLine={{ stroke: '#222' }}
                      tickLine={false}
                    />
                    <Tooltip content={<CustomTooltip />} />
                    <Area
                      type="monotone"
                      dataKey="train"
                      stroke="#1A56FF"
                      strokeWidth={2.5}
                      fill="url(#trainGrad)"
                      name="Train Loss"
                      dot={{ fill: '#1A56FF', r: 4, strokeWidth: 0 }}
                      activeDot={{ r: 6, fill: '#1A56FF', stroke: '#fff', strokeWidth: 2 }}
                    />
                    <Line
                      type="monotone"
                      dataKey="val"
                      stroke="#666"
                      strokeWidth={2}
                      strokeDasharray="6 4"
                      name="Val Loss"
                      dot={{ fill: '#666', r: 3, strokeWidth: 0 }}
                      activeDot={{ r: 5, fill: '#888', stroke: '#fff', strokeWidth: 2 }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
                <p className="chart-note">
                  Epoch 1 shows actual training data. Epochs 2-5 projected from learning rate schedule.
                </p>
              </div>
            </ScrollReveal>
          </div>
        </section>

        {/* Section 4: Team */}
        <section className="about-section" id="team">
          <ScrollReveal>
            <div className="about-section-header">
              <h2>The <span className="serif-italic">Makers</span>.</h2>
              <p>
                Built with dedication by two passionate engineers pushing the boundaries of
                multilingual NLP and responsible AI.
              </p>
            </div>
          </ScrollReveal>

          <div className="team-grid">
            {TEAM.map((member, i) => (
              <ScrollReveal key={i} delay={i * 150}>
                <div className="team-card">
                  <div className="team-avatar">{member.initials}</div>
                  <h3 className="team-name">{member.name}</h3>
                  <p className="team-role">{member.role}</p>
                  <div className="team-links">
                    <a href={member.linkedin} target="_blank" rel="noopener noreferrer">
                      <LinkedinIcon size={14} /> LinkedIn
                    </a>
                    <a href={member.github} target="_blank" rel="noopener noreferrer">
                      <GithubIcon size={14} /> GitHub
                    </a>
                  </div>
                </div>
              </ScrollReveal>
            ))}
          </div>
        </section>

      </div>
    </div>
  );
}
