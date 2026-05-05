import { useState } from 'react';
import { Zap, ShieldAlert, ShieldCheck, Cpu } from 'lucide-react';
import { predict } from '../utils/predictor';
import ScrollReveal from '../components/ScrollReveal';
import './Playground.css';

const EXAMPLES = [
  "I hate you, you are such a moron",
  "I am going to kill you, you stupid idiot",
  "Tu ek gadha hai, tujhse baat karna bekar hai",
  "This is a wonderful day to learn something new",
  "Go back to your country, you don't belong here",
  "Thank you for your help, I really appreciate it",
  "Shut the hell up you worthless piece of garbage",
  "तुम बहुत बेवकूफ हो, तुम्हें शर्म आनी चाहिए",
];

const CATEGORY_META = {
  neutral: { label: 'Neutral', className: 'neutral' },
  hate: { label: 'Hate Speech', className: 'hate' },
  offensive: { label: 'Offensive Language', className: 'offensive' },
  threat: { label: 'Threat', className: 'threat' },
};

export default function Playground() {
  const [text, setText] = useState('');
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleAnalyze = () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);

    // Simulate inference latency
    setTimeout(() => {
      const prediction = predict(text);
      setResult(prediction);
      setLoading(false);
    }, 600);
  };

  const handleExample = (example) => {
    setText(example);
    setResult(null);
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && e.ctrlKey) {
      handleAnalyze();
    }
  };

  return (
    <section className="playground">
      <div className="container">
        <ScrollReveal>
          <div className="playground-header">
            <h1>
              Interact with the <span className="serif-italic">Engine</span>.
            </h1>
            <p>Enter any text in English, Hindi, or Hinglish and see how the model classifies it.</p>
            <div className="model-badge">
              <Cpu size={14} />
              <span>Powered by XLM-RoBERTa · 278M params · 87.6% accuracy</span>
            </div>
          </div>
        </ScrollReveal>

        <ScrollReveal delay={150}>
          <div className="playground-input-section">
            <textarea
              id="playground-input"
              className="playground-textarea"
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Enter a social media post in English, Hindi, or Hinglish here..."
              rows={5}
            />

            <div className="playground-actions">
              <button
                className="btn btn-primary"
                onClick={handleAnalyze}
                disabled={!text.trim() || loading}
                id="analyze-btn"
              >
                {loading ? (
                  'Analyzing...'
                ) : (
                  <>
                    <Zap size={17} /> Analyze Text
                  </>
                )}
              </button>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                Ctrl + Enter
              </span>
            </div>

            <div className="playground-examples">
              <span className="playground-examples-label">Try an example</span>
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  className="example-chip"
                  onClick={() => handleExample(ex)}
                >
                  {ex.length > 45 ? ex.slice(0, 45) + '...' : ex}
                </button>
              ))}
            </div>
          </div>
        </ScrollReveal>

        {/* Results */}
        {result && (
          <div className="playground-results">
            <div className="results-grid">
              {/* Score Card */}
              <div className="score-card">
                <div className={`score-badge ${result.isToxic ? 'toxic' : 'safe'}`}>
                  {result.isToxic ? <ShieldAlert size={16} /> : <ShieldCheck size={16} />}
                  {result.isToxic ? 'Toxic' : 'Neutral'}
                </div>
                <div className="score-confidence">
                  {(result.confidence * 100).toFixed(1)}%
                </div>
                <div className="score-label-sub">
                  Classified as: {result.label.charAt(0).toUpperCase() + result.label.slice(1)}
                </div>
              </div>

              {/* Category Breakdown */}
              <div className="category-breakdown">
                {['neutral', 'offensive', 'hate', 'threat'].map((cat) => (
                  <div key={cat} className="category-bar-wrapper">
                    <div className="category-bar-header">
                      <span className="category-bar-label">{CATEGORY_META[cat].label}</span>
                      <span className="category-bar-value">
                        {(result.scores[cat] * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="category-bar-track">
                      <div
                        className={`category-bar-fill ${CATEGORY_META[cat].className}`}
                        style={{ width: `${result.scores[cat] * 100}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Explainability */}
            <div className="explainability-section">
              <h3>Explainability — Toxic Span Highlighting</h3>
              <p className="explainability-subtitle">
                Words highlighted in red are tokens the model identified as contributing to toxicity.
              </p>
              <div className="highlighted-text">
                {result.spans.map((span, i) => (
                  <span key={i}>
                    {span.toxic ? (
                      <span className="toxic-span">{span.text}</span>
                    ) : (
                      <span className="safe-span">{span.text}</span>
                    )}
                    {' '}
                  </span>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </section>
  );
}
