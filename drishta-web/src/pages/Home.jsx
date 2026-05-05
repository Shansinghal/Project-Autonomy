import { Link } from 'react-router-dom';
import { ArrowRight } from 'lucide-react';
import OrbitalHero from '../components/OrbitalHero';
import ScrollReveal from '../components/ScrollReveal';
import './Home.css';

export default function Home() {
  return (
    <>
      <div className="home-bg-grid" />
      <section className="home-hero">
        <div className="container home-hero-inner">
          <div className="home-hero-content">
            <ScrollReveal>
              <h1 className="home-hero-title">
                The <span className="serif-italic">Toxicity</span> Classifier.
              </h1>
            </ScrollReveal>

            <ScrollReveal delay={150}>
              <p className="home-hero-subtitle">
                Your scalable system for <em>moderating code-mixed social content</em>.
              </p>
            </ScrollReveal>

            <ScrollReveal delay={300}>
              <p className="home-hero-description">
                Social platforms struggle to moderate toxic, hate-speech, and threatening content at scale —
                especially in Hinglish. We built a synchronized deep learning engine using mBERT/XLM-RoBERTa
                that aligns detection and explainability to turn raw text into a safer ecosystem.
              </p>
            </ScrollReveal>

            <ScrollReveal delay={400}>
              <div className="home-hero-cta">
                <Link to="/playground" className="btn btn-primary">
                  Test the Model <ArrowRight size={17} />
                </Link>
                <Link to="/about" className="btn btn-outline">
                  Learn More
                </Link>
              </div>
            </ScrollReveal>

            <ScrollReveal delay={550}>
              <div className="home-stats">
                <div className="home-stat">
                  <span className="home-stat-value">278M</span>
                  <span className="home-stat-label">Parameters</span>
                </div>
                <div className="home-stat">
                  <span className="home-stat-value">172K+</span>
                  <span className="home-stat-label">Training Samples</span>
                </div>
                <div className="home-stat">
                  <span className="home-stat-value">87.6%</span>
                  <span className="home-stat-label">Test Accuracy</span>
                </div>
                <div className="home-stat">
                  <span className="home-stat-value">4</span>
                  <span className="home-stat-label">Toxicity Classes</span>
                </div>
              </div>
            </ScrollReveal>
          </div>

          <div className="home-hero-visual">
            <OrbitalHero />
          </div>
        </div>
      </section>
    </>
  );
}
