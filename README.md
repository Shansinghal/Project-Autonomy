# Multilingual Toxicity Classifier

Project Autonomy is an end-to-end machine learning project aimed at detecting toxic content across multiple languages, specifically targeting English, Hindi, and Hinglish. It features a robust, fine-tuned transformer model coupled with an intuitive React-based web interface named **DRISHTA**.

## Project Overview

The project is split into two primary components:
1. **Toxicity Classifier (`toxicity-classifier/`)**: A production-ready machine learning pipeline for training and evaluating a multilingual toxicity classification model using `xlm-roberta-base`. It includes data preprocessing, PyTorch training loops with mixed-precision, and SHAP-based explainability modules.
2. **DRISHTA Web App (`drishta-web/`)**: A dark-themed, high-performance React application that serves as the user-facing frontend. It includes a Home page, an interactive Playground to test the model in real-time, and an About page detailing the project's background and metrics.

## Datasets

The model was trained on a heavily curated dataset comprising multiple sources to ensure robustness and multilingual support:

1. **Jigsaw Toxic Comment Classification Challenge**:
   - **Size**: 159,571 samples
   - **Description**: A large-scale English dataset containing Wikipedia talk page edits.

2. **HASOC 2021 (Hate Speech and Offensive Content)**:
   - **Size**: 12,988 samples
   - **Description**: Multilingual dataset containing posts in English, Hindi, and conversational Hinglish.

### Merged Data Distribution
The unified dataset maps diverse labels into 4 distinct categories: `neutral` (0), `offensive` (1), `hate` (2), and `threat` (3).

- **Train Split**: 138,047 samples
- **Validation Split**: 17,256 samples
- **Test Split**: 17,256 samples

## Model Architecture & Training

The core of the system is the **`xlm-roberta-base`** model, chosen for its exceptional cross-lingual capabilities.

- **Base Model**: `xlm-roberta-base`
- **Trainable Parameters**: ~278.6 Million
- **Max Sequence Length**: 128 tokens
- **Training Setup**: PyTorch with mixed-precision (FP16), AdamW optimizer, and gradient accumulation.
- **Explainability**: Integrated SHAP (SHapley Additive exPlanations) and custom Attention-based visualizations to highlight toxic spans within the input text. The model provides a confidence score alongside probabilities for each category (`neutral`, `offensive`, `hate`, `threat`), allowing granular insight into its decision-making process.

## Evaluation Metrics

The model was evaluated on the unseen test set (17,256 samples) and achieved the following performance metrics:

- **Accuracy**: 87.59% (0.8759)
- **Macro F1-Score**: 61.55% (0.6155)
- **MCC (Matthews Correlation Coefficient)**: 0.6077
- **ROC-AUC (Macro)**: 95.98% (0.9598)

## Tech Stack

- **Machine Learning**: PyTorch, Transformers (Hugging Face), scikit-learn, SHAP, pandas, numpy
- **Web Development**: React, Vite, CSS (Modern dark-mode UI with glassmorphism)
- **Environment**: Google Colab (for GPU acceleration), Node.js

## Getting Started

### Prerequisites
- Node.js (for the frontend)
- Python 3.8+ & PyTorch (for the backend/model)

### Running the Web App (DRISHTA)
```bash
cd drishta-web
npm install
npm run dev
```

### Model Training & Evaluation
The training scripts and notebooks are located in the `toxicity-classifier/` directory. Due to computational requirements, executing the notebooks (`notebooks/`) on Google Colab with GPU support is highly recommended.
