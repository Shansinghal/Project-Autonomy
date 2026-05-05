import matplotlib.pyplot as plt
import seaborn as sns
import base64
from wordcloud import WordCloud
from IPython.display import HTML

def plot_confusion_matrix(cm, class_names):
    """Plots a normalized confusion matrix heatmap."""
    import numpy as np
    cm = np.array(cm)
    row_sums = cm.sum(axis=1, keepdims=True)
    # Avoid division by zero
    row_sums = np.where(row_sums == 0, 1, row_sums)
    cm_normalized = cm.astype('float') / row_sums
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Normalized Confusion Matrix')
    plt.show()

def plot_training_history(history):
    """Plots loss and macro F1 curves over epochs."""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, history['val_f1'], label='Val F1 (Macro)', marker='s', color='green')
    ax2.set_title('Validation F1 Score Iteration')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Macro F1')
    ax2.legend()
    ax2.grid(True)
    
    plt.show()

def generate_wordcloud(text_series, title):
    """Generates and plots a word cloud."""
    text = " ".join(text_series.astype(str).tolist())
    # Specify a font that supports Devanagari if needed, fallback to default
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title)
    plt.show()

def render_attention_html(tokens, attention_scores, title="Token Attention"):
    """
    Given a list of tokens and a list of attention scores (same length),
    returns an HTML string highlighting tokens based on probability.
    Red intensity corresponds to higher attention.
    """
    import numpy as np
    
    # Normalize scores to 0-1 range to handle varying attention scales gracefully
    mx = np.max(attention_scores) if len(attention_scores) > 0 else 1
    # Adding tiny epsilon to prevent div0
    mx = mx if mx > 0 else 1.0 
    
    html = f"<h3>{title}</h3><div style='font-size: 16px; line-height: 1.6; padding: 10px; border: 1px solid #ccc; border-radius: 5px; background-color: #fff;'>"
    
    for token, score in zip(tokens, attention_scores):
        if token in ['<s>', '</s>', '<pad>']:
            continue
            
        norm_score = score / mx
        # Calculate red color (255, 255-alpha, 255-alpha)
        alpha = int((1 - norm_score) * 255)
        color = f"rgb(255,{alpha},{alpha})"
        
        # Clean token presentation
        display_token = token.replace(' ', ' ') # XLM roberta underscores
        
        span = f"<span style='background-color: {color}; padding: 0.1em; margin: 0 0.1em; border-radius: 3px;'>{display_token}</span>"
        html += span
        
    html += "</div>"
    return HTML(html)
