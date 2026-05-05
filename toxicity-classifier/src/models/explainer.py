import torch
import numpy as np
import shap

class ExplainerUtil:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def extract_attention_weights(self, text):
        """
        Extracts attention weights from the last layer of the transformer,
        averaged across all attention heads.
        """
        self.model.eval()
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            logits, attentions = self.model(**inputs)
            
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]
        predicted_class = np.argmax(probs)
        confidence = probs[predicted_class]
        
        # Attentions is a tuple of layers. Taking the last layer.
        # Shape of last layer: [batch_size, num_heads, seq_len, seq_len]
        last_layer_attention = attentions[-1][0] 
        
        # Average across the num_heads dimension
        avg_attention = torch.mean(last_layer_attention, dim=0).cpu().numpy()
        
        # We find how much attention the [CLS] token (idx 0) pays to each token
        cls_attention = avg_attention[0]
        
        # Extract tokens for alignment
        input_ids = inputs['input_ids'][0].cpu().numpy()
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids)
        
        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "tokens": tokens,
            "attention_scores": cls_attention,
            "probs": probs
        }

    def build_shap_explainer(self, model_predict_fn):
        """
        Builds a customized SHAP explainer for the text model. 
        model_predict_fn must wrap the model such that it takes a list of strings 
        and returns numpy array of probabilities.
        """
        explainer = shap.Explainer(model_predict_fn, self.tokenizer)
        return explainer
