import pandas as pd
from sklearn.model_selection import train_test_split
from langdetect import detect, DetectorFactory

# Set seed for reproducible langdetect
DetectorFactory.seed = 42

class DataMerger:
    def __init__(self, config=None):
        """
        config should map strings to integers according to the unified schema
        e.g., config['labels'] = {'neutral': 0, 'offensive': 1, 'hate': 2, 'threat': 3}
        """
        self.label_map = config['labels'] if config else {'neutral': 0, 'offensive': 1, 'hate': 2, 'threat': 3}

    def _map_jigsaw(self, df):
        """Maps Jigsaw explicitly multi-label to our unified schema."""
        if df.empty: return pd.DataFrame()
        
        def reduce_labels(row):
            if row.get('threat', 0) == 1:
                return self.label_map['threat']
            elif row.get('identity_hate', 0) == 1 or row.get('severe_toxic', 0) == 1:
                return self.label_map['hate']
            elif row.get('toxic', 0) == 1 or row.get('obscene', 0) == 1 or row.get('insult', 0) == 1:
                return self.label_map['offensive']
            else:
                return self.label_map['neutral']
        
        df['unified_label'] = df.apply(reduce_labels, axis=1)
        df['unified_text'] = df['comment_text']
        df['source'] = 'jigsaw'
        df['language'] = 'en'
        
        return df[['unified_text', 'unified_label', 'language', 'source']]

    def _map_hasoc(self, df):
        """Maps HASOC task_2 (HATE/OFFN/PRFN/NONE) to unified schema."""
        if df.empty: return pd.DataFrame()
        
        def reduce_labels(row):
            task2 = str(row.get('task_2', 'NONE')).strip().upper()
            if task2 == 'HATE': return self.label_map['hate']
            if task2 in ['OFFN', 'PRFN']: return self.label_map['offensive']
            # Fallback if only task_1 is provided
            task1 = str(row.get('task_1', 'NOT')).strip().upper()
            if task1 == 'HOF' and task2 == 'NONE': return self.label_map['offensive'] 
            return self.label_map['neutral']

        text_col = 'text' if 'text' in df.columns else df.columns[1]
        df['unified_text'] = df[text_col]
        df['unified_label'] = df.apply(reduce_labels, axis=1)
        df['source'] = 'hasoc'
        df['language'] = df.get('lang_heuristic', 'en')
        
        return df[['unified_text', 'unified_label', 'language', 'source']]

    def _map_multimodal(self, df):
        """Maps Multimodal Hate Speech (0=neutral/offensive, 1=hate)."""
        if df.empty: return pd.DataFrame()
        
        # Assume class 1 is hate, 0 is neither, or check specific dataset docs
        def reduce_labels(row):
            lbl = row.get('class', -1)
            # Typically in multimodal hs labeled_data, 0: hate, 1: offensive, 2: neither
            if lbl == 0: return self.label_map['hate']
            if lbl == 1: return self.label_map['offensive']
            return self.label_map['neutral']
            
        df['unified_text'] = df['tweet']
        df['unified_label'] = df.apply(reduce_labels, axis=1)
        df['source'] = 'multimodal_hs'
        df['language'] = 'en'
        
        return df[['unified_text', 'unified_label', 'language', 'source']]

    def merge_and_split(self, jigsaw_df, hasoc_df, multi_df, test_size=0.1, val_size=0.1):
        """Takes raw DataFrames, maps them, merges them, and splits."""
        j_merged = self._map_jigsaw(jigsaw_df)
        h_merged = self._map_hasoc(hasoc_df)
        m_merged = self._map_multimodal(multi_df)
        
        final_df = pd.concat([j_merged, h_merged, m_merged], ignore_index=True)
        final_df = final_df.dropna(subset=['unified_text', 'unified_label'])
        
        # Detect lang using langdetect for texts where it's ambiguous
        # In a real environment, we'd only apply to non-English or sample. 
        # For simplicity, we keep original language definitions, or overwrite with detect() here.
        # final_df['language'] = final_df['unified_text'].apply(lambda x: detect(str(x)) if len(str(x)) > 5 else "unknown")

        # Stratified Split based on Label and Source (composite stratify key)
        final_df['strat_key'] = final_df['unified_label'].astype(str) + "_" + final_df['source']
        
        # Filter classes that have too few examples to split
        counts = final_df['strat_key'].value_counts()
        valid_keys = counts[counts > 5].index
        final_df = final_df[final_df['strat_key'].isin(valid_keys)]
        
        train_val, test_df = train_test_split(
            final_df, test_size=test_size, stratify=final_df['strat_key'], random_state=42
        )
        val_ratio = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val, test_size=val_ratio, stratify=train_val['strat_key'], random_state=42
        )
        
        # Drop the strat_key utility column
        for df in [train_df, val_df, test_df]:
            df.drop(columns=['strat_key'], inplace=True)
            df.rename(columns={'unified_text': 'text', 'unified_label': 'label'}, inplace=True)
            
        return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)
