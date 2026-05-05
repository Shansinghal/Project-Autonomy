import os
import pandas as pd
from pathlib import Path

class DatasetLoader:
    def __init__(self, base_path="/kaggle/input"):
        self.base_path = Path(base_path)

    def load_jigsaw(self, folder_name="jigsaw-toxic-comment-classification-challenge"):
        """
        Loads the Kaggle Jigsaw toxic comment challenge dataset.
        Columns: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate
        """
        path = self.base_path / folder_name / "train.csv"
        # If running locally, you can provide an alternate path or catch the error
        if not path.exists():
            print(f"[Warning] Jigsaw dataset not found at {path}")
            return pd.DataFrame()
            
        print(f"Loading Jigsaw dataset from {path}")
        return pd.read_csv(path)

    def load_hasoc_2021(self, folder_name="hasoc-2021"):
        """
        Loads the HASOC 2021 datasets (English, Hindi, Hinglish).
        """
        folder_path = self.base_path / folder_name
        if not folder_path.exists():
            print(f"[Warning] HASOC dataset not found at {folder_path}")
            return pd.DataFrame()
            
        dfs = []
        for file in folder_path.glob("*.tsv"):
            try:
                df = pd.read_csv(file, sep="\t")
                # Normalize column names
                df.columns = [c.lower().strip() for c in df.columns]
                # Infer language from filename heuristic
                lang = "en"
                name_lower = file.name.lower()
                if "hi" in name_lower or "hindi" in name_lower:
                    lang = "hi"
                df['lang_heuristic'] = lang
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
                
        print(f"Loading HASOC dataset from {folder_path}")
        if dfs:
            return pd.concat(dfs, ignore_index=True)
        return pd.DataFrame()

    def load_multimodal_hate_speech(self, folder_name="multimodal-hate-speech"):
        """
        Loads Multimodal Hate Speech. We only care about the text column.
        Columns: tweet, label
        """
        path = self.base_path / folder_name / "labeled_data.csv"
        if not path.exists():
            print(f"[Warning] Multimodal Hate Speech dataset not found at {path}")
            return pd.DataFrame()
            
        print(f"Loading Multimodal Hate Speech dataset from {path}")
        return pd.read_csv(path)
