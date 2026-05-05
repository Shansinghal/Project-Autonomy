import json
from pathlib import Path

notebooks_dir = Path("c:/Users/Shant/OneDrive/Desktop/Project-Autonomy/toxicity-classifier/notebooks")

colab_setup_code = [
    "# Mount Google Drive for Colab Execution\n",
    "try:\n",
    "    from google.colab import drive\n",
    "    drive.mount('/content/drive')\n",
    "except ImportError:\n",
    "    print('Not running on Google Colab. Skipping drive mount.')\n",
    "\n"
]

for nb_file in notebooks_dir.glob("*.ipynb"):
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    for cell in nb['cells']:
        if cell['cell_type'] == 'code':
            new_source = []
            for line in cell.get('source', []):
                # Replace kaggle specific paths with standard Colab drive paths
                line = line.replace('/kaggle/working/toxicity-classifier/', '/content/drive/MyDrive/toxicity-classifier/')
                # Output saves (like train.csv)
                line = line.replace('/kaggle/working/', '/content/drive/MyDrive/toxicity-classifier/data/processed/')
                line = line.replace('/kaggle/input', '/content/drive/MyDrive/toxicity-classifier/data/raw')
                new_source.append(line)
            cell['source'] = new_source
            
            # Inject Drive mount specifically into the setup cell (which is cell idx 1 based on our previous fix)
            
    # Safely inject the drive mount code into our universal local path logic cell
    if nb['cells'][1]['cell_type'] == 'code':
        # Don't add twice
        if 'drive.mount' not in "".join(nb['cells'][1]['source']):
            nb['cells'][1]['source'] = colab_setup_code + nb['cells'][1]['source']

    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

print("Notebooks successfully optimized for Google Colab.")
