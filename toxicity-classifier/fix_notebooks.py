import json
from pathlib import Path
import os

notebooks_dir = Path("c:/Users/Shant/OneDrive/Desktop/Project-Autonomy/toxicity-classifier/notebooks")

setup_code = [
    "import sys\n",
    "import os\n",
    "# Ensure custom 'src' module can be found both locally and on Kaggle\n",
    "if os.path.abspath('..') not in sys.path:\n",
    "    sys.path.insert(0, os.path.abspath('..'))\n",
    "sys.path.append('/kaggle/working/toxicity-classifier/')\n"
]

setup_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": setup_code
}

for nb_file in notebooks_dir.glob("*.ipynb"):
    with open(nb_file, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # Identify if it has already been inserted
    already_inserted = any("os.path.abspath('..')" in "".join(c.get('source', [])) for c in nb['cells'] if c['cell_type'] == 'code')
    
    if not already_inserted:
        # Insert as the second cell (right after the first markdown title)
        nb['cells'].insert(1, setup_cell)
        
    # Remove the hardcoded sys.path.append from all other code cells
    for cell in nb['cells']:
        if cell['cell_type'] == 'code' and "os.path.abspath" not in "".join(cell.get('source', [])):
            cell['source'] = [line for line in cell['source'] if '/kaggle/working/toxicity-classifier/' not in line]
            
    with open(nb_file, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=2)

print("Fixed local paths in all notebooks.")
