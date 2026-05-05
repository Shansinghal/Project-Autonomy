import json, sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

with open('new_notebook.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

out_lines = []
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        outputs_text = ''
        for out in cell.get('outputs', []):
            if out.get('output_type') == 'stream':
                outputs_text += ''.join(out.get('text', []))[:1200]
            elif out.get('output_type') == 'execute_result':
                data = out.get('data', {})
                if 'text/plain' in data:
                    outputs_text += ''.join(data['text/plain'])[:1200]
        out_lines.append(f"=== CODE CELL {i} ===")
        out_lines.append(source)
        if outputs_text.strip():
            out_lines.append("--- OUTPUT ---")
            out_lines.append(outputs_text[:2000])
        out_lines.append("")
    elif cell['cell_type'] == 'markdown':
        source = ''.join(cell['source'])
        out_lines.append(f"=== MARKDOWN CELL {i} ===")
        out_lines.append(source[:500])
        out_lines.append("")

with open('cells_output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out_lines))
print("Done")
