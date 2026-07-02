import os
import re

scripts = [
    'generate_bar_graphs.py',
    'generate_requested_graphs.py',
    'generate_confusion_matrix.py',
    'draw_architectures.py',
    'generate_sensitivity_2feats.py'
]

rc_params_code = """
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 32
})
"""

for script in scripts:
    if not os.path.exists(script):
        print(f"Skipping {script}, not found.")
        continue
        
    with open(script, 'r') as f:
        content = f.read()

    # 1. Update output directory to 2x
    content = content.replace("out_dir = '../report'", "out_dir = '../report/2x'")
    content = content.replace("'../report/", "'../report/2x/")
    content = content.replace('"../report/', '"../report/2x/')
    
    # 2. Double explicit fontsizes and labelsizes
    def double_size(match):
        prefix = match.group(1)
        size = int(match.group(2))
        return f"{prefix}{size * 2}"
        
    content = re.sub(r'(fontsize=)(\d+)', double_size, content)
    content = re.sub(r'(labelsize=)(\d+)', double_size, content)
    content = re.sub(r'(size=\')(\d+)', double_size, content) # e.g. size='12'
    content = re.sub(r"({'size': )(\d+)", double_size, content)
    
    # 3. Inject rcParams if not already there
    if 'plt.rcParams.update' not in content:
        # insert after import matplotlib.pyplot as plt
        if 'import matplotlib.pyplot as plt' in content:
            content = content.replace('import matplotlib.pyplot as plt', rc_params_code)
            
    with open(script, 'w') as f:
        f.write(content)
    
    print(f"Updated {script}")

# Also make the 2x directory
os.makedirs('../report/2x', exist_ok=True)
print("Created ../report/2x directory")
