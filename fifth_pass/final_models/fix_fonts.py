import os

scripts = [
    'generate_bar_graphs.py',
    'generate_requested_graphs.py',
    'generate_confusion_matrix.py',
    'draw_architectures.py',
    'generate_sensitivity_2feats.py'
]

rc_params_code = """
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

for s in scripts:
    if not os.path.exists(s): continue
    with open(s, 'r') as f:
        content = f.read()

    # fix 2x/2x issue
    content = content.replace('../report/2x/2x', '../report/2x')
    
    # remove my old inserts if they exist
    old_rc = """plt.rcParams.update({
    'font.size': 20,
    'axes.titlesize': 28,
    'axes.labelsize': 24,
    'xtick.labelsize': 20,
    'ytick.labelsize': 20,
    'legend.fontsize': 20,
    'figure.titlesize': 32
})"""
    content = content.replace(old_rc, "")
    
    # remove any duplicate newlines left over
    content = content.replace("\n\n\n", "\n\n")

    # Find the place to inject rcParams: right after `plt.style.use` or `sns.set` if they exist.
    # Otherwise, right before the first function `def `
    if "plt.style.use('default')" in content:
        content = content.replace("plt.style.use('default')", "plt.style.use('default')\n" + rc_params_code)
    else:
        # inject before first def
        idx = content.find("def ")
        if idx != -1:
            content = content[:idx] + rc_params_code + "\n" + content[idx:]
        else:
            content += "\n" + rc_params_code

    with open(s, 'w') as f:
        f.write(content)
    print(f"Fixed {s}")
