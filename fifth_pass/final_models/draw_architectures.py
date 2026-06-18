import matplotlib.pyplot as plt
import os

def draw_nn_box(ax, layers, title, filename):
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(layers)*2 + 1)
    ax.axis('off')
    
    y = len(layers)*2 - 0.5
    for i, layer in enumerate(layers):
        # Draw box
        box = plt.Rectangle((2, y-1), 6, 1.5, fill=True, color='#00d2ff', alpha=0.3, ec='white', lw=2)
        ax.add_patch(box)
        # Draw text
        ax.text(5, y-0.25, layer, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        
        # Draw arrow to next layer
        if i < len(layers) - 1:
            ax.annotate('', xy=(5, y-1.5), xytext=(5, y-1), arrowprops=dict(facecolor='white', edgecolor='white', shrink=0, width=2, headwidth=8))
        
        y -= 2
        
    ax.set_title(title, fontsize=16, color='white')
    plt.tight_layout()
    plt.savefig(filename, dpi=300, facecolor='#1a1a1a')
    plt.close()

plt.style.use('dark_background')

# Win Model
win_layers = [
    "Input Layer\n(7 Differentials)",
    "Hidden Layer\n(32 units, Swish, L2=1e-4)",
    "Dropout Layer\n(Rate=0.2)",
    "Output Layer\n(1 unit, Sigmoid)"
]
fig, ax = plt.subplots(figsize=(6, 8))
draw_nn_box(ax, win_layers, "Win Prediction Neural Network", "win_model_arch.png")

# Score Model
score_layers = [
    "Input Layer\n(3 Differentials)",
    "Hidden Layer\n(16 units, Swish, L2=1e-3)",
    "Output Layer\n(1 unit, Linear)"
]
fig, ax = plt.subplots(figsize=(6, 6))
draw_nn_box(ax, score_layers, "Score Prediction Neural Network", "score_model_arch.png")

print("Architecture graphics generated successfully.")
