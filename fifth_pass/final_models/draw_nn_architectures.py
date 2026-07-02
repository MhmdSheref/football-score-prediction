import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 18,
    'axes.titlesize': 24,
    'figure.titlesize': 32
})

def draw_neural_net(ax, layer_sizes, drawn_sizes, layer_labels, layer_types, features=None, title=''):
    ax.axis('off')
    
    n_layers = len(layer_sizes)
    xs = np.linspace(0.1, 0.9, n_layers)
    
    ax.set_xlim(-0.3, 1.1)
    ax.set_ylim(-7.5, 10.5)
    
    node_coords = []
    
    for i, (n, n_to_draw, l_type) in enumerate(zip(layer_sizes, drawn_sizes, layer_types)):
        
        if n_to_draw == 1:
            ys = [0]
        else:
            # Span from -5.5 to 5.5 so all columns have the same visual height
            ys = np.linspace(-5.5, 5.5, n_to_draw)
            ys = ys[::-1] # top to bottom
            
        nodes = []
        x = xs[i]
        
        # Colors for the layer
        if l_type == 'input':
            color = '#A3E4D7'
            edgecolor = '#117A65'
        elif l_type == 'hidden':
            color = '#D2B4DE'
            edgecolor = '#633974'
        elif l_type == 'dropout':
            color = 'white'
            edgecolor = '#E74C3C'
        else: # output
            color = '#FAD7A1'
            edgecolor = '#D35400'
            
        linestyle = '--' if l_type == 'dropout' else '-'
        
        for m, y in enumerate(ys):
            nodes.append((x, y))
            
            # Use scatter so it's always a circle, or we can use plot with marker='o'
            # Marker size is in points squared. 
            # We want large circles: s=2500 is roughly a 50x50 point circle.
            if linestyle == '--':
                ax.scatter([x], [y], s=3500, facecolors=color, edgecolors=edgecolor, zorder=4, linewidths=3, linestyle='--')
            else:
                ax.scatter([x], [y], s=3500, facecolors=color, edgecolors=edgecolor, zorder=4, linewidths=3)
            
            # If input layer, add feature names
            if i == 0 and features and m < len(features):
                ax.text(x - 0.08, y, features[m], ha='right', va='center', fontsize=28)
                
        if n > n_to_draw:
            ax.text(x, -6.5, '...', ha='center', va='center', fontsize=48, rotation=90)
            
        node_coords.append(nodes)
        
        # Stagger the layer descriptions to prevent text overlap in the Win model
        y_label = 8.5 if i % 2 == 1 else 6.5
        ax.text(x, y_label, layer_labels[i], ha='center', va='bottom', fontsize=24, fontweight='bold', bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
    # Draw edges
    for i in range(n_layers - 1):
        for n1 in node_coords[i]:
            for n2 in node_coords[i+1]:
                line = plt.Line2D([n1[0], n2[0]], [n1[1], n2[1]], c='gray', alpha=0.5, zorder=1)
                ax.add_artist(line)

    ax.set_title(title, pad=40, fontweight='bold', fontsize=40)
    
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(28, 14))

win_feats = ['alt_diff', 'avg_attend_diff', 'avg_per_player_diff', 'elo_diff', 'humid_diff', 'total_value_diff', 'xg_for_diff']
win_sizes = [7, 32, 32, 1] 
win_drawn = [7, 12, 12, 1]
win_labels = [
    "Input Layer\n(7 Features)",
    "Dense Layer\n32 Units\nSwish + L2(1e-4)",
    "Dropout Layer\nRate=0.2",
    "Output Layer\n1 Unit\nSigmoid"
]
win_types = ['input', 'hidden', 'dropout', 'output']

draw_neural_net(ax1, win_sizes, win_drawn, win_labels, win_types, features=win_feats, title='Win Prediction Neural Network')

score_feats = ['avg_attend_diff', 'avg_per_player_diff', 'xg_for_diff']
score_sizes = [3, 16, 1] 
score_drawn = [3, 6, 1]
score_labels = [
    "Input Layer\n(3 Features)",
    "Dense Layer\n16 Units\nSwish + L2(1e-3)",
    "Output Layer\n1 Unit\nLinear"
]
score_types = ['input', 'hidden', 'output']

draw_neural_net(ax2, score_sizes, score_drawn, score_labels, score_types, features=score_feats, title='Score Prediction Neural Network')

plt.tight_layout()
os.makedirs('../report/2x', exist_ok=True)
plt.savefig('../report/2x/combined_model_architectures.png', dpi=300, transparent=False, bbox_inches='tight')
plt.close()
