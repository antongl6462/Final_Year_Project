"""
Generate a professional flowchart diagram for the Hybrid Quantum CNN architecture.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path

# Set up the figure
fig, ax = plt.subplots(figsize=(10, 14))
ax.set_xlim(0, 10)
ax.set_ylim(0, 14)
ax.axis('off')

# Define colors - shades of blue to white gradient
input_color = '#1E3A5F'       # Dark blue for input
classical_color = '#4A6FA5'   # Medium blue for classical layers
quantum_color = '#7FA3CC'     # Light blue for quantum layers
classical_hidden_color = '#B3CCE6'  # Very light blue for classical hidden
output_color = '#E6F0F7'      # Almost white blue for output
text_color = 'white'
dark_text_color = '#2C3E50'   # Dark text for light backgrounds

# Define box positions (x, y, width, height)
boxes = [
    # (x, y, width, height, label, color, sub_label, use_dark_text)
    (3, 12, 4, 1.2, 'Input Image', input_color, '32×32 pixels', False),
    (3, 9.5, 4, 1.5, 'Classical CNN', classical_color, 'Feature Extraction\nConv + ReLU + MaxPool', False),
    (3, 7, 4, 1.2, 'Quantum Embedding', classical_color, 'Linear: dim=8', False),
    (3, 4.5, 4, 1.8, '4-Qubit Variational Circuit', quantum_color, '2 Quantum Layers\nParameterized Rotations + Entanglement', False),
    (3, 2.2, 4, 1.2, 'Classical Hidden Layer', classical_hidden_color, 'Linear: dim=64 + ReLU', True),
    (3, 0.2, 4, 1.2, 'Output Layer', output_color, '2 Classes: Cracked / Not Cracked', True),
]

# Draw boxes
drawn_boxes = []
for x, y, w, h, label, color, sub_label, use_dark_text in boxes:
    # Create fancy box with rounded corners
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor='#2C3E50',
        linewidth=2.5,
        zorder=2
    )
    ax.add_patch(box)
    drawn_boxes.append((x, y, w, h))
    
    # Determine text color based on background
    current_text_color = dark_text_color if use_dark_text else text_color
    
    # Main label
    ax.text(
        x + w/2, y + h*0.65,
        label,
        ha='center', va='center',
        fontsize=14,
        fontweight='bold',
        color=current_text_color,
        zorder=3
    )
    
    # Sub label
    ax.text(
        x + w/2, y + h*0.28,
        sub_label,
        ha='center', va='center',
        fontsize=10,
        color=current_text_color,
        style='italic',
        zorder=3
    )

# Draw arrows between boxes
arrow_style = dict(
    arrowstyle='->,head_width=0.6,head_length=0.8',
    color='#34495E',
    linewidth=3,
    zorder=1
)

for i in range(len(drawn_boxes) - 1):
    x1, y1, w1, h1 = drawn_boxes[i]
    x2, y2, w2, h2 = drawn_boxes[i + 1]
    
    # Arrow from bottom of current box to top of next box
    arrow = FancyArrowPatch(
        (x1 + w1/2, y1),
        (x2 + w2/2, y2 + h2),
        **arrow_style
    )
    ax.add_patch(arrow)

# Add title
ax.text(
    5, 13.7,
    'Hybrid Quantum-Classical CNN Architecture',
    ha='center', va='center',
    fontsize=18,
    fontweight='bold',
    color='#2C3E50'
)

# Add legend (positioned to avoid overlap)
legend_x = 0.5
legend_y = 10.5
legend_spacing = 0.7

legend_items = [
    (input_color, 'Input', False),
    (classical_color, 'Classical CNN', False),
    (quantum_color, 'Quantum Circuit', False),
    (classical_hidden_color, 'Classical Hidden', True),
    (output_color, 'Output', True)
]

for idx, (color, label, use_dark_text) in enumerate(legend_items):
    y_pos = legend_y + (len(legend_items) - idx - 1) * legend_spacing
    
    # Legend box
    legend_box = FancyBboxPatch(
        (legend_x, y_pos - 0.15), 0.5, 0.4,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor='#2C3E50',
        linewidth=1.5
    )
    ax.add_patch(legend_box)
    
    # Legend text
    ax.text(
        legend_x + 0.8, y_pos + 0.05,
        label,
        ha='left', va='center',
        fontsize=10,
        color='#2C3E50'
    )

# Add information box
info_x = 0.3
info_y = 0.5
info_text = (
    "Total Parameters: ~500K\n"
    "Quantum Qubits: 4\n"
    "Trainable via Backpropagation"
)

ax.text(
    info_x, info_y,
    info_text,
    ha='left', va='bottom',
    fontsize=9,
    color='#7F8C8D',
    style='italic',
    bbox=dict(boxstyle='round,pad=0.5', facecolor='#ECF0F1', edgecolor='#BDC3C7', linewidth=1.5)
)

plt.tight_layout()

# Save the figure
output_dir = Path(__file__).parent.parent.parent / 'runs' / 'quantum_cnn' / 'notebook_run'
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'qcnn_architecture_flowchart.png'

plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Flowchart saved to: {output_path}")

# Also save to current directory for easy access
local_path = Path(__file__).parent / 'qcnn_architecture_flowchart.png'
plt.savefig(local_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f"Flowchart also saved to: {local_path}")

plt.show()
