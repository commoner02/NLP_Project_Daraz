import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

os.makedirs('figures', exist_ok=True)

# -------------------------------------------------------------
# Figure 1: End-to-End System Architecture
# -------------------------------------------------------------
def create_system_architecture():
    fig, ax = plt.subplots(figsize=(11, 5.8), dpi=300)
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 5.8)
    ax.axis('off')

    # Color definitions
    c_blue = '#1E3A8A'
    c_teal = '#0F766E'
    c_amber = '#B45309'
    c_purple = '#6D28D9'
    c_border = '#64748B'
    c_text = '#0F172A'

    def draw_box(x, y, w, h, title, subtitle, bg_col, border_col, title_col=c_text, fontsize_title=9.2, fontsize_sub=7.2):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.1",
                             facecolor=bg_col, edgecolor=border_col, linewidth=1.3, zorder=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.65, title, ha='center', va='center', fontsize=fontsize_title, fontweight='bold', color=title_col, zorder=4)
        ax.text(x + w/2, y + h*0.28, subtitle, ha='center', va='center', fontsize=fontsize_sub, color='#334155', zorder=4)

    # Top pipeline
    draw_box(0.4, 4.3, 2.3, 1.15, "Raw Bangla Review", "Daraz Bangladesh Dataset\n(N = 2,016 annotated reviews)", '#EFF6FF', '#3B82F6', c_blue)
    draw_box(3.1, 4.3, 2.4, 1.15, "Linguistic Preprocessor", "Unicode NFC, Zero-Width Clean\nNoise Regex, De-elongation & Cues", '#F0FDF4', '#10B981', c_teal)
    draw_box(5.9, 4.3, 2.4, 1.15, "Representation Engine", "Hybrid Word+Char TF-IDF Union\nOR 2-Layer BiLSTM Trunk (256d)", '#F5F3FF', '#8B5CF6', c_purple)

    # Hierarchical Grouped Box
    group_box = FancyBboxPatch((0.4, 1.1), 7.9, 2.65, boxstyle="round,pad=0.08,rounding_size=0.12",
                               facecolor='#F8FAFC', edgecolor=c_border, linewidth=1.2, linestyle='--', zorder=1)
    ax.add_patch(group_box)
    ax.text(0.7, 3.52, "Hierarchical Multi-Task Classification Engine (3 Structured Levels)", ha='left', va='center', fontsize=9.2, fontweight='bold', color=c_blue, zorder=2)

    # 3 Level Heads
    draw_box(0.65, 1.35, 2.3, 1.8, "Level 1: Global Sentiment", "Overall 3-Class Polarity\n[Negative, Neutral, Positive]\nBalanced LogReg / BiLSTM Head", '#FEF3C7', '#F59E0B', c_amber, fontsize_title=8.6, fontsize_sub=7.0)
    draw_box(3.2, 1.35, 2.3, 1.8, "Level 2: Aspect Detection", "Multi-Label Category Det.\n5 Schemas (OvR / Sigmoid > 0.5)\n[Quality, Price, Delivery...]", '#FEF3C7', '#F59E0B', c_amber, fontsize_title=8.6, fontsize_sub=7.0)
    draw_box(5.75, 1.35, 2.3, 1.8, "Level 3: Aspect Polarity", "Conditioned Binary Polarity\n5 Dedicated Classifiers\n[Positive vs. Negative]", '#FEF3C7', '#F59E0B', c_amber, fontsize_title=8.6, fontsize_sub=7.0)

    # Output / Streamlit App
    draw_box(8.7, 1.1, 1.9, 4.35, "Streamlit Dashboard\n& Production API", "• Model Family Switcher\n• Real-Time Review ABSA\n• Aspect Breakdown Cards\n• Batch CSV Analytics\n• Confidence Gauges\n• Live Metric Comparison\n• Sub-ms CPU Inference", '#E0F2FE', '#0284C7', '#0369A1', fontsize_title=8.8, fontsize_sub=7.0)

    # Arrows
    def draw_arrow(x1, y1, x2, y2, label=None, label_y=0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.3,head_length=0.4", color='#334155', lw=1.4), zorder=5)
        if label:
            ax.text((x1+x2)/2, ((y1+y2)/2) + label_y, label, ha='center', va='center', fontsize=7.0, fontweight='bold', color='#1E293B', backgroundcolor='white', zorder=6)

    draw_arrow(2.7, 4.87, 3.1, 4.87)
    draw_arrow(5.5, 4.87, 5.9, 4.87)

    # Clean bus distribution line from feature extractor
    ax.plot([7.1, 7.1], [4.3, 3.35], color='#334155', lw=1.4, zorder=5)
    ax.plot([1.8, 7.1], [3.35, 3.35], color='#334155', lw=1.4, zorder=5)
    draw_arrow(1.8, 3.35, 1.8, 3.15)
    draw_arrow(4.35, 3.35, 4.35, 3.15)
    draw_arrow(6.9, 3.35, 6.9, 3.15)

    # Dependency arrow from Level 2 to Level 3
    draw_arrow(5.5, 2.25, 5.75, 2.25, "Active\nAspects", label_y=0.25)

    # Route output to Streamlit
    draw_arrow(8.05, 2.25, 8.7, 2.25)

    plt.tight_layout()
    plt.savefig('figures/system_architecture.pdf', bbox_inches='tight')
    plt.savefig('figures/system_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved system_architecture")

# -------------------------------------------------------------
# Figure 2: Preprocessing Pipeline Flow (Exact to src/preprocessing.py)
# -------------------------------------------------------------
def create_preprocessing_pipeline():
    fig, ax = plt.subplots(figsize=(10.8, 3.2), dpi=300)
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 3.2)
    ax.axis('off')

    stages = [
        ("Step 1: Unicode NFC", "Canonical NFC normalization\nunicodedata.normalize()", "#EFF6FF", "#3B82F6"),
        ("Step 2: Zero-Width Strip", "Removes ZWNJ, ZWJ, BOM,\nZWSP, LRM, RLM marks", "#F0FDF4", "#10B981"),
        ("Step 3: Noise Cleaning", "Regex stripping for HTML tags,\nURLs, emails & invalid noise", "#FEF3C7", "#F59E0B"),
        ("Step 4: De-elongation", "Collapses character runs >= 3\nRegex (.)\\1{2,} -> \\1", "#F5F3FF", "#8B5CF6"),
        ("Step 5: Token Safeguard", "Preserves Bangla, loanwords,\nstandard darai & intact negations", "#FEE2E2", "#EF4444")
    ]

    for i, (title, sub, bg, border) in enumerate(stages):
        x = 0.5 + i * 2.0
        y = 0.4
        w = 1.8
        h = 2.4
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.1",
                             facecolor=bg, edgecolor=border, linewidth=1.3, zorder=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.75, title, ha='center', va='center', fontsize=8.2, fontweight='bold', color='#0F172A', zorder=4)
        ax.text(x + w/2, y + h*0.35, sub, ha='center', va='center', fontsize=6.8, color='#334155', zorder=4)

        if i < len(stages) - 1:
            ax.annotate('', xy=(x + w + 0.2, y + h/2), xytext=(x + w, y + h/2),
                        arrowprops=dict(arrowstyle="->,head_width=0.25,head_length=0.35", color='#475569', lw=1.3), zorder=5)

    # Input and Output markers
    ax.text(0.35, 1.6, "Raw\nBangla\nReview", ha='right', va='center', fontsize=7.2, fontweight='bold', color='#1E3A8A')
    ax.annotate('', xy=(0.5, 1.6), xytext=(0.38, 1.6),
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.25", color='#1E3A8A', lw=1.2))

    ax.text(10.45, 1.6, "Clean\nNormalized\nText", ha='left', va='center', fontsize=7.2, fontweight='bold', color='#15803D')
    ax.annotate('', xy=(10.4, 1.6), xytext=(0.5 + 4*2.0 + 1.8, 1.6),
                arrowprops=dict(arrowstyle="->,head_width=0.2,head_length=0.25", color='#15803D', lw=1.2))

    plt.tight_layout()
    plt.savefig('figures/preprocessing_pipeline.pdf', bbox_inches='tight')
    plt.savefig('figures/preprocessing_pipeline.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved preprocessing_pipeline")

# -------------------------------------------------------------
# Figure 3: BiLSTM Architecture (Exact to src/features.py)
# -------------------------------------------------------------
def create_bilstm_architecture():
    fig, ax = plt.subplots(figsize=(9.2, 4.6), dpi=300)
    ax.set_xlim(0, 9.2)
    ax.set_ylim(0, 4.6)
    ax.axis('off')

    def draw_layer(x, y, w, h, title, sub, bg, border, tcol='#0F172A'):
        box = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.08",
                             facecolor=bg, edgecolor=border, linewidth=1.2, zorder=3)
        ax.add_patch(box)
        ax.text(x + w/2, y + h*0.62, title, ha='center', va='center', fontsize=8.2, fontweight='bold', color=tcol, zorder=4)
        ax.text(x + w/2, y + h*0.28, sub, ha='center', va='center', fontsize=7.0, color='#334155', zorder=4)

    # Trunk layers
    draw_layer(1.6, 0.3, 6.0, 0.58, "Input Review Tokens", "Length L=128 (Padded/Truncated), Vocab |V|=15,000, padding_idx=0", "#F8FAFC", "#64748B")
    draw_layer(1.6, 1.15, 6.0, 0.58, "Dense Embedding Layer", "Embedding(15000, 300) + Spatial Dropout (p=0.5)", "#EFF6FF", "#3B82F6", "#1E3A8A")
    draw_layer(1.6, 2.0, 6.0, 0.58, "2-Layer Bidirectional LSTM Trunk", "Hidden Dim H=128/dir (Total 256), num_layers=2, Recurrent Dropout p=0.5", "#F0FDF4", "#10B981", "#0F766E")
    draw_layer(1.6, 2.85, 6.0, 0.58, "Concatenated Sequence State", "h_rep = [h_forward_L || h_backward_1] in R^256, Dropout (p=0.5)", "#F5F3FF", "#8B5CF6", "#6D28D9")

    # Three task heads
    draw_layer(0.3, 3.75, 2.6, 0.68, "Global Sentiment Head", "Linear(256, 3) -> Softmax\n(Negative, Neutral, Positive)", "#FEF3C7", "#F59E0B", "#B45309")
    draw_layer(3.3, 3.75, 2.6, 0.68, "Aspect Detection Head", "Linear(256, 5) -> Sigmoid\nMulti-Label Category Detection", "#FEF3C7", "#F59E0B", "#B45309")
    draw_layer(6.3, 3.75, 2.6, 0.68, "Aspect Polarity Heads", "Linear(256, 2) per Active Aspect\n(TF-IDF LogReg Fallback in App)", "#FEF3C7", "#F59E0B", "#B45309")

    # Connecting arrows
    def draw_arr(x1, y1, x2, y2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.22,head_length=0.3", color='#334155', lw=1.2), zorder=5)

    draw_arr(4.6, 0.88, 4.6, 1.15)
    draw_arr(4.6, 1.73, 4.6, 2.0)
    draw_arr(4.6, 2.58, 4.6, 2.85)

    draw_arr(4.6, 3.43, 1.6, 3.75)
    draw_arr(4.6, 3.43, 4.6, 3.75)
    draw_arr(4.6, 3.43, 7.6, 3.75)

    plt.tight_layout()
    plt.savefig('figures/bilstm_architecture.pdf', bbox_inches='tight')
    plt.savefig('figures/bilstm_architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved bilstm_architecture")

if __name__ == '__main__':
    create_system_architecture()
    create_preprocessing_pipeline()
    create_bilstm_architecture()
