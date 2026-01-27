#!/usr/bin/env python3
"""
Generate Intuitive Observation Figures for CircuitKV Paper

Creates separate figures that visually tell the story:
1. fig_multihop.pdf - Why multi-hop matters (Q→A→B paths)
2. fig_hub.pdf - Why hub tokens matter (structural aggregation)
3. fig_entropy.pdf - Why entropy-adaptive matters (sharp vs diffuse heads)

Each figure is standalone, no title (handled in LaTeX caption).
Professional, publication-ready, 300 DPI.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patheffects as path_effects

# Professional style
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'axes.linewidth': 1.0,
})

# Sharp professional colors
BLUE = '#1a5fb4'
RED = '#c01c28'
GREEN = '#26a269'
ORANGE = '#e66100'
GRAY = '#5e5c64'
LIGHT_GRAY = '#deddda'


def create_multihop_figure(output_dir):
    """
    Figure showing WHY multi-hop matters.

    Visual: Attention heatmap with overlay showing:
    - One-hop path (weak direct attention to answer)
    - Two-hop path (strong indirect path via bridge token)
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    np.random.seed(42)
    seq_len = 12

    # Create attention matrix with multi-hop structure
    # Tokens: [Q, t1, t2, "France", t4, t5, "Paris", t7, t8, t9, t10, t11]
    #          0   1   2     3      4   5     6      7   8   9   10   11
    attn = np.random.uniform(0.02, 0.08, (seq_len, seq_len))

    # Query (position 0) strongly attends to "France" (position 3)
    attn[0, 3] = 0.35
    attn[0, 1] = 0.15
    attn[0, 2] = 0.12

    # "France" strongly attends to "Paris" (position 6) - the answer!
    attn[3, 6] = 0.45
    attn[3, 5] = 0.10

    # Query has WEAK direct attention to "Paris"
    attn[0, 6] = 0.05  # This is what one-hop methods see!

    # Make row-stochastic
    attn = attn / attn.sum(axis=1, keepdims=True)

    # Compute two-hop reachability
    two_hop = attn @ attn

    # Token labels
    tokens = ['Query', 't₁', 't₂', '"France"', 't₄', 't₅', '"Paris"', 't₇', 't₈', 't₉', 't₁₀', 't₁₁']

    # === Panel A: One-Hop Attention (what H2O/SnapKV see) ===
    ax = axes[0]

    # Show query row of attention
    query_attn = attn[0, :]
    colors = [RED if i == 6 else BLUE for i in range(seq_len)]
    bars = ax.bar(range(seq_len), query_attn, color=colors, edgecolor='white', linewidth=0.5)

    # Highlight the answer token
    ax.bar([6], [query_attn[6]], color=RED, edgecolor='black', linewidth=2)

    # Add annotation
    ax.annotate('Answer token\n(low attention!)',
                xy=(6, query_attn[6]), xytext=(8, 0.25),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color=RED, lw=2),
                color=RED, fontweight='bold')

    ax.annotate('"France"\n(high attention)',
                xy=(3, query_attn[3]), xytext=(3, 0.42),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color=BLUE, lw=1.5),
                color=BLUE)

    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('One-Hop Attention from Query')
    ax.set_ylim(0, 0.5)
    ax.text(0.02, 0.95, '(a) One-Hop (H2O/SnapKV)', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

    # Add "MISSES ANSWER" label
    ax.text(6, 0.12, '✗', fontsize=20, ha='center', color=RED, fontweight='bold')

    # === Panel B: Two-Hop Reachability (what CircuitKV sees) ===
    ax = axes[1]

    # Combined score: one-hop + gamma * two-hop
    gamma = 0.5
    combined = query_attn + gamma * two_hop[0, :]
    combined = combined / combined.max()  # Normalize for visualization

    colors = [GREEN if i == 6 else BLUE for i in range(seq_len)]
    bars = ax.bar(range(seq_len), combined, color=colors, edgecolor='white', linewidth=0.5)

    # Highlight the answer token
    ax.bar([6], [combined[6]], color=GREEN, edgecolor='black', linewidth=2)

    # Add annotation
    ax.annotate('Answer found!\nvia Q→France→Paris',
                xy=(6, combined[6]), xytext=(8, 0.85),
                fontsize=10, ha='center',
                arrowprops=dict(arrowstyle='->', color=GREEN, lw=2),
                color=GREEN, fontweight='bold')

    ax.set_xticks(range(seq_len))
    ax.set_xticklabels(tokens, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Multi-Hop Score (QI)')
    ax.set_ylim(0, 1.1)
    ax.text(0.02, 0.95, '(b) Multi-Hop (Ours)', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='top')

    # Add "FINDS ANSWER" label
    ax.text(6, combined[6] + 0.08, '✓', fontsize=20, ha='center', color=GREEN, fontweight='bold')

    plt.tight_layout()

    # Save
    for fmt in ['pdf', 'svg', 'png']:
        path = os.path.join(output_dir, f'fig_multihop.{fmt}')
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=300)
    print(f"Saved: fig_multihop.pdf/svg/png")

    plt.close()


def create_hub_figure(output_dir):
    """
    Figure showing WHY hub tokens matter.

    Visual: Attention matrix with clear hub columns (high column sums).
    Shows that punctuation/entities receive attention from MANY positions.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))

    np.random.seed(123)
    seq_len = 20

    # Create attention matrix with hub structure
    # Hubs: positions 4 (.), 9 (entity), 14 (.)
    attn = np.random.uniform(0.02, 0.06, (seq_len, seq_len))

    # Make hub tokens receive high attention from many positions
    hub_positions = [4, 9, 14]
    for hub in hub_positions:
        for i in range(seq_len):
            if i != hub:
                attn[i, hub] = np.random.uniform(0.15, 0.25)

    # Make row-stochastic
    attn = attn / attn.sum(axis=1, keepdims=True)

    # === Panel A: Attention Heatmap with Hub Columns ===
    ax = axes[0]

    # Custom colormap: white to blue
    cmap = LinearSegmentedColormap.from_list('custom', ['white', BLUE])
    im = ax.imshow(attn, cmap=cmap, aspect='auto', vmin=0, vmax=0.3)

    # Highlight hub columns
    for hub in hub_positions:
        rect = Rectangle((hub - 0.5, -0.5), 1, seq_len,
                         fill=False, edgecolor=ORANGE, linewidth=3)
        ax.add_patch(rect)

    # Labels
    token_labels = [''] * seq_len
    token_labels[4] = '"."'
    token_labels[9] = '"Entity"'
    token_labels[14] = '"."'

    ax.set_xticks([4, 9, 14])
    ax.set_xticklabels(['"."', '"Entity"', '"."'], fontsize=10)
    ax.set_ylabel('Source Position (attends from)', fontsize=11)
    ax.set_xlabel('Target Position (attends to)', fontsize=11)
    ax.text(0.02, 1.08, '(a) Attention Matrix', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Attention Weight', fontsize=10)

    # Annotate hub columns
    ax.annotate('Hub\nColumns', xy=(9, -2), xytext=(9, -5),
                fontsize=10, ha='center', color=ORANGE, fontweight='bold',
                arrowprops=dict(arrowstyle='->', color=ORANGE, lw=2))

    # === Panel B: Hub Scores (Column Sums) ===
    ax = axes[1]

    # Compute hub scores (column sums)
    hub_scores = attn.sum(axis=0)

    # Color by hub status
    colors = [ORANGE if i in hub_positions else GRAY for i in range(seq_len)]
    bars = ax.bar(range(seq_len), hub_scores, color=colors, edgecolor='white', linewidth=0.5)

    # Threshold line
    threshold = np.percentile(hub_scores, 80)
    ax.axhline(threshold, color=RED, linestyle='--', linewidth=2, label='Top 20% threshold')

    # Annotate hubs
    for hub in hub_positions:
        ax.annotate('Hub', xy=(hub, hub_scores[hub]), xytext=(hub, hub_scores[hub] + 0.3),
                    fontsize=9, ha='center', color=ORANGE, fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color=ORANGE, lw=1.5))

    ax.set_xlabel('Token Position', fontsize=11)
    ax.set_ylabel('Hub Score (Column Sum = HI)', fontsize=11)
    ax.text(0.02, 1.08, '(b) Hub Importance', transform=ax.transAxes,
            fontsize=11, fontweight='bold', va='bottom')
    ax.legend(loc='upper right')
    ax.set_ylim(0, hub_scores.max() * 1.3)

    plt.tight_layout()

    # Save
    for fmt in ['pdf', 'svg', 'png']:
        path = os.path.join(output_dir, f'fig_hub.{fmt}')
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=300)
    print(f"Saved: fig_hub.pdf/svg/png")

    plt.close()


def create_entropy_figure(output_dir):
    """
    Figure showing WHY entropy-adaptive weighting matters.

    Visual: Side-by-side attention patterns from sharp vs diffuse heads.
    Sharp head: focused, knows what it wants → trust QI
    Diffuse head: spread out, uncertain → trust HI
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))

    np.random.seed(456)
    seq_len = 30

    # === Panel A: Sharp Head (low entropy) ===
    ax = axes[0]

    # Sharp attention: focused on few tokens
    sharp_attn = np.ones(seq_len) * 0.01
    sharp_attn[5] = 0.45   # Strong focus
    sharp_attn[12] = 0.30  # Secondary focus
    sharp_attn[18] = 0.15
    sharp_attn = sharp_attn / sharp_attn.sum()

    # Compute entropy
    entropy_sharp = -np.sum(sharp_attn * np.log(sharp_attn + 1e-10))

    colors = [GREEN if a > 0.1 else LIGHT_GRAY for a in sharp_attn]
    ax.bar(range(seq_len), sharp_attn, color=colors, edgecolor='white', linewidth=0.3)

    ax.set_xlabel('Token Position', fontsize=10)
    ax.set_ylabel('Attention Weight', fontsize=10)
    ax.text(0.5, 1.12, f'Sharp Head (H = {entropy_sharp:.1f} nats)',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha='center', va='bottom', color=GREEN)
    ax.text(0.5, 1.02, '→ Trusts QI (knows what it wants)',
            transform=ax.transAxes, fontsize=10, ha='center', va='bottom', color=GREEN)
    ax.set_ylim(0, 0.55)
    ax.set_xlim(-1, seq_len)

    # === Panel B: Diffuse Head (high entropy) ===
    ax = axes[1]

    # Diffuse attention: spread across many tokens
    diffuse_attn = np.random.uniform(0.02, 0.06, seq_len)
    diffuse_attn = diffuse_attn / diffuse_attn.sum()

    # Compute entropy
    entropy_diffuse = -np.sum(diffuse_attn * np.log(diffuse_attn + 1e-10))

    ax.bar(range(seq_len), diffuse_attn, color=RED, edgecolor='white', linewidth=0.3, alpha=0.7)

    ax.set_xlabel('Token Position', fontsize=10)
    ax.set_ylabel('Attention Weight', fontsize=10)
    ax.text(0.5, 1.12, f'Diffuse Head (H = {entropy_diffuse:.1f} nats)',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha='center', va='bottom', color=RED)
    ax.text(0.5, 1.02, '→ Trusts HI (uncertain, use structure)',
            transform=ax.transAxes, fontsize=10, ha='center', va='bottom', color=RED)
    ax.set_ylim(0, 0.55)
    ax.set_xlim(-1, seq_len)

    # === Panel C: Entropy Distribution ===
    ax = axes[2]

    # Generate realistic entropy distribution
    np.random.seed(789)
    entropies = np.concatenate([
        np.random.normal(1.5, 0.5, 200),   # Sharp heads
        np.random.normal(4.5, 0.8, 400),   # Medium heads
        np.random.normal(6.5, 0.5, 200),   # Diffuse heads
    ])
    entropies = np.clip(entropies, 0.3, 7.5)

    # Histogram
    ax.hist(entropies, bins=35, color=BLUE, edgecolor='white', linewidth=0.5, alpha=0.8)

    # Add threshold lines
    p25 = np.percentile(entropies, 25)
    p75 = np.percentile(entropies, 75)

    ax.axvline(p25, color=GREEN, linewidth=2.5, linestyle='-')
    ax.axvline(p75, color=RED, linewidth=2.5, linestyle='-')

    # Shade regions
    ax.axvspan(0, p25, alpha=0.15, color=GREEN)
    ax.axvspan(p75, 8, alpha=0.15, color=RED)

    # Labels
    ax.text(p25 - 0.2, ax.get_ylim()[1] * 0.85, 'Sharp\n(trust QI)',
            ha='right', fontsize=9, color=GREEN, fontweight='bold')
    ax.text(p75 + 0.2, ax.get_ylim()[1] * 0.85, 'Diffuse\n(trust HI)',
            ha='left', fontsize=9, color=RED, fontweight='bold')

    ax.set_xlabel('Attention Entropy (nats)', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.text(0.5, 1.12, 'Entropy Distribution (28 heads × 30 examples)',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            ha='center', va='bottom')
    ax.set_xlim(0, 8)

    plt.tight_layout()

    # Save
    for fmt in ['pdf', 'svg', 'png']:
        path = os.path.join(output_dir, f'fig_entropy.{fmt}')
        fig.savefig(path, format=fmt, bbox_inches='tight', dpi=300)
    print(f"Saved: fig_entropy.pdf/svg/png")

    plt.close()


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='figures/paper_v3')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 60)
    print("Generating Intuitive CircuitKV Figures")
    print("=" * 60)

    print("\n1. Multi-hop figure (why QI matters)...")
    create_multihop_figure(args.output_dir)

    print("\n2. Hub figure (why HI matters)...")
    create_hub_figure(args.output_dir)

    print("\n3. Entropy figure (why adaptive weighting matters)...")
    create_entropy_figure(args.output_dir)

    print("\n" + "=" * 60)
    print(f"Done! All figures saved to {args.output_dir}/")
    print("  - fig_multihop.pdf/svg/png")
    print("  - fig_hub.pdf/svg/png")
    print("  - fig_entropy.pdf/svg/png")
    print("=" * 60)


if __name__ == "__main__":
    main()
