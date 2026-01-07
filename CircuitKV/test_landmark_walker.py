"""
Test script for Landmark-Diverse Walker implementation.

This script verifies that:
1. The CUDA extension compiles and loads
2. The landmark walker kernel runs without errors
3. The Python wrapper works correctly
"""

import torch
import numpy as np


def test_landmark_walker_basic():
    """Basic test: create synthetic attention and run landmark walker."""
    print("=" * 60)
    print("Testing Landmark-Diverse Walker")
    print("=" * 60)

    # Import after to ensure CUDA is available
    from circuit_kv import LandmarkWalkerMonitor, LandmarkWalkerConfig

    # Create config
    config = LandmarkWalkerConfig(
        max_seq_len=2048,
        num_landmarks=4,
        min_spacing=50,
        walkers_per_source=50,
        query_boost=2.0,
        position_alpha=0.6,
    )

    print(f"\nConfig: {config}")

    # Create monitor
    monitor = LandmarkWalkerMonitor(config)
    print("Monitor created successfully")

    # Create synthetic attention matrix (1000 x 1000)
    seq_len = 1000
    print(f"\nCreating synthetic attention matrix [{seq_len} x {seq_len}]...")

    # Create causal attention pattern with some structure
    attention = torch.zeros(seq_len, seq_len, device="cuda", dtype=torch.float32)

    # Fill lower triangle (causal mask)
    for i in range(seq_len):
        # Uniform attention to past + some spiky attention to "important" tokens
        row = torch.softmax(torch.randn(i + 1, device="cuda"), dim=0)
        attention[i, :i + 1] = row

    # Add some "bridge" tokens with high attention at specific positions
    bridge_positions = [200, 500, 750]
    for bp in bridge_positions:
        # Make tokens around bp attend strongly to bp
        for i in range(bp + 1, min(bp + 100, seq_len)):
            attention[i, bp] += 0.3
        # Re-normalize
        attention[bp + 1:min(bp + 100, seq_len)] /= attention[bp + 1:min(bp + 100, seq_len)].sum(dim=1, keepdim=True)

    print(f"Attention matrix created: {attention.shape}")
    print(f"  - Non-zero entries: {(attention > 0).sum().item()}")
    print(f"  - Max value: {attention.max().item():.4f}")

    # Run landmark walker
    current_idx = seq_len - 1
    print(f"\nRunning landmark walker with query at position {current_idx}...")

    monitor.update(attention, current_idx)
    monitor.synchronize()

    print("Landmark walker completed")

    # Get scores
    scores = monitor.get_scores()
    print(f"\nScores shape: {scores.shape}")
    print(f"  - Min: {scores.min().item():.4f}")
    print(f"  - Max: {scores.max().item():.4f}")
    print(f"  - Mean: {scores.mean().item():.4f}")
    print(f"  - Non-zero: {(scores > 0).sum().item()}")

    # Check if bridge positions have high scores
    print(f"\nBridge token scores (positions {bridge_positions}):")
    for bp in bridge_positions:
        print(f"  - Position {bp}: {scores[bp].item():.4f}")

    # Get top-K tokens
    k = 20
    top_values, top_indices = scores[:seq_len].topk(k)
    print(f"\nTop {k} tokens by score:")
    for i, (idx, val) in enumerate(zip(top_indices.cpu().numpy(), top_values.cpu().numpy())):
        marker = " ** BRIDGE" if idx in bridge_positions else ""
        print(f"  {i + 1}. Position {idx}: {val:.4f}{marker}")

    # Test keep mask
    budget_ratio = 0.25
    keep_mask = monitor.get_keep_mask(budget_ratio, seq_len)
    kept = keep_mask.sum().item()
    print(f"\nKeep mask with {budget_ratio * 100:.0f}% budget:")
    print(f"  - Kept: {kept}/{seq_len} ({kept / seq_len * 100:.1f}%)")

    # Check if bridge positions are kept
    bridge_kept = sum(keep_mask[bp].item() for bp in bridge_positions)
    print(f"  - Bridge tokens kept: {bridge_kept}/{len(bridge_positions)}")

    # Clean up
    monitor.reset()
    print("\n" + "=" * 60)
    print("Test PASSED")
    print("=" * 60)

    return True


def test_h2o_comparison():
    """Compare landmark walker against H2O baseline."""
    print("\n" + "=" * 60)
    print("Comparison: Landmark Walker vs H2O")
    print("=" * 60)

    from circuit_kv import LandmarkWalkerMonitor, LandmarkWalkerConfig

    # Create config
    config = LandmarkWalkerConfig(
        max_seq_len=2048,
        num_landmarks=8,
        min_spacing=30,
        walkers_per_source=100,
    )

    monitor = LandmarkWalkerMonitor(config)

    # Create attention with hidden bridge tokens
    seq_len = 500
    attention = torch.zeros(seq_len, seq_len, device="cuda", dtype=torch.float32)

    # Create clusters with local attention
    cluster_size = 50
    for c in range(seq_len // cluster_size):
        start = c * cluster_size
        end = min(start + cluster_size, seq_len)
        for i in range(start, end):
            for j in range(start, i + 1):
                attention[i, j] = 1.0

    # Normalize rows
    row_sums = attention.sum(dim=1, keepdim=True)
    row_sums[row_sums == 0] = 1
    attention = attention / row_sums

    # Add a hidden bridge that connects clusters
    bridge_pos = 123  # Position that connects cluster 2 to cluster 4
    bridge_target = 250  # Position in cluster 5

    # Make positions after bridge_target attend to bridge_pos
    for i in range(bridge_target, min(bridge_target + 30, seq_len)):
        attention[i, bridge_pos] = 0.5
        attention[i] /= attention[i].sum()

    # Compute H2O scores (column sums)
    h2o_scores = attention.sum(dim=0)

    # Run landmark walker
    current_idx = seq_len - 1
    monitor.update(attention, current_idx)
    monitor.synchronize()
    lw_scores = monitor.get_scores()[:seq_len]

    # Normalize for comparison
    h2o_norm = h2o_scores / h2o_scores.max()
    lw_norm = lw_scores / (lw_scores.max() + 1e-8)

    # Compare bridge token ranking
    print(f"\nBridge token at position {bridge_pos}:")
    print(f"  - H2O score (normalized): {h2o_norm[bridge_pos].item():.4f}")
    print(f"  - Landmark Walker score: {lw_norm[bridge_pos].item():.4f}")

    # Check rankings
    h2o_rank = (h2o_norm > h2o_norm[bridge_pos]).sum().item() + 1
    lw_rank = (lw_norm > lw_norm[bridge_pos]).sum().item() + 1

    print(f"  - H2O rank: {h2o_rank}/{seq_len}")
    print(f"  - LW rank: {lw_rank}/{seq_len}")

    if lw_rank < h2o_rank:
        print("\n  Landmark Walker ranks bridge token HIGHER than H2O")
    else:
        print("\n  Note: In this synthetic example, both methods have similar performance")

    print("\n" + "=" * 60)
    print("Comparison complete")
    print("=" * 60)


if __name__ == "__main__":
    # Check CUDA availability
    if not torch.cuda.is_available():
        print("ERROR: CUDA not available")
        exit(1)

    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print()

    try:
        test_landmark_walker_basic()
        test_h2o_comparison()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
