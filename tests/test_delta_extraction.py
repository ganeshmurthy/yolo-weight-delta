#!/usr/bin/env python3
"""
Test script for weight_delta.py

Uses real YOLOX-S model, adds known perturbations to head layers,
then verifies the extracted delta matches the perturbations.
"""

import os
import sys
import torch

# Add parent directory to path to import weight_delta
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from weight_delta import extract_head_from_state, compute_head_delta


def test_delta_extraction():
    """
    Test the delta extraction process with ground truth:
    1. Load real YOLOX-S model as base
    2. Create "trained" model by adding known perturbations to head layers
    3. Extract delta using our program
    4. Verify extracted delta matches the known perturbations
    """
    print("=" * 70)
    print("WEIGHT DELTA EXTRACTION TEST")
    print("=" * 70)
    print()

    # Step 1: Load base YOLOX-S model
    print("Step 1: Loading base YOLOX-S model...")
    base_path = "tests/yolox_s.pth"

    if not os.path.exists(base_path):
        print(f"ERROR: Base model not found at {base_path}")
        print("Please ensure yolox_s.pth is in the tests/ directory")
        return False

    base_checkpoint = torch.load(base_path, map_location='cpu')
    base_state = base_checkpoint.get('model', base_checkpoint)
    if hasattr(base_state, 'state_dict'):
        base_state = base_state.state_dict()

    print(f"  Loaded base model")
    print(f"  Total layers: {len(base_state)}")

    # Step 2: Extract head layers from base
    print("\nStep 2: Extracting head layers from base model...")
    base_head = extract_head_from_state(base_state)
    print(f"✓ Extracted {len(base_head)} head layers")
    print(f"  Total parameters: {sum(t.numel() for t in base_head.values()):,}")

    # Step 3: Create known perturbations
    print("\nStep 3: Creating known perturbations...")
    known_perturbations = {}
    torch.manual_seed(42)  # For reproducibility

    for key, value in base_head.items():
        # Skip non-float tensors (e.g., integer indices)
        if not value.dtype.is_floating_point:
            print(f"  Skipping {key} (dtype: {value.dtype})")
            continue

        # Add small random perturbation (mean ~0.01)
        perturbation = torch.randn_like(value) * 0.01
        known_perturbations[key] = perturbation.half().cpu()  # Match our program's float16 compression

    print(f" Created perturbations for {len(known_perturbations)} layers")

    # Calculate statistics of perturbations
    all_perturb = torch.cat([p.flatten().float() for p in known_perturbations.values()])
    print(f"  Mean |perturbation|: {all_perturb.abs().mean().item():.6f}")
    print(f"  Max |perturbation|: {all_perturb.abs().max().item():.6f}")

    # Step 4: Create "trained" model by applying perturbations
    print("\nStep 4: Creating trained model (base + perturbations)...")
    trained_state = {}
    for key, value in base_state.items():
        if key in known_perturbations:
            # Add perturbation to head layers (only float tensors)
            trained_state[key] = value + known_perturbations[key].float()
        else:
            # Keep backbone/neck/integer tensors unchanged
            trained_state[key] = value.clone()

    print(f" Created trained model")

    # Step 5: Extract head from trained model
    print("\nStep 5: Extracting head from trained model...")
    trained_head = extract_head_from_state(trained_state)
    print(f" Extracted {len(trained_head)} head layers")

    # Step 6: Compute delta using our program
    print("\nStep 6: Computing delta using weight_delta.py functions...")
    extracted_delta = compute_head_delta(trained_head, base_head)
    print(f" Computed delta for {len(extracted_delta)} layers")

    # Calculate statistics of extracted delta
    all_delta = torch.cat([d.flatten().float() for d in extracted_delta.values()])
    print(f"  Mean |delta|: {all_delta.abs().mean().item():.6f}")
    print(f"  Max |delta|: {all_delta.abs().max().item():.6f}")

    # Step 7: Verify extracted delta matches known perturbations
    print("\nStep 7: Verifying extracted delta matches known perturbations...")

    all_match = True
    max_diff = 0.0

    for key in known_perturbations.keys():
        known = known_perturbations[key].float()
        extracted = extracted_delta[key].float()

        # Check if they're close (allowing for float16 precision loss)
        if not torch.allclose(known, extracted, rtol=1e-3, atol=1e-5):
            print(f"  Mismatch in layer: {key}")
            diff = (known - extracted).abs().max().item()
            print(f"    Max difference: {diff:.6f}")
            all_match = False
            max_diff = max(max_diff, diff)
        else:
            diff = (known - extracted).abs().max().item()
            max_diff = max(max_diff, diff)

    if all_match:
        print(f" All {len(known_perturbations)} layers match!")
        print(f"  Max difference across all layers: {max_diff:.8f}")
    else:
        print(f" Some layers do not match")
        return False

    # Step 8: Test saving and loading delta
    print("\nStep 8: Testing delta save/load...")
    from weight_delta import save_delta_payload

    test_output = "tests/test_delta.pth"
    metadata = {
        "test": "delta_extraction_validation",
        "num_layers": len(extracted_delta),
        "base_model": "yolox_s.pth"
    }

    save_delta_payload(extracted_delta, test_output, metadata)
    print(f" Saved delta to {test_output}")

    # Load it back
    payload = torch.load(test_output)
    loaded_delta = payload['delta']
    loaded_metadata = payload['metadata']

    print(f"  Loaded delta back")
    print(f"  Metadata: {loaded_metadata}")

    # Verify loaded delta matches
    for key in known_perturbations.keys():
        if not torch.allclose(known_perturbations[key].float(), loaded_delta[key].float(), rtol=1e-3, atol=1e-5):
            print(f" Loaded delta doesn't match for {key}")
            return False

    print(f" Loaded delta matches original")

    # Report file size
    file_size = os.path.getsize(test_output)
    print(f"  Delta file size: {file_size:,} bytes ({file_size/1024:.2f} KB)")

    # Final result
    print("\n" + "=" * 70)
    print(" TEST PASSED")
    print("=" * 70)
    print("\nConclusion:")
    print("  - Delta extraction is mathematically correct")
    print("  - Extracted delta matches known perturbations")
    print("  - Float16 compression working properly")
    print("  - Save/load functionality verified")

    return True


if __name__ == "__main__":
    try:
        success = test_delta_extraction()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n TEST FAILED WITH EXCEPTION")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
