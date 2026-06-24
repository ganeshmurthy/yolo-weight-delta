#!/usr/bin/env python3
"""
Test the EXACT code the customer is using
"""

from weight_delta import apply_head_delta
import torch

print("=" * 70)
print("TESTING CUSTOMER'S EXACT CODE")
print("=" * 70)
print()

print("Executing customer's code:")
print()

# Load base YOLOX-S model (already on device)
print("Step 1: Loading base model...")
base_model = torch.load("tests/yolox_s.pth", map_location='cpu')['model']
print(f"  Type of base_model: {type(base_model)}")
print(f"  Is it a dict? {isinstance(base_model, dict)}")
if isinstance(base_model, dict):
    print(f"  Number of layers: {len(base_model)}")
print()

# Load delta file (received via OTA update)
print("Step 2: Loading delta file...")
delta_payload = torch.load("tests/test_delta.pth", map_location='cpu')
print(f"  Delta keys: {list(delta_payload.keys())}")
print(f"  Number of delta layers: {len(delta_payload['delta'])}")
print()

# Apply delta to base model
print("Step 3: Applying delta to base model...")
try:
    apply_head_delta(base_model, delta_payload['delta'])
    print("  OK Delta applied successfully!")
    print()
    print("=" * 70)
    print("SUCCESS! Customer's code now works with updated apply_head_delta()")
    print("=" * 70)
except Exception as e:
    print(f"  FAILED: {e}")
    print()
    import traceback
    traceback.print_exc()
