# Tests

This directory contains validation tests for the YOLO weight delta extraction tool.

## Test Files

- `yolox_s.pth` - Base YOLOX-S model (69 MB) - downloaded separately
- `test_delta_extraction.py` - Main validation test
- `test_delta.pth` - Output from test run (ignored by git)

---

## Prerequisites

1. **PyTorch installed**
   ```bash
   pip install torch
   ```

2. **YOLOX-S base model downloaded**
   
   The test requires the real YOLOX-S model. Download it:
   ```bash
   cd tests
   wget https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_s.pth
   ```
   
   This will download a 69 MB file to `tests/yolox_s.pth`.

---

## Running the Test

From the project root directory:

```bash
python tests/test_delta_extraction.py
```

Or from within the tests directory:

```bash
cd tests
python test_delta_extraction.py
```

---

## What the Test Does

The test validates the mathematical correctness of delta extraction:

1. **Loads** the real YOLOX-S model (69 MB, 462 layers, 108 head layers)
2. **Creates** known random perturbations to head layers only
3. **Generates** a "trained" model by adding perturbations to the base model
4. **Extracts** the delta using `weight_delta.py` functions
5. **Verifies** the extracted delta matches the known perturbations
6. **Tests** save/load functionality

**Ground Truth Validation**: Since we know exactly what perturbations we added, we can verify the extracted delta is mathematically correct.

---

## Expected Output

```
======================================================================
WEIGHT DELTA EXTRACTION TEST
======================================================================

Step 1: Loading base YOLOX-S model...
✓ Loaded base model
  Total layers: 462

Step 2: Extracting head layers from base model...
✓ Extracted 108 head layers
  Total parameters: 1,924,750

Step 3: Creating known perturbations...
  Skipping head.cls_convs.0.0.bn.num_batches_tracked (dtype: torch.int64)
  ... (15 integer batch tracking tensors skipped)
✓ Created perturbations for 93 layers
  Mean |perturbation|: 0.007978
  Max |perturbation|: 0.048309

Step 4: Creating trained model (base + perturbations)...
✓ Created trained model

Step 5: Extracting head from trained model...
✓ Extracted 108 head layers

Step 6: Computing delta using weight_delta.py functions...
✓ Computed delta for 108 layers
  Mean |delta|: 0.007978
  Max |delta|: 0.048309

Step 7: Verifying extracted delta matches known perturbations...
✓ All 93 layers match!
  Max difference across all layers: 0.00000024

Step 8: Testing delta save/load...
✓ Saved delta to tests/test_delta.pth
✓ Loaded delta back
  Metadata: {'test': 'delta_extraction_validation', 'num_layers': 108, 'base_model': 'yolox_s.pth'}
✓ Loaded delta matches original
  Delta file size: 3,882,941 bytes (3791.93 KB)

======================================================================
✓ TEST PASSED
======================================================================

Conclusion:
  - Delta extraction is mathematically correct
  - Extracted delta matches known perturbations
  - Float16 compression working properly
  - Save/load functionality verified
```

---

## Test Exit Codes

- **Exit 0**: Test passed
- **Exit 1**: Test failed

---

## What Gets Validated

✅ **Mathematical Correctness**: Extracted delta matches known perturbations within floating-point precision (max diff ~0.00000024)

✅ **Real Model Architecture**: Uses actual YOLOX-S model with authentic layer structure

✅ **Float16 Compression**: Verifies conversion to half-precision works correctly

✅ **Save/Load**: Confirms delta files can be saved and loaded with metadata intact

✅ **Layer Handling**: Properly handles both float tensors (weights/biases) and integer tensors (batch tracking)

---

## Troubleshooting

**Error: "Base model not found"**
- Make sure `yolox_s.pth` is downloaded in the `tests/` directory
- Run the wget command above

**Error: "No module named 'torch'"**
- Install PyTorch: `pip install torch`

**Error: "No module named 'weight_delta'"**
- Run the test from the project root directory, not from inside `tests/`
- Or ensure the parent directory is in your Python path

---

## Notes

- The test creates a `test_delta.pth` file (~3.8 MB) which is ignored by git
- Integer batch tracking tensors are skipped (cannot add random noise to integers)
- Test uses `torch.manual_seed(42)` for reproducibility
- The 3.8 MB delta size is for full YOLOX-S head; production deltas for smaller models or fewer changes will be ~270 KB
