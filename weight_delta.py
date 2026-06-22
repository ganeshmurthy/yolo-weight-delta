import argparse
import os
import sys
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, Any

def detect_architecture_prefix(state_dict: Dict[str, torch.Tensor]) -> str:
    """
    Scans the model state keys to identify the structural contract 
    of the underlying architecture (YOLOX vs. YOLOv26/Ultralytics).
    """
    if any(k.startswith("model.head.") for k in state_dict.keys()):
        return "model.head."
    elif any(k.startswith("head.") for k in state_dict.keys()):
        return "head."
    else:
        raise ValueError("CRITICAL: Input model architecture layout is unknown or unsupported.")


def freeze_backbone_and_neck(model: nn.Module) -> None:
    """
    MOTHERSHIP UTILITY: Freezes feature extraction (backbone/neck) layers
    and unlocks only the architecture-specific detection head layers for retraining.

    NOTE: This function is NOT used in this program (weight_delta.py handles Step 2: delta extraction only).
          It is provided for REFERENCE purposes for the training team (Step 1) who need to freeze
          backbone/neck layers before retraining on the OpenShift AI cluster.

    Usage (in your training script):
        from weight_delta import freeze_backbone_and_neck
        model = load_yolox_model()
        freeze_backbone_and_neck(model)
        # ... run your training loop ...
    """
    state_dict = model.state_dict()
    prefix = detect_architecture_prefix(state_dict)

    for name, param in model.named_parameters():
        if name.startswith(prefix):
            param.requires_grad = True
        else:
            param.requires_grad = False


def extract_head_from_state(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    UTILITY: Automatically detects the architecture style and isolates
    a deep copy of only the detection head layers from a full state dict.
    """
    prefix = detect_architecture_prefix(state_dict)

    # Extract only the head layers (keys that start with the detected prefix)
    head_state = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            # Move tensor to CPU and create a deep copy
            head_state[key] = value.cpu().clone()

    return head_state


def compute_head_delta(retrained_head: Dict[str, torch.Tensor], base_head: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    MOTHERSHIP UTILITY: Computes the element-wise mathematical difference (Delta)
    between a newly retrained detection head and the baseline central model head.

    This function acts as the core weight-compression engine of the pipeline. By 
    isolating the matrix changes ($\Delta = \mathbf{W}_{\text{retrained}} - \mathbf{W}_{\text{base}}$) 
    and casting them to half-precision (float16), it generates an ultra-lightweight 
    state dictionary payload tailored for transmission across bandwidth-constrained networks.

    Mathematical Operations:
    -----------------------
    1. Upcasts both input tensors to full precision (`float32`) via `.float()` prior 
       to subtraction to guarantee numerical stability and eliminate underflow/rounding errors.
    2. Performs element-wise matrix subtraction for every tracking layer key.
    3. Compresses the resulting difference matrix down to half precision (`float16`) 
       via `.half()` to cut the physical file footprint in half.
    4. Offloads the resulting tensor payload to host system memory (`.cpu()`) to 
       free up active GPU memory allocations on the cluster.

    Args:
        retrained_head (Dict[str, torch.Tensor]): The state dictionary slice containing 
            only the detection head weights extracted from the post-training model snapshot.
            Expected layer values should be standard floating-point weights.
        base_head (Dict[str, torch.Tensor]): The state dictionary slice containing 
            the pristine detection head weights extracted from the central base model 
            *prior* to the training run.

    Returns:
        Dict[str, torch.Tensor]: A standard Python dictionary where:
            - Keys (str): Match the exact layer names of the detection head 
              (e.g., 'model.head.cv2.0.0.conv.weight').
            - Values (torch.Tensor): Element-wise delta adjustment maps compressed 
              explicitly to `torch.float16`.

    Raises:
        KeyError: If a structural contract mismatch occurs where a layer name string 
            present in the `base_head` is missing from the `retrained_head`.

    Example Memory Impact:
        - Full YOLOv26 model dictionary footprint: ~300 MB (`float32`)
        - Untargeted Full Delta dictionary footprint: ~300 MB
        - Isolated Head Delta payload folder (`float16`): **~270 KB to 1 MB**
    """
    # Validate that both heads have the same structure
    if set(base_head.keys()) != set(retrained_head.keys()):
        raise KeyError("Mismatched layers between Base Head and Retrained Head structures.")

    delta = {}
    for key in base_head:
        diff = retrained_head[key].float() - base_head[key].float()
        delta[key] = diff.half().cpu()
    return delta


def save_delta_payload(delta: Dict[str, torch.Tensor], save_path: str, metadata: Dict[str, Any] = None) -> None:
    """
    MOTHERSHIP UTILITY: Packages the compressed float16 head delta and 
    optional tracking metadata into a standalone .pth file.
    """
    payload = {
        "delta": delta,
        "metadata": metadata or {}
    }
    torch.save(payload, save_path)


def apply_head_delta(edge_model: nn.Module, delta: Dict[str, torch.Tensor]) -> None:
    """
    EDGE UTILITY: Injects the incoming float16 delta adjustments back into the
    base edge model layers in-memory prior to ONNX/TensorRT compilation.

    NOTE: This function is NOT used in this program (weight_delta.py handles Step 2: delta extraction only).
          It is provided for REFERENCE purposes for the edge device integration team who need to
          apply deltas to their base YOLOX-S models in their RTSP camera inferencing scripts.

    Usage (in edge device inference script):
        from weight_delta import apply_head_delta
        base_model = load_yolox_model()
        delta_payload = torch.load("delta.pth")
        apply_head_delta(base_model, delta_payload['delta'])
        # ... run inference or export to ONNX/TensorRT ...
    """
    current_state = edge_model.state_dict()
    updated_layers = {}

    for key in delta:
        if key in current_state:
            updated_layers[key] = current_state[key].float() + delta[key].float()

    current_state.update(updated_layers)
    edge_model.load_state_dict(current_state)


def main():
    """
    Separation of Duties:
    ─────────────────────
    STEP 1 (Training Phase - NOT this tool's responsibility):
      - Edge devices stream 40-50% confidence images to Mothership queue
      - OpenShift AI cluster training pipeline processes images
      - Training team loads base model, freezes backbone/neck, runs training loop
      - OUTPUT: new_model.pt saved to disk

    STEP 2 (Delta Extraction - THIS TOOL):
      - Takes two static files: base_model.pt and new_model.pt
      - Extracts detection head layers from both
      - Computes element-wise delta (New - Old)
      - Compresses to float16
      - OUTPUT: tiny_delta.pth (~270 KB)

    STEP 3 (Distribution - NOT this tool's responsibility):
      - Network tools (rsync/scp) distribute delta to 3,000 edge devices
    """
    parser = argparse.ArgumentParser(
        description='YOLO Weight Delta Extraction - Step 2 of Mothership-to-Edge Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract delta from training run
  python weight_delta.py --base-model models/yolov8n_base.pt \\
                         --trained-model models/yolov8n_retrained_20260620.pt \\
                         --output deltas/delta_20260620.pth \\
                         --dataset "edge_images_batch_42"

  # With custom metadata
  python weight_delta.py --base-model base.pt --trained-model new.pt --output delta.pth \\
                         --dataset "parking_lot_v3" \\
                         --notes "Retrained on 5K images from Fleet-West"
        """
    )

    parser.add_argument('-b', '--base-model', required=True,
                        help='Path to base model file (e.g., yolov8n_base.pt)')
    parser.add_argument('-t', '--trained-model', required=True,
                        help='Path to newly trained model file from Step 1')
    parser.add_argument('-o', '--output', required=True,
                        help='Path to save delta file (e.g., delta_20260620.pth)')

    # METADATA PARAMETERS (Optional - for tracking/auditing only, do not affect delta computation)
    # --dataset: Identifies which batch of edge device images was used for training.
    #            Examples: "fleet_west_may2026_batch_42", "pedestrian_retraining_june2026"
    #            Use case: Track which image batch produced which delta for auditing,
    #                      performance analysis, and rollback scenarios.
    parser.add_argument('--dataset', default='unknown',
                        help='Dataset identifier used in training')

    # --notes: Freeform text field for additional context that doesn't fit structured fields.
    #          Examples: "Emergency retraining for bicycle false positives",
    #                    "Trained with learning_rate=0.001 for 15 epochs"
    #          Use case: Document why this run happened, configuration details, or observations.
    parser.add_argument('--notes', default='',
                        help='Additional notes for tracking')

    args = parser.parse_args()

    print("=" * 70)
    print("YOLO WEIGHT DELTA EXTRACTION - STEP 2: DELTA GENERATION")
    print("=" * 70)
    print()

    # Validate inputs
    if not os.path.exists(args.base_model):
        print(f"ERROR: Base model not found: {args.base_model}")
        sys.exit(1)
    if not os.path.exists(args.trained_model):
        print(f"ERROR: Trained model not found: {args.trained_model}")
        sys.exit(1)

    print(f"Base Model:    {args.base_model}")
    print(f"Trained Model: {args.trained_model}")
    print(f"Output Delta Path:  {args.output}")
    print()

    # Load models
    print("Loading the base model...")
    base_checkpoint = torch.load(args.base_model, map_location='cpu')
    base_state = base_checkpoint.get('model', base_checkpoint)
    if isinstance(base_state, nn.Module):
        base_state = base_state.state_dict()
    elif not isinstance(base_state, dict):
        print(f"ERROR: Invalid model format in {args.base_model}")
        print(f"Expected nn.Module or state_dict, got {type(base_state)}")
        sys.exit(1)

    print("Loading the trained model...")
    trained_checkpoint = torch.load(args.trained_model, map_location='cpu')
    trained_state = trained_checkpoint.get('model', trained_checkpoint)
    if isinstance(trained_state, nn.Module):
        trained_state = trained_state.state_dict()
    elif not isinstance(trained_state, dict):
        print(f"ERROR: Invalid model format in {args.trained_model}")
        print(f"Expected nn.Module or state_dict, got {type(trained_state)}")
        sys.exit(1)
        
    #Base model and trained model have now been loaded.

    # Detect architecture
    print("Detecting architecture...")
    try:
        prefix = detect_architecture_prefix(base_state)
        print(f"Architecture: {'YOLOX' if prefix == 'head.' else 'YOLOv26/Ultralytics'}")
        print(f"Head prefix: {prefix}")
    except ValueError as e:
        print(f"{e}")
        sys.exit(1)

    # Extract heads
    print("Extracting detection head layers...")
    base_head = extract_head_from_state(base_state)
    trained_head = extract_head_from_state(trained_state)

    base_params = sum(t.numel() for t in base_head.values())
    print(f"Extracted {len(base_head)} layers ({base_params:,} parameters)")

    # Compute delta
    print("Computing weight delta (element-wise subtraction)...")
    delta = compute_head_delta(trained_head, base_head)

    # Calculate statistics
    total_elements = sum(t.numel() for t in delta.values())
    delta_values = torch.cat([t.flatten() for t in delta.values()])
    mean_delta = delta_values.abs().mean().item()
    max_delta = delta_values.abs().max().item()

    print(f"Delta statistics:")
    print(f"  Total elements: {total_elements:,}")
    print(f"  Mean |Δ|: {mean_delta:.6f}")
    print(f"  Max |Δ|: {max_delta:.6f}")

    # Package metadata - stored inside delta file for tracking/auditing
    # This metadata does NOT affect delta computation, it's documentation only.
    # Later retrieval: payload = torch.load("delta.pth"); print(payload['metadata'])
    metadata = {
        "base_model": os.path.basename(args.base_model),
        "trained_model": os.path.basename(args.trained_model),
        "dataset": args.dataset,  # Which image batch: e.g., "fleet_west_batch_42"
        "extraction_timestamp": datetime.now().isoformat(),
        "architecture_prefix": prefix,
        "num_layers": len(delta),
        "num_parameters": total_elements,
        "mean_abs_delta": mean_delta,
        "max_abs_delta": max_delta,
        "notes": args.notes  # Freeform context: e.g., "Emergency bicycle retraining"
    }

    # Save delta
    print("Saving compressed delta payload...")
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    save_delta_payload(delta, args.output, metadata)

    # Report file size
    delta_size_bytes = os.path.getsize(args.output)
    delta_size_kb = delta_size_bytes / 1024
    delta_size_mb = delta_size_kb / 1024

    print()
    print("=" * 70)
    print("DELTA EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Output file: {args.output}")
    print(f"File size: {delta_size_kb:.2f} KB ({delta_size_mb:.3f} MB)")
    print(f"Compression: float16 (50% of original precision)")
    print()
    print("Next Step: Distribute this delta file to edge devices via your")
    print("   network pipeline (rsync/scp/OTA update mechanism)")
    print("=" * 70)


if __name__ == "__main__":
    main()
