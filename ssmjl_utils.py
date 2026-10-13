"""Validation shared by the pretraining and downstream training scripts."""

from pathlib import Path


def class_mapping(root, class_to_idx=None):
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Dataset directory not found: {root}")
    names = sorted(path.name for path in root.iterdir() if path.is_dir())
    if class_to_idx is None:
        if not names:
            raise ValueError(f"No class directories found under {root}")
        return {name: index for index, name in enumerate(names)}
    mapping = dict(class_to_idx)
    if sorted(mapping.values()) != list(range(len(mapping))):
        raise ValueError("Class indices must be contiguous and start at zero")
    unknown = set(names) - set(mapping)
    if unknown:
        raise ValueError(f"Unknown validation classes under {root}: {sorted(unknown)}")
    return mapping


def validate_bands(bands, in_chans=None):
    bands = list(bands)
    if not bands or any(band < 0 for band in bands) or len(set(bands)) != len(bands):
        raise ValueError("Band indices must be a nonempty list of distinct nonnegative integers")
    if in_chans is not None and len(bands) != in_chans:
        raise ValueError(f"Selected {len(bands)} bands, but in_chans is {in_chans}")
    return bands


def load_checkpoint(path):
    """Read tensor checkpoints, including raw and commonly wrapped state dictionaries."""
    import torch

    # ConvMAE training checkpoints also contain an argparse.Namespace and
    # optimizer metadata, so preserve support for the repository's own format.
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    for key in ('model', 'state_dict', 'model_state_dict'):
        if isinstance(checkpoint, dict) and isinstance(checkpoint.get(key), dict):
            checkpoint = checkpoint[key]
            break
    if not isinstance(checkpoint, dict) or not checkpoint:
        raise ValueError(f"No state dictionary found in checkpoint: {path}")
    return {key.removeprefix('module.'): value for key, value in checkpoint.items()}


def load_required_weights(module, state_dict, required_keys=None):
    """Refuse partial pretrained encoders instead of silently using random parameters."""
    expected = module.state_dict()
    required = set(expected if required_keys is None else required_keys)
    missing = sorted(required - set(state_dict))
    mismatched = sorted(
        key for key in required & set(state_dict)
        if not hasattr(state_dict[key], 'shape') or state_dict[key].shape != expected[key].shape
    )
    if missing or mismatched:
        raise ValueError(
            f"Incompatible checkpoint: missing keys={missing}; shape mismatches={mismatched}"
        )
    module.load_state_dict({key: state_dict[key] for key in required}, strict=False)
    return len(required)


def prepare_legacy_timm():
    """Supply the Python aliases timm 0.3.2 expects from removed torch._six APIs."""
    import collections.abc
    import importlib
    import sys
    import types

    try:
        six = importlib.import_module('torch._six')
    except ModuleNotFoundError:
        six = types.ModuleType('torch._six')
        sys.modules['torch._six'] = six
    for name, value in {
        'container_abcs': collections.abc,
        'string_classes': (str,),
        'int_classes': (int,),
    }.items():
        if not hasattr(six, name):
            setattr(six, name, value)
