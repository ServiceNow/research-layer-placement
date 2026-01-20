#!/usr/bin/env python3
"""
Prepare a model directory for evaluation by copying tokenizer and modeling files.

Usage:
    python scripts/prepare_model_dir.py <target_model_dir>
"""

import shutil
import sys
from pathlib import Path

# Source paths
TOKENIZER_SOURCE_DIR = Path("/mnt/plato/checkpoints/upstream/Apriel-1.6-15b-Thinker")
MODELING_SOURCE_FILE = Path("/home/toolkit/code/Fast-LLM/fast_llm_external_models/apriel2/cache.py")

# Files to copy
TOKENIZER_FILES = [
    "chat_template.jinja",
    "generation_config.json",
    "preprocessor_config.json",
    "special_tokens_map.json",
    "tokenizer_config.json",
    "tokenizer.json",
]


def prepare_model_dir(target_dir: Path) -> None:
    """Copy tokenizer and modeling files to target directory."""
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Preparing model directory: {target_dir}")

    # Copy tokenizer files
    print("\nCopying tokenizer files...")
    for filename in TOKENIZER_FILES:
        source = TOKENIZER_SOURCE_DIR / filename
        target = target_dir / filename

        if not source.exists():
            print(f"  ⚠️  Warning: {source} does not exist, skipping")
            continue

        if target.exists():
            print(f"  ✗ Error: {target} already exists")
            sys.exit(1)

        shutil.copy2(source, target)
        print(f"  ✓ {filename}")

    # Copy modeling files
    print("\nCopying modeling files...")
    target = target_dir / "cache.py"

    if not MODELING_SOURCE_FILE.exists():
        print(f"  ⚠️  Warning: {MODELING_SOURCE_FILE} does not exist, skipping")
    else:
        if target.exists():
            print(f"  ✗ Error: {target} already exists")
            sys.exit(1)

        shutil.copy2(MODELING_SOURCE_FILE, target)
        print(f"  ✓ cache.py")

    print(f"\n✓ Model directory prepared: {target_dir}")


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/prepare_model_dir.py <target_model_dir>")
        sys.exit(1)

    target_dir = sys.argv[1]
    prepare_model_dir(target_dir)


if __name__ == "__main__":
    main()
