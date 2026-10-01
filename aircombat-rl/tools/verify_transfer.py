"""Verify the archived experiment files after git clone (standard library only)."""
import hashlib
import json
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    manifest = root / "docs/verification/artifact_manifest.json"
    if not manifest.exists():
        raise SystemExit("Run this tool from the RL_KAU transfer checkout containing docs/verification/.")
    records = json.loads(manifest.read_text(encoding="utf-8"))["files"]
    errors = []
    for record in records:
        path = (root / record["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            errors.append(record["path"] + ": missing or outside checkout")
        elif path.stat().st_size != record["bytes"] or hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            errors.append(record["path"] + ": content differs from archived experiment")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Verified {len(records)} archived experiment files, including model weights and replay buffers.")


if __name__ == "__main__":
    main()
