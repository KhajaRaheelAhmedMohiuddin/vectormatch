"""Publish VectorMatch to the Hugging Face Hub from your own machine.

Downloads the v1.1.0 artifacts from the repo's `release-assets` branch, verifies their SHA-256
checksums, and creates/updates:
  * model  <user>/bge-small-vectormatch   (fine-tuned embedder + model card)
  * Space  <user>/vectormatch             (Gradio live demo, ZeroGPU hardware)

    pip install -U huggingface_hub
    hf auth login                      # paste a Write token when asked
    python publish.py                  # or: python publish.py --user <your-hf-username>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import HfApi

ASSETS_URL = "https://raw.githubusercontent.com/KhajaRaheelAhmedMohiuddin/vectormatch/release-assets"
ASSETS = {"model": ("bge-small-vectormatch.zip", 2), "space": ("vectormatch-space.zip", 0)}  # (name, n parts)


def _checksums() -> dict[str, str]:
    text = urllib.request.urlopen(f"{ASSETS_URL}/SHA256SUMS").read().decode()
    return {name: digest for digest, name in (line.split() for line in text.splitlines() if line.strip())}


def fetch(asset: str, dest: Path) -> Path:
    """Download (and reassemble, if split) an asset, verify its checksum, unzip it."""
    name, parts = ASSETS[asset]
    target = dest / name
    urls = [f"{ASSETS_URL}/{name}.part{i}" for i in range(parts)] if parts else [f"{ASSETS_URL}/{name}"]
    with open(target, "wb") as out_f:
        for url in urls:
            print(f"downloading {url}")
            with urllib.request.urlopen(url) as r:
                while chunk := r.read(1 << 20):
                    out_f.write(chunk)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != _checksums()[name]:
        sys.exit(f"checksum mismatch for {name}: refusing to publish a corrupted file")
    print(f"  verified sha256 {digest[:12]}…")
    out = dest / name.removesuffix(".zip")
    with zipfile.ZipFile(target) as z:
        z.extractall(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", help="Hugging Face username (default: the logged-in account)")
    ap.add_argument("--skip-space", action="store_true")
    args = ap.parse_args()

    api = HfApi()
    me = api.whoami()
    user = args.user or me["name"]
    print(f"logged in as {me['name']}; publishing under {user}")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        model_dir = fetch("model", tmp)
        model_id = f"{user}/bge-small-vectormatch"
        api.create_repo(model_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=model_id, folder_path=model_dir, repo_type="model",
                          commit_message="Fine-tuned bge-small for CV→job retrieval (VectorMatch v1.1.0)")
        print(f"✓ model: https://huggingface.co/{model_id}")

        if args.skip_space:
            return
        space_dir = fetch("space", tmp)
        if user != "KhajaHugging":  # the bundle points at the model id; keep it consistent for other accounts
            for f in (space_dir / "app.py", space_dir / "README.md"):
                f.write_text(f.read_text().replace("KhajaHugging/bge-small-vectormatch", model_id))
        space_id = f"{user}/vectormatch"
        try:
            api.create_repo(space_id, repo_type="space", space_sdk="gradio", space_hardware="zero-a10g", exist_ok=True)
        except Exception as exc:
            print(f"\n✗ could not create the Space on ZeroGPU: {exc}\n"
                  "  Free accounts can host ZeroGPU Spaces only if the email is verified and the account is 30+ days old.\n"
                  "  The model is published; once the account qualifies, simply run:  python publish.py")
            sys.exit(1)
        api.upload_folder(repo_id=space_id, folder_path=space_dir, repo_type="space",
                          commit_message="VectorMatch live demo (v1.1.0)")
        print(f"✓ Space: https://huggingface.co/spaces/{space_id}  (first build takes ~3-5 minutes)")


if __name__ == "__main__":
    main()
