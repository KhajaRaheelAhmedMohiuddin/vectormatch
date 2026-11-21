"""Publish VectorMatch to the Hugging Face Hub from your own machine.

Downloads the release artifacts from GitHub and creates/updates:
  * model  <user>/bge-small-vectormatch   (fine-tuned embedder + model card)
  * Space  <user>/vectormatch             (Gradio live demo, ZeroGPU hardware)

    pip install -U huggingface_hub
    hf auth login                      # paste a Write token when asked
    python publish.py                  # or: python publish.py --user <your-hf-username>
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from huggingface_hub import HfApi

RELEASE = "https://github.com/KhajaRaheelAhmedMohiuddin/vectormatch/releases/download/v1.1.0"
ASSETS = {"model": "bge-small-vectormatch.zip", "space": "vectormatch-space.zip"}


def fetch(name: str, dest: Path) -> Path:
    url = f"{RELEASE}/{name}"
    print(f"downloading {url}")
    target = dest / name
    urllib.request.urlretrieve(url, target)
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

        model_dir = fetch(ASSETS["model"], tmp)
        model_id = f"{user}/bge-small-vectormatch"
        api.create_repo(model_id, repo_type="model", exist_ok=True)
        api.upload_folder(repo_id=model_id, folder_path=model_dir, repo_type="model",
                          commit_message="Fine-tuned bge-small for CV→job retrieval (VectorMatch v1.1.0)")
        print(f"✓ model: https://huggingface.co/{model_id}")

        if args.skip_space:
            return
        space_dir = fetch(ASSETS["space"], tmp)
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
