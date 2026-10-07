#!/usr/bin/env python3
"""Fetch the pinned public checkpoint; normal inference never needs a network."""
import argparse
import hashlib
import json
import pathlib
import subprocess

MODEL = "Qwen/Qwen3-ASR-0.6B"
REVISION = "5eb144179a02acc5e5ba31e748d22b0cf3e303b0"
WEIGHT_SIZE = 1876091704
WEIGHT_SHA256 = "79d6cbd4c98c7bbffe9db2edac07f56cd6637d0d5944b27f6c2b8353840323ea"
FILES = ["config.json", "preprocessor_config.json", "generation_config.json", "tokenizer_config.json", "chat_template.json", "vocab.json", "merges.txt", "README.md", "model.safetensors"]

def digest(path):
    sha = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=pathlib.Path, default=pathlib.Path("models/Qwen3-ASR-0.6B"))
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    manifest = {"model": MODEL, "revision": REVISION, "license": "Apache-2.0", "files": {}}
    for name in FILES:
        destination = args.directory / name
        if not destination.exists():
            temporary = destination.with_suffix(destination.suffix + ".partial")
            print(f"Downloading {name}", flush=True)
            url = f"https://huggingface.co/{MODEL}/resolve/{REVISION}/{name}?download=true"
            subprocess.run(["curl", "--location", "--fail", "--retry", "3", "--continue-at", "-", "--output", str(temporary), url], check=True)
            if name == "model.safetensors":
                if temporary.stat().st_size != WEIGHT_SIZE or digest(temporary) != WEIGHT_SHA256:
                    raise RuntimeError("Checkpoint size or SHA-256 does not match the pinned official model")
            temporary.replace(destination)
        sha = digest(destination)
        if name == "model.safetensors" and (sha != WEIGHT_SHA256 or destination.stat().st_size != WEIGHT_SIZE):
            raise RuntimeError("Existing model.safetensors does not match the pinned official checkpoint")
        manifest["files"][name] = {"sha256": sha, "bytes": destination.stat().st_size}
    (args.directory / "download-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {MODEL}@{REVISION} in {args.directory}", flush=True)

if __name__ == "__main__":
    main()
