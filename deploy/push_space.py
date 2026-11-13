"""Publish the server + web app to a Hugging Face Space (Docker SDK).

    python deploy/push_space.py <hf-username>/<space-name>

Builds a clean copy with only what the Space needs (earworm/, server/, web/,
Dockerfile, the Space README) in a temporary folder, commits it and pushes it
to https://huggingface.co/spaces/<hf-username>/<space-name>. The repository
history of the Space is just "one commit per deploy"; the real history lives here.

Authentication is git's: the first push asks for your Hugging Face username and
an access token with "write" permission (Git Credential Manager remembers it).
Secrets (SUPABASE_URL, SUPABASE_SECRET_KEY) are set in the Space's settings,
never sent from here.
"""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["earworm", "server", "web", "Dockerfile"]
IGNORE = shutil.ignore_patterns("__pycache__", "*.pyc", ".env", "*.npz", "*.wav", "*.mp3", "*.m4a")


def git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout


def main(space: str) -> int:
    if space.count("/") != 1:
        print("usage: python deploy/push_space.py <hf-username>/<space-name>")
        return 2
    commit = git("rev-parse", "--short", "HEAD", cwd=ROOT).strip()
    dirty = git("status", "--porcelain", "--", *INCLUDE, cwd=ROOT).strip()
    if dirty:
        print("uncommitted changes in the deployed files; commit them first:\n" + dirty)
        return 1
    with tempfile.TemporaryDirectory(prefix="earworm-space-") as tmp:
        out = Path(tmp)
        for name in INCLUDE:
            src = ROOT / name
            if src.is_dir():
                shutil.copytree(src, out / name, ignore=IGNORE)
            else:
                shutil.copy2(src, out / name)
        shutil.copy2(ROOT / "deploy" / "space-README.md", out / "README.md")
        leaked = [p for p in out.rglob("*") if p.name == ".env"]
        if leaked:
            raise RuntimeError(f"refusing to deploy secrets: {leaked}")
        git("init", "-q", "-b", "main", cwd=out)
        git("add", "-A", cwd=out)
        git("-c", "user.name=Earworm deploy", "-c", "user.email=deploy@earworm.invalid",
            "commit", "-q", "-m", f"Deploy {commit}", cwd=out)  # fmt: skip
        url = f"https://huggingface.co/spaces/{space}"
        print(f"pushing commit {commit} to {url} ...")
        subprocess.run(["git", "push", "--force", url, "main"], cwd=out, check=True)
    print(f"done: {url} (the Space rebuilds in a few minutes)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
