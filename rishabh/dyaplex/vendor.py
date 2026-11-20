"""Pinned upstream code, imported in place (not copied).

third_party/ is git-ignored and recreated with ``python -m dyaplex fetch-upstream``.
"""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1] / "third_party"
PINS = {
    # Last PersonaPlex code commit before the paper (arXiv v1, June 2026). HEAD (16419a2, Nov 2026)
    # changes context to 2048 and adds prompt pinning; the paper predates it.
    "personaplex": ("https://github.com/NVIDIA/personaplex.git", "3428dfd95309a7f3c84fd93259ded0f810d1ff91"),
    # Public GestureLSM RVQ-VAE (MIT). DyaPlex D.3: "only two changes to the public GestureLSM RVQ-VAE".
    "GestureLSM": ("https://github.com/andypinxinliu/GestureLSM.git", "f52ac2f53dd1b99beb74bbcdbdf1a98118a36a70"),
}
PERSONAPLEX_CHECKPOINT = ("nvidia/personaplex-7b-v1", "fdaf4090a61cb315c138a1faee287ffd6c716309")


def fetch_upstream() -> dict:
    ROOT.mkdir(exist_ok=True)
    result = {}
    for name, (url, commit) in PINS.items():
        target = ROOT / name
        if not target.exists():
            subprocess.run(["git", "clone", "-q", "--filter=blob:none", url, str(target)], check=True)
        subprocess.run(["git", "-C", str(target), "checkout", "-q", commit], check=True)
        result[name] = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], check=True,
                                      capture_output=True, text=True).stdout.strip()
    return result


def verify() -> dict:
    found = {}
    for name, (_, commit) in PINS.items():
        target = ROOT / name
        if not target.exists():
            raise RuntimeError(f"{target} is missing; run: python -m dyaplex fetch-upstream")
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
        if head != commit:
            raise RuntimeError(f"{name} is at {head}, expected pinned {commit}")
        found[name] = head
    return found


def use_personaplex() -> None:
    path = str(ROOT / "personaplex" / "moshi")
    if path not in sys.path:
        sys.path.insert(0, path)


def use_gesturelsm() -> None:
    path = str(ROOT / "GestureLSM")
    if path not in sys.path:
        sys.path.insert(0, path)
