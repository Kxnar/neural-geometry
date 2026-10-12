"""Content hashes capture uncommitted experiment code as well as Git revision."""
import hashlib
from pathlib import Path


def source_hashes():
    root = Path(__file__).resolve().parent
    return {str(p.relative_to(root.parent)).replace("\\", "/"):
            hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.py"))}
