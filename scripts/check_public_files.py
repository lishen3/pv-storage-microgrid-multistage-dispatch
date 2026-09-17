"""Fail CI when tracked files contain common private artifacts or secrets."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_SUFFIXES = {
    ".xlsx", ".xls", ".csv", ".tsv", ".zip", ".pdf", ".png",
    ".db", ".sqlite", ".sqlite3", ".pem", ".key", ".p12", ".pfx",
}
FORBIDDEN_PARTS = {"data/raw", "__pycache__", ".venv", ".idea", ".vscode"}
TEXT_SUFFIXES = {
    ".py", ".md", ".txt", ".yml", ".yaml", ".json", ".toml", ".ini",
    ".cfg", ".env", "",
}
PATTERNS = {
    "Windows user path": re.compile(r"[A-Za-z]:[\\/]Users[\\/][^\\/\s]+", re.I),
    "Unix home path": re.compile(r"/(?:Users|home)/[^/\s]+"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "GitHub token": re.compile(r"\b(?:ghp|github_pat)_[A-Za-z0-9_]{20,}\b"),
    "generic secret assignment": re.compile(
        r"(?i)\b(?:api[_-]?key|access[_-]?token|secret|password)\b\s*[:=]\s*['\"][^'\"\s]{8,}['\"]"
    ),
}


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
    )
    return [ROOT / item.decode("utf-8") for item in result.stdout.split(b"\0") if item]


def main() -> None:
    problems: list[str] = []
    for path in tracked_files():
        relative = path.relative_to(ROOT).as_posix()
        lower = relative.lower()
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden tracked file type: {relative}")
        if any(part in lower for part in FORBIDDEN_PARTS):
            problems.append(f"forbidden tracked path: {relative}")
        if path.suffix.lower() not in TEXT_SUFFIXES or not path.is_file():
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            if pattern.search(content):
                problems.append(f"{label}: {relative}")

    if problems:
        raise SystemExit("Public-release check failed:\n- " + "\n- ".join(sorted(set(problems))))
    print(f"Public-release check passed for {len(tracked_files())} tracked files.")


if __name__ == "__main__":
    main()
