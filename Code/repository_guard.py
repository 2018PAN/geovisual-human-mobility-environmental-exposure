"""Fail closed when public-repository data rules are violated."""

from __future__ import annotations

import csv
import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "Data"
MAX_FILE_BYTES = 25 * 1024 * 1024

BLOCKED_DATA_DIRECTORIES = {"raw", "private", "intermediate", "population"}
BLOCKED_DATA_SUFFIXES = {
    ".parquet",
    ".pq",
    ".feather",
    ".arrow",
    ".nc",
    ".tif",
    ".tiff",
    ".joblib",
    ".pkl",
    ".pickle",
    ".zip",
    ".7z",
    ".gz",
    ".bz2",
}
SENSITIVE_COLUMNS = {
    "population",
    "estimated_population",
    "hourly_population",
    "official_population_2018",
    "preliminary_population",
    "daily_population",
    "snapshot_population",
    "app_count",
    "pre_mean_count",
    "festival_mean_count",
    "post_mean_count",
    "festival_pre_count_change",
    "post_festival_count_change",
}
SENSITIVE_IDENTIFIER = re.compile(
    r"(^|_)(device|subscriber|user|trajectory|individual|person)_?(id|key)?($|_)",
    re.IGNORECASE,
)


def iter_public_files() -> list[Path]:
    return sorted(
        path
        for path in REPO_ROOT.rglob("*")
        if path.is_file() and ".git" not in path.relative_to(REPO_ROOT).parts
    )


def csv_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return next(csv.reader(stream), [])


def main() -> int:
    violations: list[str] = []
    for path in iter_public_files():
        relative = path.relative_to(REPO_ROOT)
        if path.stat().st_size > MAX_FILE_BYTES:
            violations.append(
                f"large file ({path.stat().st_size / 1024 / 1024:.2f} MiB): {relative}"
            )
        if DATA_ROOT not in path.parents:
            continue

        data_parts = {part.casefold() for part in path.relative_to(DATA_ROOT).parts[:-1]}
        blocked_parts = data_parts.intersection(BLOCKED_DATA_DIRECTORIES)
        if blocked_parts:
            violations.append(
                f"blocked Data directory {sorted(blocked_parts)}: {relative}"
            )
        if path.suffix.casefold() in BLOCKED_DATA_SUFFIXES:
            violations.append(f"blocked Data format {path.suffix}: {relative}")
        if path.suffix.casefold() == ".csv":
            try:
                header = csv_header(path)
            except (OSError, UnicodeError, csv.Error) as exc:
                violations.append(f"unreadable CSV header: {relative}: {exc}")
                continue
            normalized = {name.strip().casefold() for name in header}
            exposed = normalized.intersection(SENSITIVE_COLUMNS)
            identifiers = sorted(name for name in normalized if SENSITIVE_IDENTIFIER.search(name))
            if exposed:
                violations.append(
                    f"population-bearing CSV columns {sorted(exposed)}: {relative}"
                )
            if identifiers:
                violations.append(
                    f"potential record identifiers {identifiers}: {relative}"
                )

    if violations:
        print("Repository disclosure guard FAILED:")
        for item in violations:
            print(f"- {item}")
        return 1

    print(
        "Repository disclosure guard passed: no blocked data locations, formats, "
        "population-bearing CSV schemas, record identifiers, or files over 25 MiB."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
