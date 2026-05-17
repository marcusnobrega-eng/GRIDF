#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GRIDF project audit script.

Purpose
-------
Inspect the GRIDF repository structure before cleaning/documenting GitHub.

This script:
1. Computes folder sizes.
2. Lists largest files.
3. Summarizes file extensions.
4. Finds likely temporary/cache/intermediate files.
5. Generates a readable folder tree.
6. Saves CSV reports to _project_audit/.

IMPORTANT:
This script DOES NOT delete anything.
It only reports candidates for manual review.
"""

from pathlib import Path
from collections import defaultdict, Counter
import csv
import os
import argparse
from datetime import datetime


# ============================================================
# Configuration
# ============================================================

DEFAULT_ROOT = Path("/Users/mngomes/Documents/GitHub/GRIDF")

EXCLUDE_DIR_NAMES = {
    ".git",
    "__pycache__",
    ".ipynb_checkpoints",
    ".DS_Store",
    ".venv",
    "venv",
    "env",
    "node_modules",
}

TEMP_FILE_PATTERNS = {
    ".DS_Store",
    "Thumbs.db",
    "desktop.ini",
}

TEMP_EXTENSIONS = {
    ".tmp",
    ".temp",
    ".bak",
    ".backup",
    ".old",
    ".log",
    ".aux",
    ".bbl",
    ".blg",
    ".out",
    ".toc",
    ".synctex.gz",
}

LARGE_FILE_MB = 100

TREE_MAX_DEPTH = 3


# ============================================================
# Helper functions
# ============================================================

def human_size(num_bytes: float) -> str:
    """Convert bytes to human-readable string."""
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)

    for unit in units:
        if size < 1024:
            return f"{size:,.2f} {unit}"
        size /= 1024

    return f"{size:,.2f} PB"


def safe_stat(path: Path):
    """Safely stat a path."""
    try:
        return path.stat()
    except Exception:
        return None


def should_skip_dir(path: Path) -> bool:
    """Return True if a directory should be skipped."""
    return path.name in EXCLUDE_DIR_NAMES


def write_csv(path: Path, rows: list, fieldnames: list):
    """Write list of dictionaries to CSV."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def relative(path: Path, root: Path) -> str:
    """Return relative path as string."""
    try:
        return str(path.relative_to(root))
    except Exception:
        return str(path)


# ============================================================
# Main audit
# ============================================================

def scan_project(root: Path):
    """
    Scan all files and folders in the project.

    Returns
    -------
    files : list of dict
    folder_sizes : dict
    folder_file_counts : dict
    extension_counter : Counter
    cleanup_candidates : list of dict
    errors : list of dict
    """

    files = []
    folder_sizes = defaultdict(int)
    folder_file_counts = defaultdict(int)
    extension_counter = Counter()
    cleanup_candidates = []
    errors = []

    for current_dir, dirnames, filenames in os.walk(root):
        current_path = Path(current_dir)

        # Remove excluded directories in-place so os.walk does not descend into them.
        dirnames[:] = [
            d for d in dirnames
            if not should_skip_dir(current_path / d)
        ]

        for filename in filenames:
            file_path = current_path / filename

            # Skip files inside excluded folders just in case.
            if any(part in EXCLUDE_DIR_NAMES for part in file_path.parts):
                continue

            st = safe_stat(file_path)
            if st is None:
                errors.append({
                    "path": relative(file_path, root),
                    "error": "Could not stat file",
                })
                continue

            size = st.st_size
            suffix = file_path.suffix.lower()
            extension_counter[suffix if suffix else "[no extension]"] += 1

            file_record = {
                "path": relative(file_path, root),
                "folder": relative(file_path.parent, root),
                "filename": file_path.name,
                "extension": suffix if suffix else "[no extension]",
                "size_bytes": size,
                "size_human": human_size(size),
                "modified_time": datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds"),
            }
            files.append(file_record)

            # Add size to folder and all parents up to root.
            parent = file_path.parent
            while True:
                folder_sizes[parent] += size
                folder_file_counts[parent] += 1
                if parent == root:
                    break
                parent = parent.parent

            # Candidate cleanup files.
            lower_name = file_path.name.lower()
            if (
                file_path.name in TEMP_FILE_PATTERNS
                or suffix in TEMP_EXTENSIONS
                or "copy" in lower_name
                or "backup" in lower_name
                or "old" in lower_name
                or "tmp" in lower_name
                or "temp" in lower_name
            ):
                cleanup_candidates.append({
                    "path": relative(file_path, root),
                    "reason": "temporary / backup / generated-file pattern",
                    "size_bytes": size,
                    "size_human": human_size(size),
                })

            if size >= LARGE_FILE_MB * 1024 * 1024:
                cleanup_candidates.append({
                    "path": relative(file_path, root),
                    "reason": f"large file >= {LARGE_FILE_MB} MB",
                    "size_bytes": size,
                    "size_human": human_size(size),
                })

    return files, folder_sizes, folder_file_counts, extension_counter, cleanup_candidates, errors


def summarize_top_folders(root: Path, folder_sizes, folder_file_counts):
    """Create folder size summary rows."""
    rows = []

    for folder, size in folder_sizes.items():
        rows.append({
            "folder": relative(folder, root),
            "size_bytes": size,
            "size_human": human_size(size),
            "file_count_recursive": folder_file_counts.get(folder, 0),
        })

    rows.sort(key=lambda x: x["size_bytes"], reverse=True)
    return rows


def summarize_extensions(extension_counter):
    """Create extension summary rows."""
    rows = []
    for ext, count in extension_counter.most_common():
        rows.append({
            "extension": ext,
            "file_count": count,
        })
    return rows


def build_tree(root: Path, folder_sizes, max_depth: int = 3):
    """Build a readable tree showing folder sizes."""
    lines = []

    def _walk(folder: Path, depth: int):
        if depth > max_depth:
            return

        indent = "    " * depth
        size = folder_sizes.get(folder, 0)
        label = folder.name if folder != root else root.name
        lines.append(f"{indent}{label}/  [{human_size(size)}]")

        try:
            children = [
                p for p in folder.iterdir()
                if p.is_dir() and not should_skip_dir(p)
            ]
        except Exception:
            return

        children.sort(key=lambda p: folder_sizes.get(p, 0), reverse=True)

        for child in children:
            _walk(child, depth + 1)

    _walk(root, 0)
    return "\n".join(lines)


def print_summary(root: Path, files, folder_rows, extension_rows, cleanup_candidates, out_dir: Path):
    """Print concise terminal summary."""
    total_size = sum(f["size_bytes"] for f in files)
    total_files = len(files)

    print("\n" + "=" * 80)
    print("GRIDF PROJECT AUDIT")
    print("=" * 80)
    print(f"Root folder : {root}")
    print(f"Total files : {total_files:,}")
    print(f"Total size  : {human_size(total_size)}")
    print(f"Reports     : {out_dir}")
    print("=" * 80)

    print("\nTop 15 folders by recursive size:")
    for row in folder_rows[:15]:
        print(f"  {row['size_human']:>12} | {row['file_count_recursive']:>7} files | {row['folder']}")

    print("\nTop 15 file extensions:")
    for row in extension_rows[:15]:
        print(f"  {row['file_count']:>7} files | {row['extension']}")

    print("\nTop 15 largest files:")
    largest_files = sorted(files, key=lambda x: x["size_bytes"], reverse=True)[:15]
    for f in largest_files:
        print(f"  {f['size_human']:>12} | {f['path']}")

    print("\nCleanup / review candidates:")
    print(f"  {len(cleanup_candidates):,} candidate records saved to cleanup_candidates.csv")
    print("\nIMPORTANT: No files were deleted.")


def main():
    parser = argparse.ArgumentParser(description="Inspect GRIDF project folder size and structure.")
    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_ROOT),
        help="Path to GRIDF project folder.",
    )
    parser.add_argument(
        "--tree-depth",
        type=int,
        default=TREE_MAX_DEPTH,
        help="Maximum folder tree depth to print/save.",
    )
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()

    if not root.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root}")

    out_dir = root / "_project_audit"
    out_dir.mkdir(exist_ok=True)

    files, folder_sizes, folder_file_counts, extension_counter, cleanup_candidates, errors = scan_project(root)

    folder_rows = summarize_top_folders(root, folder_sizes, folder_file_counts)
    extension_rows = summarize_extensions(extension_counter)
    largest_files = sorted(files, key=lambda x: x["size_bytes"], reverse=True)

    # Save CSV reports.
    write_csv(
        out_dir / "folder_sizes.csv",
        folder_rows,
        ["folder", "size_bytes", "size_human", "file_count_recursive"],
    )

    write_csv(
        out_dir / "largest_files.csv",
        largest_files,
        ["path", "folder", "filename", "extension", "size_bytes", "size_human", "modified_time"],
    )

    write_csv(
        out_dir / "file_extensions.csv",
        extension_rows,
        ["extension", "file_count"],
    )

    write_csv(
        out_dir / "cleanup_candidates_REVIEW_ONLY.csv",
        cleanup_candidates,
        ["path", "reason", "size_bytes", "size_human"],
    )

    if errors:
        write_csv(
            out_dir / "scan_errors.csv",
            errors,
            ["path", "error"],
        )

    # Save tree.
    tree_text = build_tree(root, folder_sizes, max_depth=args.tree_depth)
    with (out_dir / "folder_tree.txt").open("w", encoding="utf-8") as f:
        f.write(tree_text)

    print_summary(root, files, folder_rows, extension_rows, cleanup_candidates, out_dir)

    print("\nSaved files:")
    print(f"  {out_dir / 'folder_sizes.csv'}")
    print(f"  {out_dir / 'largest_files.csv'}")
    print(f"  {out_dir / 'file_extensions.csv'}")
    print(f"  {out_dir / 'cleanup_candidates_REVIEW_ONLY.csv'}")
    print(f"  {out_dir / 'folder_tree.txt'}")
    if errors:
        print(f"  {out_dir / 'scan_errors.csv'}")

    print("\nNext step:")
    print("  Open _project_audit/folder_sizes.csv and largest_files.csv to identify oversized folders/files.")
    print("  Open cleanup_candidates_REVIEW_ONLY.csv to manually inspect files that may be safe to remove.")


if __name__ == "__main__":
    main()