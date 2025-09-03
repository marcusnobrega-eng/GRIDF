import re
from pathlib import Path

# === CONFIG ===
BASE_DIR = r"G:\My Drive\New folder"   # <- your folder
KEEP_VERSION = 2                       # we keep "(2)"
DRY_RUN  = True                        # set to False to actually delete/rename

# Regex to capture: "<base> (N)<ext>" where N is an integer like 1,2,3
PAREN_RE = re.compile(r"^(?P<base>.*)\s\((?P<num>\d+)\)(?P<ext>\.[^.]+)?$", re.IGNORECASE)

def group_key(p: Path):
    """Group by (parent_dir, base_without_parens, extension)."""
    m = PAREN_RE.match(p.name)
    if m:
        base = m.group("base")
        ext  = m.group("ext") or ""
        return (p.parent, base, ext.lower())
    else:
        return (p.parent, p.stem, p.suffix.lower())

def version_number(p: Path):
    """Return the version number: '(N)' -> N, unnumbered -> 0."""
    m = PAREN_RE.match(p.name)
    return int(m.group("num")) if m else 0

def main():
    root = Path(BASE_DIR)
    if not root.exists():
        print(f"Base directory does not exist: {root}")
        return

    files = [p for p in root.rglob("*") if p.is_file()]
    groups = {}
    for p in files:
        groups.setdefault(group_key(p), []).append(p)

    actions_planned = 0
    for (parent, base, ext), paths in groups.items():
        versions = {}
        for p in paths:
            v = version_number(p)
            # If multiple files share same version (rare), keep the latest by mtime
            if v in versions:
                if p.stat().st_mtime > versions[v].stat().st_mtime:
                    versions[v] = p
            else:
                versions[v] = p

        if KEEP_VERSION not in versions:
            # No "(2)" present -> skip
            continue

        keep_path = versions[KEEP_VERSION]
        target_plain = parent / f"{base}{ext}"
        to_delete = [p for v, p in versions.items() if v != KEEP_VERSION]

        print(f"\n=== GROUP: {parent}\\{base}{ext} ===")
        print(f"Keep:    {keep_path.name}")
        if to_delete:
            print("Delete: ", ", ".join(p.name for p in to_delete))
        else:
            print("Delete: (none)")

        # Execute deletions
        if not DRY_RUN:
            for p in to_delete:
                try:
                    p.unlink()
                except Exception as e:
                    print(f"  [ERROR] Could not delete {p}: {e}")

        # Rename '(2)' to plain name (overwrite plain if still exists)
        if not DRY_RUN:
            try:
                if target_plain.exists() and target_plain != keep_path:
                    target_plain.unlink()
                keep_path.rename(target_plain)
            except Exception as e:
                print(f"  [ERROR] Could not rename {keep_path.name} -> {target_plain.name}: {e}")
        else:
            print(f"Rename: {keep_path.name} -> {target_plain.name}")

        actions_planned += 1

    if actions_planned == 0:
        print("No groups with a '(2)' file were found.")
    else:
        print(f"\nPlanned {actions_planned} rename groups. DRY_RUN={DRY_RUN}")
        if DRY_RUN:
            print("Review the plan above. If it looks right, set DRY_RUN = False and run again.")

if __name__ == "__main__":
    main()
