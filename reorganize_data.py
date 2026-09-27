"""One-time migration script: flatten OCR facility subfolders and consolidate all_manifests.

Before:
  data/R5/{county}/{consultant}/{ocr_type}/{facility_folder}/{facility}.txt
  data/R5/{county}/{consultant}/{ocr_type}/{facility_folder}/manifest_*.txt
  data/R5/{county}/{consultant}/{ocr_type}/{facility_folder}/manifest_*.pdf
  data/all_manifests/{county}/{facility_folder}/...

After:
  data/R5/{county}/{consultant}/{ocr_type}/{facility}.txt
  data/R5/{county}/all_manifests/{facility_folder}/manifest_*.txt
  data/R5/{county}/all_manifests/{facility_folder}/manifest_*.pdf
  data/R5/{county}/all_manifests/{facility_folder}/all_manifests.pdf

Run from the repo root:
  python reorganize_data.py [--dry-run]
"""

import argparse
import shutil
import sys
from pathlib import Path

DATA_ROOT = Path("data")
OCR_SUFFIXES = ("tesseract_output", "fitz_output", "llmwhisperer_output")

MOVED = 0
DELETED = 0
DIRS_REMOVED = 0


def move(src: Path, dst: Path, dry_run: bool) -> None:
    global MOVED
    if not src.exists():
        return
    if dst.exists():
        print(f"  SKIP (exists): {src} -> {dst}")
        return
    print(f"  move: {src} -> {dst}")
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    MOVED += 1


def delete(path: Path, dry_run: bool) -> None:
    global DELETED
    if not path.exists():
        return
    print(f"  delete: {path}")
    if not dry_run:
        path.unlink()
    DELETED += 1


def rmdir_if_empty(path: Path, dry_run: bool) -> None:
    global DIRS_REMOVED
    if not path.exists() or not path.is_dir():
        return
    if any(path.iterdir()):
        print(f"  SKIP rmdir (not empty): {path}")
        return
    print(f"  rmdir: {path}")
    if not dry_run:
        path.rmdir()
    DIRS_REMOVED += 1


def migrate_ocr_subfolders(dry_run: bool) -> None:
    print("\n=== Step 1: Flatten OCR subfolders ===")
    for ocr_dir in sorted(DATA_ROOT.glob(f"R5/*/*/*")):
        if not ocr_dir.is_dir() or not ocr_dir.name.endswith("_output"):
            continue
        county = ocr_dir.parts[2]

        for facility_folder in sorted(f for f in ocr_dir.iterdir() if f.is_dir()):
            name = facility_folder.name

            # Move full-extraction txt up one level (flatten)
            # Prefer file matching folder name; fall back to any non-manifest txt
            # If dest already exists, delete source (duplicate)
            full_txts = [f for f in facility_folder.glob("*.txt") if not f.name.startswith("manifest_")]
            primary = facility_folder / f"{name}.txt"
            for txt in ([primary] if primary in full_txts else full_txts):
                dst = ocr_dir / txt.name
                if dst.exists():
                    delete(txt, dry_run)
                else:
                    move(txt, dst, dry_run)

            # Delete JSON files
            for jf in facility_folder.glob("*.json"):
                delete(jf, dry_run)

            # Move manifest files into R5/{county}/all_manifests/{facility_folder}/
            all_manifests_dir = DATA_ROOT / "R5" / county / "all_manifests" / name
            if not dry_run:
                all_manifests_dir.mkdir(parents=True, exist_ok=True)

            for pattern in ("manifest_*.txt", "manifest_*.pdf"):
                for mf in sorted(facility_folder.glob(pattern)):
                    dst = all_manifests_dir / mf.name
                    if dst.exists():
                        delete(mf, dry_run)
                    else:
                        move(mf, dst, dry_run)
            all_pdf = facility_folder / "all_manifests.pdf"
            dst_all = all_manifests_dir / "all_manifests.pdf"
            if dst_all.exists():
                delete(all_pdf, dry_run)
            else:
                move(all_pdf, dst_all, dry_run)

            # Remove now-empty facility subfolder
            rmdir_if_empty(facility_folder, dry_run)


def migrate_top_level_all_manifests(dry_run: bool) -> None:
    old_root = DATA_ROOT / "all_manifests"
    if not old_root.exists():
        print("\n=== Step 2: No top-level all_manifests/ found, skipping ===")
        return

    print("\n=== Step 2: Migrate top-level all_manifests/ into R5/{county}/all_manifests/ ===")
    for county_dir in sorted(old_root.iterdir()):
        if not county_dir.is_dir():
            continue
        county = county_dir.name

        for facility_dir in sorted(county_dir.iterdir()):
            if not facility_dir.is_dir():
                continue
            dst_dir = DATA_ROOT / "R5" / county / "all_manifests" / facility_dir.name
            if not dry_run:
                dst_dir.mkdir(parents=True, exist_ok=True)

            for f in sorted(facility_dir.iterdir()):
                if f.is_file():
                    move(f, dst_dir / f.name, dry_run)

            rmdir_if_empty(facility_dir, dry_run)

        rmdir_if_empty(county_dir, dry_run)

    rmdir_if_empty(old_root, dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print actions without executing")
    args = parser.parse_args()

    if not DATA_ROOT.exists():
        print(f"ERROR: {DATA_ROOT} not found. Run from the repo root.", file=sys.stderr)
        sys.exit(1)

    if args.dry_run:
        print("DRY RUN — no changes will be made\n")

    migrate_ocr_subfolders(args.dry_run)
    migrate_top_level_all_manifests(args.dry_run)

    print(f"\nDone: {MOVED} moved, {DELETED} deleted, {DIRS_REMOVED} dirs removed")
    if args.dry_run:
        print("(dry run — nothing was actually changed)")


if __name__ == "__main__":
    main()
