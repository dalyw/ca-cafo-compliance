import os, glob
from datetime import datetime
from collections import defaultdict
import pymupdf as fitz
from helper_functions.read_report_helpers import GDRIVE_BASE

CUTOFF_TIME = datetime(2026, 1, 17, 14, 0, 0)

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# Helpers
def g(pattern):
    return [p for p in glob.glob(os.path.join(GDRIVE_BASE, pattern), recursive=True) 
            if "all_manifests" not in p]

# One-page manifests
manifest_pdfs = g("**/manifest_*.pdf")
by_pages = defaultdict(list)
for p in manifest_pdfs:
    with fitz.open(p) as doc:
        n = len(doc)
        by_pages[n].append(p)
   
folders_1pg = {os.path.dirname(p) for p in by_pages[1]}
one_page = sorted({f for folder in folders_1pg 
                   for ext in ("txt", "pdf")
                   for f in glob.glob(os.path.join(folder, f"manifest_*.{ext}"))})

# OCR outputs before time cutoff
ocr_patterns = ("**/fitz_output/**/*.txt", "**/fitz_output/**/*.json",
                "**/marker_output/**/*.txt", "**/marker_output/**/*.json")
ocr_files = [p for pat in ocr_patterns for p in g(pat) 
             if not os.path.basename(p).startswith("manifest_")]
ocr_before_cutoff = sorted([p for p in ocr_files if datetime.fromtimestamp(os.path.getmtime(p)) < CUTOFF_TIME])

# All manifests for each OCR approach
engine_patterns = {
    "fitz": "**/fitz_output/**/manifest_*",
    "marker": "**/marker_output/**/manifest_*",
    "tesseract": "**/tesseract_output/**/manifest_*",
    "llmwhisperer": "**/llmwhisperer_output/**/manifest_*",
}
engine_delete_lists = {k: sorted([p for p in g(pat) if os.path.isfile(p)]) 
                       for k, pat in engine_patterns.items()}

# Empty subdirectories under an output_type folder (marker/fitz/tesseract)
def empty_subdirs(output_folder):
    dirs = [d for d in g(f"**/{output_folder}/**/") if os.path.isdir(d)]
    dirs = sorted(set(dirs), key=lambda p: p.count(os.sep), reverse=True)
    return [d for d in dirs if os.path.isdir(d) and len(os.listdir(d)) == 0]

empty_subfolders = sum([empty_subdirs(f) for f in 
                       ["marker_output", "fitz_output", "tesseract_output", "llmwhisperer_output"]], [])

# Optional: expand to include parent dirs that become empty
# to_delete = set(empty_subfolders)
# for d in sorted(empty_subfolders, key=lambda p: p.count(os.sep), reverse=True):
#     cur = os.path.dirname(d)
#     while cur and os.path.isdir(cur):
#         if os.path.basename(cur) == "marker_output":
#             break
#         try:
#             remaining = [c for n in os.listdir(cur) 
#                         if (c := os.path.join(cur, n)) not in to_delete]
#             if not remaining:
#                 to_delete.add(cur)
#                 cur = os.path.dirname(cur)
#             else:
#                 break
#         except FileNotFoundError:
#             break
# empty_subfolders = sorted(to_delete, key=lambda p: p.count(os.sep), reverse=True)

# Write delete lists
delete_lists = {
    "delete_list_1_page.txt": one_page,
    "delete_list_ocr_before_cutoff.txt": ocr_before_cutoff,
    "delete_list_all_fitz.txt": engine_delete_lists["fitz"],
    "delete_list_all_marker.txt": engine_delete_lists["marker"],
    "delete_list_all_tesseract.txt": engine_delete_lists["tesseract"],
    "delete_list_all_llmwhisperer.txt": engine_delete_lists["llmwhisperer"],
    "delete_list_empty_subfolders.txt": empty_subfolders,
}

for fname, items in delete_lists.items():
    with open(os.path.join(OUT_DIR, fname), "w", encoding="utf-8") as f:
        f.write("\n".join(items) + "\n")


# Deletion commands:
#   while IFS= read -r f; do rm -f "$f"; done < ca_cafo_compliance/outputs/delete_list_all_fitz.txt
#   while IFS= read -r f; do rm -f "$f"; done < ca_cafo_compliance/outputs/delete_list_all_marker.txt
#   while IFS= read -r f; do rm -f "$f"; done < ca_cafo_compliance/outputs/delete_list_all_tesseract.txt
#   while IFS= read -r f; do rm -f "$f"; done < ca_cafo_compliance/outputs/delete_list_all_llmwhisperer.txt
#   while IFS= read -r d; do rmdir "$d" 2>/dev/null; done < ca_cafo_compliance/outputs/delete_list_empty_subfolders.txt