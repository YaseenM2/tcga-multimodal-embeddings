#!/usr/bin/env python3
"""Download, average and delete UNI2-h archives one TCGA project at a time.

For each project this script

1. downloads ``TCGA/TCGA-<PROJECT>.tar.gz`` from the gated Hugging Face
   dataset ``MahmoodLab/UNI2-h-features`` (your own approved account; the
   token is read from the ``HF_TOKEN`` environment variable / Colab secret),
2. mean-pools every study slide with ``aggregate_uni2_histology.py``
   (same averaging as the authors' ``embed_hist_uni2.py``),
3. deletes the archive (unless ``--keep-archives``) and writes a small
   ``_uni2_done/<project>.json`` marker with aggregate counts only.

Projects with a marker are skipped, so the command can be rerun after a
Colab disconnect. Only one archive is on disk at a time.

Licence: UNI2-h features are CC BY-NC-ND 4.0 and the access terms forbid
redistributing them. Slide vectors averaged from them are derived material:
keep them private to the account that downloaded them (do not commit them).

Example::

    python scripts/run_uni2_projects.py --splits /content/tcga10/split_cases.csv \\
        --out-root /content/tcga10 --download-dir /content/uni2_tmp
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))
import aggregate_uni2_histology as agg  # noqa: E402

REPO_ID = "MahmoodLab/UNI2-h-features"
DEFAULT_PROJECTS = ("CESC", "CHOL", "DLBC", "ESCA", "KICH", "MESO", "OV", "READ", "UCS", "UVM")


def hf_download(project: str, download_dir: Path) -> Path:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"TCGA/TCGA-{project}.tar.gz",
        local_dir=str(download_dir),
    )
    return Path(path)


def marker_path(out_root: Path, project: str) -> Path:
    return Path(out_root) / "_uni2_done" / f"TCGA-{project}.json"


def run(projects: list[str], splits: Path, out_root: Path, download_dir: Path, *,
        download: Callable[[str, Path], Path] = hf_download, keep_archives: bool = False,
        min_free_gib: float = 15.0) -> dict[str, dict]:
    study_ids = agg.load_study_ids(splits)
    results: dict[str, dict] = {}
    for raw in projects:
        project = raw.upper().removeprefix("TCGA-")
        marker = marker_path(out_root, project)
        if marker.is_file():
            results[project] = json.loads(marker.read_text())
            print(f"TCGA-{project}: already done, skipping", flush=True)
            continue
        Path(download_dir).mkdir(parents=True, exist_ok=True)
        free_gib = shutil.disk_usage(download_dir).free / 2**30
        if free_gib < min_free_gib:
            raise RuntimeError(f"Only {free_gib:.1f} GiB free in {download_dir}; need {min_free_gib}")
        started = time.time()
        print(f"TCGA-{project}: downloading ...", flush=True)
        archive = download(project, Path(download_dir))
        size_gib = archive.stat().st_size / 2**30
        print(f"TCGA-{project}: {size_gib:.2f} GiB downloaded; averaging slides ...", flush=True)
        counts = agg.process_archive(archive, study_ids, Path(out_root))
        if counts["missing_or_invalid"]:
            raise RuntimeError(f"TCGA-{project}: {counts['missing_or_invalid']} slides failed")
        summary = {**counts, "archive_gib": round(size_gib, 2),
                   "minutes": round((time.time() - started) / 60, 1)}
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text(json.dumps(summary, indent=2) + "\n")
        if not keep_archives:
            archive.unlink()
        results[project] = summary
        print(f"TCGA-{project}: {summary}", flush=True)
    return results


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--projects", nargs="+", default=list(DEFAULT_PROJECTS))
    p.add_argument("--splits", type=Path, required=True, help="study split_cases.csv")
    p.add_argument("--out-root", type=Path, required=True, help="data root for embeddings/histology")
    p.add_argument("--download-dir", type=Path, required=True)
    p.add_argument("--keep-archives", action="store_true")
    p.add_argument("--min-free-gib", type=float, default=15.0)
    args = p.parse_args(argv)
    results = run(args.projects, args.splits, args.out_root, args.download_dir,
                  keep_archives=args.keep_archives, min_free_gib=args.min_free_gib)
    total = sum(r.get("matched", 0) for r in results.values())
    print(f"Study slides with vectors across {len(results)} projects: {total}")
    print("UNI2-H PROJECTS COMPLETE!")


if __name__ == "__main__":
    main()
