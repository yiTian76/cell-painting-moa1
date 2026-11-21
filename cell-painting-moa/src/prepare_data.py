"""Rebuild two fixed-dose Cell Painting cohorts from a pinned Broad LINCS release.

Python >= 3.10; standard library only. Use --help for paths. Source data are CC0.
"""
import argparse
import csv
import gzip
import hashlib
import io
import json
import math
import re
import shutil
import statistics
import tempfile
import urllib.request
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path


REVISION = "061870127481dcd73c29df85ebcfddeac2ed0828"
REPO = "broadinstitute/lincs-cell-painting"
RAW = f"https://raw.githubusercontent.com/{REPO}/{REVISION}"
MEDIA = f"https://media.githubusercontent.com/media/{REPO}/{REVISION}"
BATCH = "2016_04_01_a549_48hr_batch1"
SOURCES = {
    "consensus_full.csv.gz": (
        f"{MEDIA}/consensus/{BATCH}/{BATCH}_consensus_modz_dmso.csv.gz",
        "23037a53106773767efdf17b3d074d9f4439bd56b046ba2fe7c3f5f993da0222",
    ),
    "resolved.tsv": (
        f"{RAW}/metadata/moa/repurposing_info_external_moa_map_resolved.tsv",
        "fa06cb8310a5cf801f5f1ba39bf8176a1b4eefdb4a23d1739ba92be31632b6be",
    ),
    "blocklist.txt": (f"{RAW}/utils/consensus_blocklist.txt",
                      "1fcbdbc668ab831a8366453e395fc3e64287a869c94af7dbb927410e38461516"),
    "LICENSE_CC0.md": (f"{RAW}/LICENSE_CC0.md",
                       "588da6ff6a65a262d9538719624f271b6aee2f244c3bc43d71cf5115fcd14ac9"),
}
DOSES = {10.0: "10uM", 3.3333: "3p33uM"}


def sha256(path):
    with Path(path).open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch_sources(cache):
    cache.mkdir(parents=True, exist_ok=True)
    for name, (url, digest) in SOURCES.items():
        path = cache / name
        if path.is_file():
            if sha256(path) != digest:
                raise ValueError(f"Cached input failed SHA-256 verification: {path}")
            continue
        with tempfile.NamedTemporaryFile(dir=cache, delete=False) as staged:
            tmp = Path(staged.name)
            try:
                with urllib.request.urlopen(url, timeout=150) as source:
                    shutil.copyfileobj(source, staged)
            except Exception:
                staged.close()
                tmp.unlink(missing_ok=True)
                raise
        if sha256(tmp) != digest:
            tmp.unlink()
            raise ValueError(f"Downloaded input failed SHA-256 verification: {url}")
        tmp.replace(path)


def field_reason(name, block):
    if not name.startswith(("Cells_", "Cytoplasm_", "Nuclei_")):
        return "metadata"
    if name in block:
        return "upstream_blocklist"
    family = name.split("_")[1]
    if (family in {"Children", "Parent", "Number", "Location"}
            or "_AreaShape_Center_" in name or "ClosestObjectNumber" in name):
        return "position_or_segmentation_bookkeeping"
    return "retained"


def identity_reason(records):
    """All annotated source records for a connectivity block must agree."""
    labels = {r["moa"].strip() for r in records}
    if labels & {"", "unknown"}:
        return "unknown_primary_moa"
    if any("|" in label or ";" in label for label in labels):
        return "multiple_primary_moa"
    if len(labels) != 1:
        return "inconsistent_primary_moa"
    label = next(iter(labels)).casefold()
    for row in records:
        alternatives = re.split(r"[|;]", row["alternative_moa"])
        if any(v.strip() and v.strip().casefold() != label for v in alternatives):
            return "conflicting_alternative_moa"
    return "retained"


def load_labels(source):
    by_sample = {}
    by_key = defaultdict(list)
    with source.open(encoding="utf-8", newline="") as handle:
        for r in csv.DictReader(handle, delimiter="\t"):
            if r["broad_sample"] in by_sample:
                raise ValueError(f"Ambiguous material ID: {r['broad_sample']}")
            by_sample[r["broad_sample"]] = r
            if re.fullmatch("[A-Z]{14}", r["InChIKey14"]):
                by_key[r["InChIKey14"]].append(r)
    return by_sample, {k: identity_reason(v) for k, v in by_key.items()}


def finite_value(raw):
    if not raw.strip() or raw.strip().lower() in {"na", "nan", "null", "none"}:
        return None
    val = float(raw)
    return val if math.isfinite(val) else None


@contextmanager
def write_gzip(path):
    with path.open("wb") as disk:
        with gzip.GzipFile(fileobj=disk, filename="", mode="wb", mtime=0, compresslevel=5) as gz:
            with io.TextIOWrapper(gz, encoding="utf-8", newline="") as handle:
                yield handle


def build(cache, dest):
    fetch_sources(cache)
    dest.mkdir(parents=True, exist_ok=True)
    block = set((cache / "blocklist.txt").read_text(encoding="utf8").splitlines()[1:])
    material, status = load_labels(cache / "resolved.tsv")
    groups = {dose: defaultdict(list) for dose in DOSES}
    flow = {dose: Counter() for dose in DOSES}
    raw_count = 0
    excluded_file = dest / "sample_flow.csv"
    with (gzip.open(cache / "consensus_full.csv.gz", "rt", encoding="utf8", newline="") as source,
          excluded_file.open("w", encoding="utf8", newline="") as log):
        audit = csv.writer(log)
        audit.writerow(["dose_uM", "broad_sample", "connectivity_key", "reason"])
        stream = csv.reader(source)
        header = next(stream)
        col = {name: idx for idx, name in enumerate(header)}
        reasons = [field_reason(f, block) for f in header]
        feature_ids = [i for i, reason in enumerate(reasons) if reason == "retained"]
        for r in stream:
            raw_count += 1
            if r[col["Metadata_cell_id"]] != "A549" or r[col["Metadata_time_point"]] != "48H":
                continue
            dose = float(r[col["Metadata_mmoles_per_liter"]].strip())
            if dose not in DOSES:
                continue
            sample = r[col["Metadata_broad_sample"]]
            annot = material.get(sample)
            key = annot["InChIKey14"] if annot else ""
            if annot is None:
                reason = "no_annotation"
            elif key not in status:
                reason = "missing_connectivity"
            elif status[key] != "retained":
                reason = status[key]
            elif r[col["Metadata_moa"]].strip() != annot["moa"].strip():
                reason = "profile_annotation_disagreement"
            else:
                reason = "retained"
            flow[dose][reason] += 1
            audit.writerow([dose, sample, key, reason])
            if reason == "retained":
                groups[dose][key].append((
                    sample, r[col["Metadata_Plate_Map_Name"]], r[col["Metadata_pert_well"]],
                    [finite_value(r[i]) for i in feature_ids],
                ))
    with (dest / "feature_manifest.csv").open("w", encoding="utf8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["source_column", "use"])
        writer.writerows(zip(header, reasons))
    manifest = {
        "revision": REVISION, "sources": {n: {"url": u, "sha256": h} for n, (u, h) in SOURCES.items()},
        "source_license": "CC0 1.0", "source_rows_all_doses": raw_count,
        "feature_counts": dict(Counter(reasons)), "source_annotation_rows": len(material), "doses": {},
    }
    for dose, name in DOSES.items():
        path = dest / f"profiles_{name}.csv.gz"
        class_counts = Counter()
        missing_values = 0
        with write_gzip(path) as out:
            writer = csv.writer(out, lineterminator="\n")
            fields = ["compound_id", "compound_name", "moa", "Metadata_broad_samples", "Metadata_dose_uM",
                      "Metadata_plate_maps", "Metadata_wells", "Metadata_merged_samples"]
            writer.writerow(fields + [header[i] for i in feature_ids])
            for key, records in sorted(groups[dose].items()):
                samples = sorted({r[0] for r in records})
                names = sorted({material[s]["pert_iname"].strip() for s in samples} - {""})
                label = material[samples[0]]["moa"].strip()
                class_counts[label] += 1
                if len(records) == 1:
                    values = records[0][3]
                else:
                    columns = zip(*(r[3] for r in records))
                    values = [statistics.median(valid) if (valid := [x for x in c if x is not None])
                              else None for c in columns]
                missing_values += sum(x is None for x in values)
                writer.writerow([key, "|".join(names), label, "|".join(samples), dose,
                                 "|".join(sorted({r[1] for r in records})),
                                 "|".join(sorted({r[2] for r in records})), len(records)] + values)
        manifest["doses"][name] = {
            "dose_uM": dose, "raw_exact_dose_rows": sum(flow[dose].values()),
            "source_row_flow": dict(flow[dose]), "output_unique_compounds": len(groups[dose]),
            "label_counts": dict(class_counts.most_common()), "nonfinite_values_as_missing": missing_values,
            "output_sha256": sha256(path),
        }
    shutil.copyfile(cache / "LICENSE_CC0.md", dest / "LICENSE_CC0.md")
    (dest / "data_provenance.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf8")
    for name, details in manifest["doses"].items():
        print(name, details["output_unique_compounds"], "unique compounds,", len(details["label_counts"]), "mechanisms")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=Path("raw_data"), help="verified source download cache")
    parser.add_argument("--output", type=Path, default=Path("data"), help="destination for the processed cohorts")
    args = parser.parse_args()
    build(args.cache, args.output)
