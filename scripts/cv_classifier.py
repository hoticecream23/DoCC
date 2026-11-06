"""Cross-validate the classifier on stored client text, against the current model.

Reads canonical text from a results JSONL (text.full, text.pages, source_path).
The label is the folder name in source_path, mapped through config/taxonomy.yaml.
Each fold fits the same pipeline as baseline.classify.train (build_estimator)
on the training docs and scores the held-out docs two ways: model alone
(top probability >= model_min) and the full rule+model classify(). The current
model needs no fitting; it is scored on the same held-out docs. It is read
from --baseline-models (default models/previous), never from models/, because
models/ holds the model fitted on these same docs.

A doc_id filed under two folders is kept once (first by source_path), so no
copy is trained on while its twin is tested.

    python scripts/cv_classifier.py results_classification.jsonl OUT_DIR [--baseline-models DIR]
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from baseline import classify as C  # noqa: E402
from baseline.config import load_config_cached  # noqa: E402


def load_docs(path, label_map):
    docs, dropped, seen, dupes = [], collections.Counter(), set(), []
    rows = [json.loads(line) for line in open(path, encoding="utf-8")]
    for r in sorted(rows, key=lambda r: r["source_path"]):
        if r.get("doc_id") in seen:
            dupes.append(r["source_path"])
            continue
        seen.add(r.get("doc_id"))
        folder = r["source_path"].replace("\\", "/").split("/")[-2]
        cls = label_map.get(folder)
        if not cls:
            dropped[folder] += 1
            continue
        t = r["text"]
        pages = [p if isinstance(p, str) else p.get("text", "") for p in t.get("pages", [])]
        docs.append({"doc_id": r.get("doc_id"), "full": t["full"], "pages": pages, "gold": cls})
    return docs, dropped, dupes


def model_alone(model, full, model_min, unknown):
    if model is None or not full.strip():
        return unknown
    label, prob = sorted(model.predict_scores(full).items(), key=lambda kv: (-kv[1], kv[0]))[0]
    return label if prob >= model_min else unknown


def score(pairs, unknown):
    """pairs: [(gold, pred)]. Coverage, per-class precision among answered, macro F1 over gold classes."""
    answered = [(g, p) for g, p in pairs if p != unknown]
    prec = {}
    for c in sorted({p for _, p in answered}):
        hits = [g == c for g, p in answered if p == c]
        prec[c] = (sum(hits), len(hits))
    f1s = []
    for c in sorted({g for g, _ in pairs}):
        tp = sum(1 for g, p in pairs if g == c and p == c)
        npred = sum(1 for _, p in pairs if p == c)
        ngold = sum(1 for g, _ in pairs if g == c)
        P = tp / npred if npred else 0.0
        R = tp / ngold if ngold else 0.0
        f1s.append(2 * P * R / (P + R) if P + R else 0.0)
    return {
        "n": len(pairs),
        "coverage": len(answered) / len(pairs) if pairs else 0.0,
        "precision": prec,
        "min_precision": min((a / b for a, b in prec.values()), default=1.0),
        "macro_f1": sum(f1s) / len(f1s) if f1s else 0.0,
    }


def main(results, out_dir, baseline_models, folds=5, seed=0):
    from sklearn.model_selection import StratifiedKFold

    cfg = load_config_cached(ROOT / "config")
    opts = cfg.classify_opts()
    unknown = str(opts.get("unknown_label", "unknown"))
    model_min = float(opts.get("model_min_confidence", 0.45))
    docs, dropped, dupes = load_docs(results, cfg.client_label_map)
    base_dir = Path(baseline_models) / "classifier"
    current = C.load_model(base_dir)
    if current is None:
        raise SystemExit(f"no baseline model in {base_dir}")
    golds = [d["gold"] for d in docs]

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    preds = []  # one row per held-out doc
    for k, (tr, te) in enumerate(skf.split(docs, golds)):
        tr_labels = [golds[i] for i in tr]
        pipe = C.build_estimator(min(collections.Counter(tr_labels).values()))
        pipe.fit([docs[i]["full"] for i in tr], tr_labels)
        new = C.ClassifierModel(pipeline=pipe, labels=[str(c) for c in pipe.classes_])
        for i in te:
            d = docs[i]
            row = {"fold": k, "doc_id": d["doc_id"], "gold": d["gold"]}
            for name, m in (("current", current), ("new", new)):
                row[f"{name}_model"] = model_alone(m, d["full"], model_min, unknown)
                row[f"{name}_full"] = C.classify(d["full"], d["pages"], cfg, m).label
            preds.append(row)

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "cv_preds.jsonl", "w", encoding="utf-8") as fh:
        for r in preds:
            fh.write(json.dumps(r) + "\n")

    report = {
        "n_docs": len(docs),
        "class_counts": dict(collections.Counter(golds)),
        "dropped": dict(dropped),
        "duplicate_doc_ids_dropped": dupes,
        "current_model_dir": str(base_dir),
        "new_model": "fitted per fold",
        "folds": folds,
        "seed": seed,
        "per_fold": [],
        "pooled": {},
    }
    keys = ["current_model", "new_model", "current_full", "new_full"]
    for k in range(folds):
        rows = [r for r in preds if r["fold"] == k]
        report["per_fold"].append({key: score([(r["gold"], r[key]) for r in rows], unknown) for key in keys})
    for key in keys:
        report["pooled"][key] = score([(r["gold"], r[key]) for r in preds], unknown)
    (out / "cv_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("docs", len(docs), "classes", report["class_counts"], "dropped", report["dropped"])
    print("duplicate doc_id copies dropped:", dupes)
    print("current_* columns use", base_dir, "| new_* columns use a model fitted per fold")
    print("fold  " + "  ".join(f"{k:>24}" for k in keys) + "   (coverage / min class P / macro F1)")
    for k, f in enumerate(report["per_fold"]):
        print(
            f"{k:>4}  "
            + "  ".join(
                f"{f[x]['coverage']:.3f}/{f[x]['min_precision']:.3f}/{f[x]['macro_f1']:.3f}".rjust(24) for x in keys
            )
        )
    for label, fn in (
        (
            "mean",
            lambda x: [
                sum(f[x][m] for f in report["per_fold"]) / folds for m in ("coverage", "min_precision", "macro_f1")
            ],
        ),
        ("pool", lambda x: [report["pooled"][x][m] for m in ("coverage", "min_precision", "macro_f1")]),
    ):
        print(f"{label:>4}  " + "  ".join("{:.3f}/{:.3f}/{:.3f}".format(*fn(x)).rjust(24) for x in keys))
    for x in keys:
        print(x, {c: f"{a}/{b}" for c, (a, b) in report["pooled"][x]["precision"].items()})
    return report


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("results")
    ap.add_argument("out_dir")
    ap.add_argument("--baseline-models", default=str(ROOT / "models" / "previous"))
    a = ap.parse_args()
    main(a.results, a.out_dir, a.baseline_models)
