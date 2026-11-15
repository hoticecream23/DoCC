"""One command for a whole cabinet: materialize, run, score.

``client_run`` copies the cabinet's documents out of the share, runs the
cascade over them in pack mode (or the baseline alone when the cascade is
turned off or no model is configured), and writes ``results.jsonl`` and the
client_eval report into ``<work>/<cabinet_slug>/``. The cascade runs the
baseline itself for every document, so there is one run, not two.

Each run starts from an empty results file, so a changed config or a newly
turned-on model stage never scores stale lines; ``resume`` opts out. The old
report is moved to ``client_eval.md.previous`` first, the new one is written
only after scoring, and a failed run leaves a short stub naming the stage.
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

from . import packmode
from .clientpack import load_pack, materialize
from .config import Config, load_config_cached
from .evaluate.client_eval import CabinetScore, render_client_report, score_cabinet
from .io import atomic_write_text, report_dry_run
from .logging_setup import get_logger
from .pipeline import Models, run_batch

log = get_logger(__name__)

RESULTS_FILE = "results.jsonl"
REPORT_FILE = "client_eval.md"
PROVENANCE_SUFFIX = ".provenance"  # the cascade's sidecar next to the results file
PREVIOUS_SUFFIX = ".previous"  # the report of the run before this one


def _run_cascade(cfg: Config, config_dir: str | None, work: Path, results: Path, root: str, resume: bool) -> str:
    """Run the cascade in pack mode and return the model name; "", after saying so, when none is configured."""
    # Imported here: baseline never needs cascade unless this branch runs.
    from cascade.config import load_cascade_config, vlm_settings
    from cascade.runner import run_batch as cascade_batch
    from cascade.vlm import VLMClient

    ccfg = load_cascade_config(config_dir)
    settings = vlm_settings(ccfg)
    if not settings.model:
        log.warning("no VLM model configured (vlm.model or DOCC_VLM_MODEL): running the baseline only")
        return ""
    backend = VLMClient(settings)  # refuses a host that is not loopback or allowed
    log.info("cascade server", extra={"base_url": settings.base_url, "model": settings.model, "input": settings.input})
    models = Models.load(cfg, Path(root))
    cascade_batch(work, results, cfg, ccfg, backend, models=models, root=root, resume=resume)
    return settings.model


def _run_header(started: datetime, seconds: float, model: str, results: Path) -> list[str]:
    stages = f"baseline + model ({model})" if model else "baseline only"
    return [
        f"Run started: {started.isoformat(timespec='seconds')} ({seconds:.0f} s)",
        f"Stages: {stages}",
        f"Results file: `{results}`",
    ]


def _run(
    pack_dir: str, cabinet: str, config_dir: str | None, work: Path, root: str, cascade: bool, resume: bool
) -> str:
    """Run the cascade, or the baseline alone, over the work folder; return the model name or ""."""
    results, client_pack = work / RESULTS_FILE, (pack_dir, cabinet)
    cfg = packmode.resolve(load_config_cached(config_dir), client_pack)
    model = _run_cascade(cfg, config_dir, work, results, root, resume) if cascade else ""
    if not model:
        run_batch(work, results, config_dir=config_dir, root=root, client_pack=client_pack, resume=resume)
    return model


def _clear_previous(results: Path, report: Path) -> None:
    """Start clean: drop the old results and move the old report aside, so neither looks current."""
    results.unlink(missing_ok=True)
    Path(f"{results}{PROVENANCE_SUFFIX}").unlink(missing_ok=True)
    if report.exists():
        report.replace(f"{report}{PREVIOUS_SUFFIX}")


def _failure_stub(cabinet: str, stage: str, started: datetime) -> str:
    failed = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return "\n".join(
        [
            f"# Client evaluation: {cabinet}",
            "",
            f"The run failed at stage `{stage}` at {failed} (started {started.isoformat(timespec='seconds')}).",
            "No scores were written. See the log for the error.",
            "",
        ]
    )


def client_run(
    pack_dir: str | Path,
    cabinet: str,
    share_root: str | Path,
    work_dir: str | Path,
    config_dir: str | None = None,
    root: str = ".",
    cascade: bool = True,
    dry_run: bool = False,
    resume: bool = False,
) -> CabinetScore | None:
    """Materialize one cabinet, run it in pack mode, and score it against the pack.

    Args:
        pack_dir: The client pack.
        cabinet: Cabinet name or slug.
        share_root: The share the inventory was taken of.
        work_dir: Work directory, outside the share.
        config_dir: Config directory; None for the default.
        root: Directory model paths are relative to.
        cascade: Run the model stage when a model is configured.
        dry_run: Log what would be written and stop.
        resume: Keep the results of an earlier run and process only new documents.

    Returns:
        The cabinet's scores; None on a dry run.
    """
    pack = load_pack(pack_dir, cabinet)
    work = Path(work_dir) / pack.cabinet.slug
    results, report = work / RESULTS_FILE, work / REPORT_FILE
    if dry_run:
        report_dry_run([work, results, report], len(pack.manifest))
        return None
    work.mkdir(parents=True, exist_ok=True)
    if not resume:
        _clear_previous(results, report)
    started, clock, stage = datetime.now(timezone.utc), time.monotonic(), "materialize"
    try:
        materialize(pack_dir, cabinet, share_root, work_dir)
        stage = "run"
        model = _run(str(pack_dir), cabinet, config_dir, work, root, cascade, resume)
        stage = "score"
        score = score_cabinet(pack_dir, cabinet, results)
    except Exception:
        atomic_write_text(report, _failure_stub(pack.cabinet.name, stage, started))
        raise
    header = _run_header(started, time.monotonic() - clock, model, results.resolve())
    atomic_write_text(report, render_client_report(score, header=header))
    log.info("cabinet scored", extra={"report": str(report), "accuracy": round(score.accuracy, 4)})
    return score
