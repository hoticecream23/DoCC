"""One command for a whole cabinet: materialize, run, score.

``client_run`` copies the cabinet's documents out of the share, runs the
cascade over them in pack mode (or the baseline alone when the cascade is
turned off or no model is configured), and writes ``results.jsonl`` and the
client_eval report into ``<work>/<cabinet_slug>/``. The cascade runs the
baseline itself for every document, so there is one run, not two.
"""

from __future__ import annotations

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


def _run_cascade(cfg: Config, config_dir: str | None, work: Path, results: Path, root: str) -> bool:
    """Run the cascade in pack mode; False, after saying so, when no model is configured."""
    # Imported here: baseline never needs cascade unless this branch runs.
    from cascade.config import load_cascade_config, vlm_settings
    from cascade.runner import run_batch as cascade_batch
    from cascade.vlm import VLMClient

    ccfg = load_cascade_config(config_dir)
    settings = vlm_settings(ccfg)
    if not settings.model:
        log.warning("no VLM model configured (vlm.model or DOCC_VLM_MODEL): running the baseline only")
        return False
    backend = VLMClient(settings)  # refuses a host that is not loopback or allowed
    log.info("cascade server", extra={"base_url": settings.base_url, "model": settings.model, "input": settings.input})
    cascade_batch(work, results, cfg, ccfg, backend, models=Models.load(cfg, Path(root)), root=root)
    return True


def client_run(
    pack_dir: str | Path,
    cabinet: str,
    share_root: str | Path,
    work_dir: str | Path,
    config_dir: str | None = None,
    root: str = ".",
    cascade: bool = True,
    dry_run: bool = False,
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

    Returns:
        The cabinet's scores; None on a dry run.
    """
    pack = load_pack(pack_dir, cabinet)
    work = Path(work_dir) / pack.cabinet.slug
    results, report = work / RESULTS_FILE, work / REPORT_FILE
    if dry_run:
        report_dry_run([work, results, report], len(pack.manifest))
        return None
    materialize(pack_dir, cabinet, share_root, work_dir)
    client_pack = (str(pack_dir), cabinet)
    cfg = packmode.resolve(load_config_cached(config_dir), client_pack)
    if not (cascade and _run_cascade(cfg, config_dir, work, results, root)):
        run_batch(work, results, config_dir=config_dir, root=root, client_pack=client_pack)
    score = score_cabinet(pack_dir, cabinet, results)
    atomic_write_text(report, render_client_report(score, RESULTS_FILE))
    log.info("cabinet scored", extra={"report": str(report), "accuracy": round(score.accuracy, 4)})
    return score
