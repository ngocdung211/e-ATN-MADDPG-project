"""Optional experiment tracking for comparison training runs."""

from typing import Dict, Optional


class ExperimentTracker:
    """Small wrapper around an optional W&B run."""

    def __init__(self, run: Optional[object] = None):
        """Initialize the tracker with an optional active run."""
        self._run = run

    @property
    def enabled(self) -> bool:
        """Return whether an external tracking run is active."""
        return self._run is not None

    @property
    def url(self) -> Optional[str]:
        """Return the run URL when the backend provides one."""
        if self._run is None:
            return None
        return getattr(self._run, "url", None)

    def log(self, metrics: Dict[str, float], step: int) -> None:
        """Log one complete episode metric record."""
        if self._run is not None:
            self._run.log(metrics, step=step)

    def finish(self) -> None:
        """Flush and close the active tracking run."""
        if self._run is not None:
            self._run.finish()
            self._run = None


def initialize_experiment_tracker(
    mode: str,
    project: str,
    run_name: str,
    group: str,
    config: Dict[str, object],
    entity: str = "",
    notes: str = "",
) -> ExperimentTracker:
    """Initialize optional W&B tracking without importing it when disabled.

    Args:
        mode: W&B mode: ``disabled``, ``online``, or ``offline``.
        project: W&B project name.
        run_name: Human-readable algorithm run name.
        group: Comparison group shared by related algorithm runs.
        config: Serializable hyperparameters and experiment metadata.
        entity: Optional W&B user or team entity.
        notes: Optional experiment note.

    Returns:
        Experiment tracker containing an active W&B run, or a disabled tracker.

    Raises:
        RuntimeError: If tracking is enabled but W&B is unavailable or cannot
            initialize.
        ValueError: If the requested mode is unsupported.
    """
    normalized_mode = mode.strip().lower()
    if normalized_mode == "disabled":
        return ExperimentTracker()
    if normalized_mode not in {"online", "offline"}:
        raise ValueError("tracking mode must be disabled, online, or offline")

    try:
        import wandb
    except ImportError as error:
        raise RuntimeError(
            "W&B tracking is enabled but the 'wandb' package is not installed. "
            "Install requirements-wandb.txt or use --wandb-mode disabled."
        ) from error

    try:
        run = wandb.init(
            project=project,
            entity=entity or None,
            name=run_name,
            group=group or None,
            notes=notes or None,
            config=config,
            mode=normalized_mode,
            force=normalized_mode == "online",
            save_code=False,
        )
    except Exception as error:
        raise RuntimeError(f"Failed to initialize W&B tracking: {error}") from error
    return ExperimentTracker(run)
