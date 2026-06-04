"""Training-step helpers shared by notebooks and scripts."""

from src.utils.training.helpers import compute_loss, compute_metrics, update_average
from src.utils.training.swiss_roll_notebook import (
    CometExperiment,
    build_ema_model_copy,
    build_periodic_loss_payload,
    build_run_metadata,
    build_swiss_roll_context,
    compose_swiss_roll_cfg,
    infer_cost_label,
    log_optional_metrics,
    make_adam,
    normalize_experiment_key,
    should_run,
)
from src.utils.training.weather_notebook import (
    build_weather_samplers,
    compose_weather_cfg,
    load_weather_tensors,
    paired_sampler_weather,
    unpaired_sampler_weather,
)

__all__ = [
    "CometExperiment",
    "build_ema_model_copy",
    "build_periodic_loss_payload",
    "build_run_metadata",
    "build_swiss_roll_context",
    "build_weather_samplers",
    "compose_swiss_roll_cfg",
    "compose_weather_cfg",
    "compute_loss",
    "compute_metrics",
    "infer_cost_label",
    "load_weather_tensors",
    "log_optional_metrics",
    "make_adam",
    "normalize_experiment_key",
    "paired_sampler_weather",
    "should_run",
    "unpaired_sampler_weather",
    "update_average",
]
