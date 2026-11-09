import argparse
from hashlib import file_digest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import wandb

from .dataset import PendulumDataset
from .evaluation import evaluate
from .trainer import train


CONDITIONS = ("l", "m", "b", "theta0", "omega0")
RELATIVE_ERRORS = (
    ("theta_relative_error_pct", "Angle (θ)"),
    ("omega_relative_error_pct", "Angular velocity (ω)"),
)


def _run_summary(history, rollouts):
    metrics_by_case = {rollout["case"]: rollout["metrics"] for rollout in rollouts}
    last_epoch = history[-1]
    summary = {
        "train_loss": last_epoch["train_loss"],
        "val_loss": last_epoch["val_loss"],
    }
    for case in ("unseen", "learned"):
        if case in metrics_by_case:
            for key, _ in RELATIVE_ERRORS:
                summary[f"{case}_rollout_{key}"] = metrics_by_case[case][key]
    return summary


def train_run(
    run: wandb.Run,
    *,
    learned_trajectory: int | None = None,
    unseen_case: dict[str, float] | None = None,
) -> None:
    """Train and evaluate one configuration assigned by the sweep agent."""
    config = run.config
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_file():
        raise FileNotFoundError(f"{dataset_path} is missing. Set dataset_path in the sweep YAML.")
    dataset = PendulumDataset(dataset_path)
    if dataset.num_trajectories < 1:
        raise ValueError("A sweep requires a compact dataset with at least one trajectory.")

    time_step = float(dataset.metadata.get("time_step", 0))
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("The dataset metadata must contain a finite positive time_step.")
    num_steps = dataset.steps_per_trajectory
    duration = num_steps * time_step
    framerate = 1 / time_step
    start_time = float(dataset.metadata.get("start_time", 0))
    num_epochs = config["num_epochs"]
    if type(num_epochs) is not int or num_epochs < 1:
        raise ValueError("num_epochs must be a positive integer.")
    relative_error_floor = float(config["relative_error_floor"])
    if not np.isfinite(relative_error_floor) or relative_error_floor <= 0:
        raise ValueError("relative_error_floor must be finite and positive.")

    if learned_trajectory is None:
        learned_trajectory = config.get("learned_trajectory", 0)
    if type(learned_trajectory) is not int or not 0 <= learned_trajectory < dataset.num_trajectories:
        raise ValueError("learned_trajectory must be a zero-based trajectory index in the dataset.")
    learned_case = dict(zip(CONDITIONS, dataset.initial_conditions(learned_trajectory), strict=True))
    cases = {"learned": learned_case}
    if unseen_case is None:
        unseen_case = config.get("unseen_case")
    if unseen_case is not None:
        if not isinstance(unseen_case, dict) or set(unseen_case) != set(CONDITIONS):
            raise ValueError("unseen_case must specify l, m, b, theta0 and omega0.")
        values = tuple(unseen_case[name] for name in CONDITIONS)
        if any(type(value) not in (int, float) for value in values) or not np.isfinite(values).all():
            raise ValueError("unseen_case must contain finite numbers.")
        if unseen_case["l"] <= 0 or unseen_case["m"] <= 0:
            raise ValueError("unseen_case requires positive length and mass.")
        if dataset.has_initial_conditions(values):
            raise ValueError("unseen_case matches a trajectory in the training dataset.")
        cases["unseen"] = unseen_case

    all_indices = np.arange(len(dataset))
    with dataset_path.open("rb") as source:
        dataset_sha256 = file_digest(source, "sha256").hexdigest()
    config.update({
        "dataset_sha256": dataset_sha256,
        "time_step": time_step,
        "epoch_evaluation_data": "training_transitions",
    })

    run_dir = Path("runs") / run.id
    run_dir.mkdir(parents=True, exist_ok=True)
    run.define_metric("epoch", hidden=True, summary="none")
    run.define_metric("loss/train", step_metric="epoch", summary="none")
    run.define_metric("loss/eval", step_metric="epoch", summary="none")
    history = []

    def log_epoch(epoch: int, train_loss: float, val_loss: float) -> None:
        metrics = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(metrics)
        run.log({"epoch": epoch, "loss/train": train_loss, "loss/eval": val_loss})

    model_path = train(
        dataset,
        model_dir=run_dir,
        hidden_size=config["hidden_size"],
        num_hidden_layers=config["num_hidden_layers"],
        learning_rate=config["learning_rate"],
        train_batch_size=config["train_batch_size"],
        eval_batch_size=config["eval_batch_size"],
        num_epochs=num_epochs,
        split=(all_indices, all_indices),
        seed=config["seed"],
        on_epoch=log_epoch,
    )

    rollouts = []
    plots = {}
    error_panels = ["<h3>Relative errors</h3>"]
    for case, conditions in cases.items():
        plot_path = run_dir / f"evaluation_{case}.png"
        metrics = evaluate(
            model_path,
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            duration=duration,
            framerate=framerate,
            start_time=start_time,
            show=False,
            relative_error_floor=relative_error_floor,
            plot_path=plot_path,
            **conditions,
        )
        rollouts.append({
            "case": case,
            "initial_conditions": conditions,
            "metrics": metrics,
            "plot": plot_path.name,
        })
        plots[f"evaluation/{case}"] = wandb.Image(str(plot_path))
        values = " · ".join(
            f"{label} <strong>{metrics[key]:.2f}%</strong>" for key, label in RELATIVE_ERRORS
        )
        error_panels.append(f"<h4>{case.title()}</h4><p>{values}</p>")
        print(
            f"{case.title()} rollout relative errors: "
            f"theta {metrics['theta_relative_error_pct']:.2f}%, "
            f"omega {metrics['omega_relative_error_pct']:.2f}%"
        )

    summary = _run_summary(history, rollouts)
    report = {
        "run_id": run.id,
        "config": dict(config),
        "history": history,
        "summary": summary,
        "rollouts": rollouts,
        "checkpoint": model_path.name,
        "plot": rollouts[0]["plot"],
        "plot_trajectory": learned_trajectory,
    }
    with TemporaryDirectory(dir=run_dir) as temp:
        report_path = Path(temp) / "metrics.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report_path.replace(run_dir / "metrics.json")

    run.summary.update(summary)
    run.log({**plots, "relative_errors": wandb.Html("".join(error_panels), inject=False)})
    print(f"Run saved to {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate one W&B sweep run.")
    parser.add_argument("--learned-trajectory", type=int, help="Index of the training trajectory to evaluate.")
    parser.add_argument("--unseen-case", type=float, nargs=5, metavar=CONDITIONS)
    args = parser.parse_args()
    unseen_case = None
    if args.unseen_case is not None:
        unseen_case = dict(zip(CONDITIONS, args.unseen_case, strict=True))

    with wandb.init() as run:
        train_run(run, learned_trajectory=args.learned_trajectory, unseen_case=unseen_case)


if __name__ == "__main__":
    main()
