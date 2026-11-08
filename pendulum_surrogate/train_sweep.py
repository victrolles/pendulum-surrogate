from hashlib import file_digest
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from sklearn.model_selection import train_test_split
import wandb

from .dataset import PendulumDataset
from .evaluation import evaluate
from .trainer import train


def train_run(run: wandb.Run) -> None:
    """Train and evaluate one configuration assigned by the sweep agent."""
    config = run.config
    dataset_path = Path(config["dataset_path"])
    if not dataset_path.is_file():
        raise FileNotFoundError(f"{dataset_path} is missing. Set dataset_path in the sweep YAML.")
    dataset = PendulumDataset(dataset_path)
    if dataset.num_trajectories < 2:
        raise ValueError("A sweep requires a compact dataset with at least two trajectories.")

    time_step = float(dataset.metadata.get("time_step", 0))
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("The dataset metadata must contain a finite positive time_step.")
    num_steps = dataset.steps_per_trajectory
    times = float(dataset.metadata.get("start_time", 0)) + time_step * np.arange(num_steps + 1)
    num_epochs = config["num_epochs"]
    if type(num_epochs) is not int or num_epochs < 1:
        raise ValueError("num_epochs must be a positive integer.")
    relative_error_floor = float(config["relative_error_floor"])
    if not np.isfinite(relative_error_floor) or relative_error_floor <= 0:
        raise ValueError("relative_error_floor must be finite and positive.")

    train_ids, val_ids = train_test_split(
        np.arange(dataset.num_trajectories), test_size=0.2, random_state=config["split_seed"]
    )
    offsets = np.arange(num_steps)
    train_indices, val_indices = (
        (ids[:, None] * num_steps + offsets).ravel() for ids in (train_ids, val_ids)
    )
    validation_trajectories = sorted(val_ids.tolist())
    with dataset_path.open("rb") as source:
        dataset_sha256 = file_digest(source, "sha256").hexdigest()
    config.update({
        "dataset_sha256": dataset_sha256,
        "time_step": time_step,
        "validation_trajectories": validation_trajectories,
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
        split=(train_indices, val_indices),
        seed=config["seed"],
        on_epoch=log_epoch,
    )

    plot_path = run_dir / "evaluation.png"
    plot_trajectory = validation_trajectories[0]
    rollouts = []
    for trajectory in validation_trajectories:
        l, m, b, theta0, omega0 = dataset.initial_conditions(trajectory)
        metrics = evaluate(
            model_path,
            hidden_size=config["hidden_size"],
            num_hidden_layers=config["num_hidden_layers"],
            num_steps=num_steps,
            l=l,
            m=m,
            b=b,
            theta0=theta0,
            omega0=omega0,
            t=times,
            show=False,
            relative_error_floor=relative_error_floor,
            plot_path=plot_path if trajectory == plot_trajectory else None,
        )
        rollouts.append({"trajectory": trajectory, "metrics": metrics})

    summary = {
        f"rollout_{key}": float(np.mean([rollout["metrics"][key] for rollout in rollouts]))
        for key in ("relative_error_pct", "theta_relative_error_pct", "omega_relative_error_pct")
    }
    report = {
        "run_id": run.id,
        "config": dict(config),
        "history": history,
        "summary": summary,
        "rollouts": rollouts,
        "checkpoint": model_path.name,
        "plot": plot_path.name,
        "plot_trajectory": plot_trajectory,
    }
    with TemporaryDirectory(dir=run_dir) as temp:
        report_path = Path(temp) / "metrics.json"
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        report_path.replace(run_dir / "metrics.json")

    run.summary.update(summary)
    relative_errors = wandb.Html(
        "<h3>Relative errors</h3>"
        f"<p>Overall <strong>{summary['rollout_relative_error_pct']:.2f}%</strong></p>"
        f"<p>Angle (θ) <strong>{summary['rollout_theta_relative_error_pct']:.2f}%</strong></p>"
        f"<p>Angular velocity (ω) <strong>{summary['rollout_omega_relative_error_pct']:.2f}%</strong></p>",
        inject=False,
    )
    run.log({"relative_errors": relative_errors, "evaluation_plot": wandb.Image(str(plot_path))})
    print(f"Rollout relative error: {summary['rollout_relative_error_pct']:.2f}%")
    print(f"Run saved to {run_dir}")


def main() -> None:
    with wandb.init() as run:
        train_run(run)


if __name__ == "__main__":
    main()
