```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```powershell
python -m pendulum_surrogate.generate_dataset --yaml configs/dataset_1_sample.yaml
python -m pendulum_surrogate.generate_dataset --yaml configs/dataset.yaml
```

```powershell
wandb sweep --project pendulum-surrogate configs/sweep_single.yaml
wandb sweep --project pendulum-surrogate configs/sweep_single_v2.yaml
wandb agent --count 1 ENTITY/pendulum-surrogate/SWEEP_ID
```

```powershell
wandb sweep --project pendulum-surrogate configs/sweep.yaml
wandb agent --count 12 ENTITY/pendulum-surrogate/SWEEP_ID
```

```powershell
python -m pendulum_surrogate.runner --mode simulate --duration 10 --framerate 50
python -m pendulum_surrogate.runner --mode infer --duration 10 --framerate 50 --model_path PATH_TO_CHECKPOINT
python -m pendulum_surrogate.runner --mode evaluate --duration 10 --framerate 50 --model_path PATH_TO_CHECKPOINT
```
