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
python -m pendulum_surrogate.runner --mode infer --duration 20 --framerate 30 --model_path saved_runs\ko9ez1lx\model_600.pth
python -m pendulum_surrogate.runner --mode evaluate --duration 10 --framerate 50 --model_path PATH_TO_CHECKPOINT
python -m pendulum_surrogate.runner --mode evaluate --duration 20 --framerate 30 --model_path saved_runs\ko9ez1lx\model_600.pth
```
First training : 1 combination dataset

```powershell
python -m pendulum_surrogate.generate_dataset --yaml configs/dataset_1_sample.yaml

wandb sweep --project pendulum-surrogate configs/sweep_single_v2.yaml

python -m pendulum_surrogate.runner --mode infer --duration 20 --framerate 30 --model_path saved_runs\ko9ez1lx\model_600.pth

python -m pendulum_surrogate.runner `
    --mode evaluate `
    --duration 20 `
    --framerate 30 `
    --model_path saved_runs\ko9ez1lx\model_600.pth `
    --l 1.0 `
    --m 1.0 `
    --b 0.1 `
    --theta0 1.5707963267948966 `
    --omega0 0.0
```

Second training : 162 combinations fixed dataset

```powershell
python -m pendulum_surrogate.generate_dataset --yaml configs\dataset\dataset_fixed_small.yaml

wandb sweep --project pendulum-surrogate configs/sweep/sweep_fixed_small.yaml

python -m pendulum_surrogate.runner --mode infer --duration 20 --framerate 30 --model_path saved_runs\ko9ez1lx\model_600.pth

python -m pendulum_surrogate.runner `
    --mode evaluate `
    --duration 20 `
    --framerate 30 `
    --model_path saved_runs\ko9ez1lx\model_600.pth `
    --l 1.0 `
    --m 1.0 `
    --b 0.1 `
    --theta0 1.5707963267948966 `
    --omega0 0.0
```

Third training : 162 combinations fixed dataset

```powershell
python -m pendulum_surrogate.generate_dataset --yaml configs\dataset\dataset_fixed_medium.yaml

wandb sweep --project pendulum-surrogate configs/sweep/sweep_fixed_medium.yaml

python -m pendulum_surrogate.runner --mode infer --duration 20 --framerate 30 --model_path saved_runs\ko9ez1lx\model_600.pth

python -m pendulum_surrogate.runner `
    --mode evaluate `
    --duration 20 `
    --framerate 30 `
    --model_path saved_runs\ko9ez1lx\model_600.pth `
    --l 1.0 `
    --m 1.0 `
    --b 0.1 `
    --theta0 1.5707963267948966 `
    --omega0 0.0
```