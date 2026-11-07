# surrogate-models

## Setup

```bash
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

```bash
python pendulum/runner.py --mode simulate
python pendulum/runner.py --mode infer
python pendulum/runner.py --mode train
```