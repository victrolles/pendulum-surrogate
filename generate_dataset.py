import argparse
from math import prod
from pathlib import Path

import yaml

from generation import Method, generate_dataset


class _UniqueKeyLoader(yaml.SafeLoader):
    def construct_mapping(self, node, deep=False):
        keys = set()
        for key_node, _ in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ValueError("Configuration keys must be strings.")
            if key in keys:
                raise ValueError(f"Duplicate configuration key: {key}")
            keys.add(key)
        return super().construct_mapping(node, deep=deep)


def _generate_from_yaml(path: Path) -> Path:
    config = yaml.load(path.read_text(encoding="utf-8-sig"), Loader=_UniqueKeyLoader)
    if not isinstance(config, dict):
        raise ValueError("The YAML document must contain a configuration mapping.")
    required = {"method", "ranges", "duration", "framerate"}
    missing = required - config.keys()
    if missing:
        raise ValueError(f"Missing configuration fields: {', '.join(sorted(missing))}")
    unknown = config.keys() - required - {"dataset_dir", "num_samples", "seed"}
    if unknown:
        raise ValueError(f"Unknown configuration fields: {', '.join(sorted(unknown))}")

    method = Method(config["method"])
    ranges = config["ranges"]
    if not isinstance(ranges, dict) or set(ranges) != {"l", "m", "b", "theta0", "omega0"}:
        raise ValueError("ranges must contain exactly l, m, b, theta0 and omega0.")
    range_arguments = {}
    for name, values in ranges.items():
        if type(values) in (int, float):
            values = [values]
        if not isinstance(values, list) or not values or any(type(value) not in (int, float) for value in values):
            raise ValueError(f"ranges.{name} must be a number or a nonempty list of numbers.")
        range_arguments[f"range_{name}"] = values

    for name in ("duration", "framerate"):
        if type(config[name]) not in (int, float):
            raise ValueError(f"{name} must be a number.")
    num_samples = config.get("num_samples")
    if "num_samples" in config and (type(num_samples) is not int or num_samples < 1):
        raise ValueError("num_samples must be a positive integer.")
    seed = config.get("seed")
    if method is Method.random_sampling:
        if num_samples is None:
            raise ValueError("random_sampling requires num_samples.")
        if type(seed) is not int or seed < 0:
            raise ValueError("random_sampling requires a nonnegative integer seed.")
    else:
        grid_size = prod(len(values) for values in range_arguments.values())
        if num_samples is not None and num_samples != grid_size:
            raise ValueError(f"num_samples must match the fixed grid size ({grid_size}).")
        num_samples = grid_size

    dataset_dir = config.get("dataset_dir", "datasets")
    if not isinstance(dataset_dir, str) or not dataset_dir.strip():
        raise ValueError("dataset_dir must be a nonempty path.")
    return generate_dataset(
        dataset_dir=(path.parent / dataset_dir).resolve(),
        method=method,
        num_samples=num_samples,
        duration=config["duration"],
        framerate=config["framerate"],
        seed=seed,
        **range_arguments,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pendulum dataset from a YAML configuration.")
    parser.add_argument("--yaml", required=True, type=Path, help="Path to the dataset configuration.")
    args = parser.parse_args()
    try:
        _generate_from_yaml(args.yaml)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(f"{args.yaml}: {error}")


if __name__ == "__main__":
    main()
