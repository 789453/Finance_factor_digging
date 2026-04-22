import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))

from src.alphagen_generic.feature_registry_manager import FeatureRegistryManager
from src.alphagen_generic.parquet_feature_loader import ParquetFeatureLoader
from src.alphagen_generic.run_artifacts import build_run_dir, save_json
from src.alphagen_generic.task_config import apply_task_config


def test_high_throughput_io():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        factor_ready_dir = tmp / "factor_ready"
        basic_dir = tmp / "basic"
        output_dir = tmp / "outputs"
        factor_ready_dir.mkdir(parents=True, exist_ok=True)
        basic_dir.mkdir(parents=True, exist_ok=True)

        registry_path = factor_ready_dir / "feature_registry.csv"
        pd.DataFrame(
            [
                {"domain": "A", "feature_name": "alpha_signal", "allow_in_alphaprobe": 1},
                {"domain": "A", "feature_name": "inactive_signal", "allow_in_alphaprobe": 0},
            ]
        ).to_csv(registry_path, index=False)

        feature_df = pd.DataFrame(
            [
                {"trade_date": "20200101", "ts_code": "000001.SZ", "alpha_signal": 1.0, "extra_col": 99.0},
                {"trade_date": "20200102", "ts_code": "000001.SZ", "alpha_signal": 2.0, "extra_col": 99.0},
                {"trade_date": "20200103", "ts_code": "000001.SZ", "alpha_signal": 3.0, "extra_col": 99.0},
                {"trade_date": "20200104", "ts_code": "000001.SZ", "alpha_signal": 4.0, "extra_col": 99.0},
                {"trade_date": "20200105", "ts_code": "000001.SZ", "alpha_signal": 5.0, "extra_col": 99.0},
            ]
        )
        feature_df.to_parquet(factor_ready_dir / "feature_A_price_volume.parquet", index=False)

        daily_df = pd.DataFrame(
            [
                {"trade_date": "20200101", "ts_code": "000001.SZ", "close": 10.0},
                {"trade_date": "20200102", "ts_code": "000001.SZ", "close": 10.5},
                {"trade_date": "20200103", "ts_code": "000001.SZ", "close": 11.0},
                {"trade_date": "20200104", "ts_code": "000001.SZ", "close": 11.5},
                {"trade_date": "20200105", "ts_code": "000001.SZ", "close": 12.0},
            ]
        )
        daily_df.to_parquet(basic_dir / "daily.parquet", index=False)

        task_config_path = tmp / "task_config.json"
        with open(task_config_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "data": {
                        "domain": "A",
                        "registry_path": str(registry_path),
                        "daily_path": str(basic_dir / "daily.parquet"),
                        "data_dir": str(factor_ready_dir),
                        "use_filled": False,
                        "max_backtrack_days": 1,
                        "max_future_days": 1,
                    },
                    "training": {
                        "n_episodes": 7,
                    },
                    "search": {
                        "delta_times": [5, 10],
                        "constants": [0.5, 1.0],
                    },
                },
                f,
                ensure_ascii=False,
                indent=2,
            )

        parser = argparse.ArgumentParser()
        parser.add_argument("--task_config", type=str, default=None)
        parser.add_argument("--domain", type=str, default="B")
        parser.add_argument("--registry_path", type=str, default="")
        parser.add_argument("--daily_path", type=str, default="")
        parser.add_argument("--data_dir", type=str, default="")
        parser.add_argument("--use_filled", type=bool, default=True)
        parser.add_argument("--max_backtrack_days", type=int, default=100)
        parser.add_argument("--max_future_days", type=int, default=30)
        parser.add_argument("--n_episodes", type=int, default=100)
        parser.add_argument("--delta_times", type=str, default=None)
        parser.add_argument("--constants", type=str, default=None)
        args = parser.parse_args([])
        args.task_config = str(task_config_path)

        raw_config = apply_task_config(args, parser)
        assert raw_config["training"]["n_episodes"] == 7
        assert args.domain == "A"
        assert args.n_episodes == 7
        assert args.use_filled is False

        registry_manager = FeatureRegistryManager(args.registry_path)
        loader = ParquetFeatureLoader(
            domain=args.domain,
            start_time="20200102",
            end_time="20200104",
            registry_manager=registry_manager,
            data_dir=args.data_dir,
            daily_path=args.daily_path,
            device=torch.device("cpu"),
            max_backtrack_days=args.max_backtrack_days,
            max_future_days=args.max_future_days,
            use_filled=args.use_filled,
        )

        assert loader.data.shape == (5, 2, 1)
        assert loader.feature_map["$alpha_signal"].name == "ALPHA_SIGNAL"

        run_dir = build_run_dir(str(output_dir), prefix="unit", run_name="io")
        save_json(os.path.join(run_dir, "manifest.json"), {"domain": args.domain, "n_episodes": args.n_episodes})
        assert os.path.exists(os.path.join(run_dir, "manifest.json"))

        print("High throughput IO test passed.")


if __name__ == "__main__":
    test_high_throughput_io()
