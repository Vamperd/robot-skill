import argparse
import json
import pickle
import shutil
import time
from pathlib import Path

from scenario_generator import DEFAULT_SPLIT_COUNTS, FAMILY_NAMES, generate_scenario, summarize_scenario


SPLIT_SEED_OFFSETS = {
    "train": 10_000,
    "val": 20_000,
    "test": 30_000,
    "stress": 40_000,
}


def _clear_generated_files(save_dir: Path) -> None:
    if not save_dir.exists():
        return
    for file_path in save_dir.rglob("*.pkl"):
        file_path.unlink()
    manifest = save_dir / "dataset_manifest.json"
    if manifest.exists():
        manifest.unlink()
    for split_dir in sorted(save_dir.glob("*"), reverse=True):
        if split_dir.is_dir() and not any(split_dir.iterdir()):
            shutil.rmtree(split_dir)


def build_maps(
    save_dir: str | Path = "offline_maps_v2",
    split_counts: dict | None = None,
    overwrite: bool = False,
    limit_per_family: int | None = None,
) -> dict:
    save_path = Path(save_dir)
    split_counts = split_counts or DEFAULT_SPLIT_COUNTS

    if overwrite:
        _clear_generated_files(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    manifest = {
        "schema_version": "v2",
        "root": str(save_path.resolve()),
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "splits": {},
    }

    total_start = time.time()
    total_count = 0

    for split, family_counts in split_counts.items():
        manifest["splits"][split] = {}
        split_dir = save_path / split
        split_dir.mkdir(parents=True, exist_ok=True)

        for family in FAMILY_NAMES:
            count = family_counts.get(family, 0)
            if limit_per_family is not None:
                count = min(count, limit_per_family)

            family_dir = split_dir / family
            family_dir.mkdir(parents=True, exist_ok=True)
            family_start = time.time()
            summaries = []

            for index in range(count):
                seed = SPLIT_SEED_OFFSETS[split] + FAMILY_NAMES.index(family) * 10_000 + index
                scenario_id = f"{split}_{family}_{index:04d}"
                scenario = generate_scenario(seed=seed, family=family, split=split, scenario_id=scenario_id)

                file_path = family_dir / f"{scenario_id}.pkl"
                with file_path.open("wb") as handle:
                    pickle.dump(scenario, handle)

                summaries.append(summarize_scenario(scenario))
                total_count += 1

            manifest["splits"][split][family] = {
                "count": count,
                "elapsed_sec": round(time.time() - family_start, 2),
                "samples": summaries[:3],
            }
            print(f"[{split}/{family}] 完成 {count} 张图。")

    manifest["total_count"] = total_count
    manifest["elapsed_sec"] = round(time.time() - total_start, 2)

    with (save_path / "dataset_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)

    print(f"\n新版数据集生成完成，共 {total_count} 个场景。")
    print(f"输出目录: {save_path.resolve()}")
    print(f"总耗时: {manifest['elapsed_sec']:.2f} 秒")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建协同调度 v2 离线地图数据集。")
    parser.add_argument("--save-dir", default="offline_maps_v2", help="输出目录，默认 offline_maps_v2")
    parser.add_argument("--overwrite", action="store_true", help="覆盖当前输出目录中的旧生成结果")
    parser.add_argument("--limit-per-family", type=int, default=None, help="调试模式下限制每个 family 的生成数量")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_maps(
        save_dir=args.save_dir,
        overwrite=args.overwrite,
        limit_per_family=args.limit_per_family,
    )
