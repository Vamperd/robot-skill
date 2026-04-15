import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_JSON = ROOT / "成果" / "json" / "ranker_test_family_eval.json"
FALLBACK_JSON = ROOT / "成果" / "json" / "ranker_test_eval.json"
OUT_DIR = ROOT / "成果" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "SimSun", "STSong"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "axes.unicode_minus": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


LABEL_MAP = {
    "scheduler": "Hetero-Ranker",
    "random": "Random",
    "nearest_eta_greedy": "Nearest-ETA",
    "role_aware_greedy": "Role-Aware",
    "wait_aware_role_greedy": "Wait-Aware",
    "upfront_wait_aware_greedy": "Upfront Teacher",
    "rollout_upfront_teacher": "Rollout Teacher",
    "hybrid_upfront_teacher": "Hybrid Teacher",
}

METHOD_ORDER = [
    "scheduler",
    "upfront_wait_aware_greedy",
    "hybrid_upfront_teacher",
    "rollout_upfront_teacher",
    "wait_aware_role_greedy",
    "role_aware_greedy",
    "nearest_eta_greedy",
    "random",
]

FAMILY_ORDER = [
    "open_balance",
    "role_mismatch",
    "single_bottleneck",
    "double_bottleneck",
    "far_near_trap",
    "multi_sync_cluster",
    "partial_coalition_trap",
]

FAMILY_LABEL_MAP = {
    "open_balance": "Open\nBalance",
    "role_mismatch": "Role\nMismatch",
    "single_bottleneck": "Single\nBottleneck",
    "double_bottleneck": "Double\nBottleneck",
    "far_near_trap": "Far-Near\nTrap",
    "multi_sync_cluster": "Multi-Sync\nCluster",
    "partial_coalition_trap": "Partial\nCoalition Trap",
}

MAIN_METRICS = [
    ("success_rate", "Success Rate", "higher"),
    ("mean_makespan", "Makespan", "lower"),
    ("mean_avoidable_wait_time", "Avoidable Wait", "lower"),
    ("mean_direct_sync_misassignment_rate", "Sync Misassign", "lower"),
]

TRAP_METRICS = [
    ("success_rate", "Trap Success Rate", "higher"),
    ("mean_makespan", "Trap Makespan", "lower"),
    ("mean_avoidable_wait_time", "Trap Avoidable Wait", "lower"),
    ("mean_direct_sync_misassignment_rate", "Trap Sync Misassign", "lower"),
]

HEATMAP_METRICS = [
    ("success_rate", "Success", "higher"),
    ("mean_makespan", "Makespan", "lower"),
    ("mean_avoidable_wait_time", "Avoid. Wait", "lower"),
    ("mean_direct_sync_misassignment_rate", "Sync Misassign", "lower"),
]

COLOR_MAP = {
    "scheduler": "#C44E52",
    "upfront_wait_aware_greedy": "#4C72B0",
    "hybrid_upfront_teacher": "#55A868",
    "rollout_upfront_teacher": "#8172B2",
    "wait_aware_role_greedy": "#CCB974",
    "role_aware_greedy": "#64B5CD",
    "nearest_eta_greedy": "#8C8C8C",
    "random": "#B0B0B0",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate paper-ready figures from scheduler evaluation JSON.")
    parser.add_argument("--json", default=None, help="Main evaluation JSON path. Defaults to ranker_test_family_eval.json.")
    parser.add_argument(
        "--error-json",
        default=None,
        help=(
            "Optional JSON containing std/error values with the same structure as the main JSON. "
            "If provided, bar charts will render error bars."
        ),
    )
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory for figures.")
    return parser.parse_args()


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def choose_default_json():
    if DEFAULT_JSON.exists():
        return DEFAULT_JSON
    return FALLBACK_JSON


def build_method_groups(data):
    main_methods = {"scheduler": data["scheduler"], **data["baselines"]}
    trap_methods = {"scheduler": data["trap_subset"], **data["trap_baselines"]}
    family_breakdown = data.get("family_breakdown", {})
    return main_methods, trap_methods, family_breakdown


def _ordered_methods(method_dict):
    return [m for m in METHOD_ORDER if m in method_dict]


def _format_value(metric_key, value):
    if metric_key == "success_rate":
        return f"{value:.3f}"
    if "rate" in metric_key:
        return f"{value:.4f}"
    return f"{value:.2f}"


def _available_methods_for_metric(method_dict, metric_key):
    methods = []
    missing = []
    for method in _ordered_methods(method_dict):
        if metric_key in method_dict[method]:
            methods.append(method)
        else:
            missing.append(method)
    return methods, missing


def _std_lookup(error_group, method, metric_key):
    if not error_group:
        return None
    method_metrics = error_group.get(method, {})
    return method_metrics.get(metric_key)


def _save_figure(fig, save_path):
    fig.savefig(save_path, bbox_inches="tight")
    fig.savefig(save_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_group(method_dict, metrics, error_group, save_path, title):
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 7.6))
    axes = axes.flatten()

    for ax, (metric_key, metric_title, direction) in zip(axes, metrics):
        methods, missing = _available_methods_for_metric(method_dict, metric_key)
        labels = [LABEL_MAP.get(m, m) for m in methods]
        values = [method_dict[m][metric_key] for m in methods]
        errors = [_std_lookup(error_group, m, metric_key) for m in methods]
        use_errors = any(err is not None for err in errors)
        yerr = np.array([0.0 if err is None else err for err in errors]) if use_errors else None

        colors = [COLOR_MAP.get(method, "#7F7F7F") for method in methods]
        bars = ax.bar(
            labels,
            values,
            color=colors,
            yerr=yerr,
            capsize=4 if use_errors else 0,
            edgecolor="#333333",
            linewidth=0.6,
        )
        ax.set_title(f"{metric_title} ({'↑' if direction == 'higher' else '↓'})")
        ax.tick_params(axis="x", rotation=22)
        ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)

        if values:
            ymax = max(values)
            offset = max(ymax * 0.02, 0.01)
            for bar, value in zip(bars, values):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    _format_value(metric_key, value),
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

        if missing:
            missing_labels = ", ".join(LABEL_MAP.get(m, m) for m in missing)
            ax.text(
                0.98,
                0.98,
                f"N/A: {missing_labels}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#555555",
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
            )

    fig.suptitle(title, fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_ranker_vs_teacher(method_dict, trap_method_dict, error_data, save_path):
    teacher_key = "upfront_wait_aware_greedy"
    ranker = method_dict["scheduler"]
    teacher = method_dict[teacher_key]
    trap_ranker = trap_method_dict["scheduler"]
    trap_teacher = trap_method_dict[teacher_key]

    labels = ["Makespan", "Avoidable Wait", "Sync Misassign", "Trap Avoidable Wait"]
    ranker_values = [
        ranker["mean_makespan"],
        ranker["mean_avoidable_wait_time"],
        ranker["mean_direct_sync_misassignment_rate"],
        trap_ranker["mean_avoidable_wait_time"],
    ]
    teacher_values = [
        teacher["mean_makespan"],
        teacher["mean_avoidable_wait_time"],
        teacher["mean_direct_sync_misassignment_rate"],
        trap_teacher["mean_avoidable_wait_time"],
    ]

    ranker_errors = None
    teacher_errors = None
    if error_data:
        main_err = error_data.get("main_methods", {})
        trap_err = error_data.get("trap_methods", {})
        ranker_errors = [
            main_err.get("scheduler", {}).get("mean_makespan", 0.0),
            main_err.get("scheduler", {}).get("mean_avoidable_wait_time", 0.0),
            main_err.get("scheduler", {}).get("mean_direct_sync_misassignment_rate", 0.0),
            trap_err.get("scheduler", {}).get("mean_avoidable_wait_time", 0.0),
        ]
        teacher_errors = [
            main_err.get(teacher_key, {}).get("mean_makespan", 0.0),
            main_err.get(teacher_key, {}).get("mean_avoidable_wait_time", 0.0),
            main_err.get(teacher_key, {}).get("mean_direct_sync_misassignment_rate", 0.0),
            trap_err.get(teacher_key, {}).get("mean_avoidable_wait_time", 0.0),
        ]

    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(10.2, 5.2))
    ax.bar(
        x - width / 2,
        ranker_values,
        width,
        label="Hetero-Ranker",
        color=COLOR_MAP["scheduler"],
        yerr=ranker_errors,
        capsize=4 if ranker_errors else 0,
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.bar(
        x + width / 2,
        teacher_values,
        width,
        label="Upfront Teacher",
        color=COLOR_MAP[teacher_key],
        yerr=teacher_errors,
        capsize=4 if teacher_errors else 0,
        edgecolor="#333333",
        linewidth=0.6,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.set_title("Hetero-Ranker vs Upfront Teacher", fontweight="bold")
    ax.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_figure(fig, save_path)


def plot_family_heatmap(family_breakdown, save_path):
    available_families = [family for family in FAMILY_ORDER if family in family_breakdown]
    display_labels = [FAMILY_LABEL_MAP.get(family, family) for family in available_families]

    raw_matrix = []
    annotations = []
    for family in available_families:
        metrics = family_breakdown[family]
        raw_row = []
        ann_row = []
        for metric_key, _, direction in HEATMAP_METRICS:
            value = metrics.get(metric_key, np.nan)
            raw_row.append(value)
            ann_row.append(_format_value(metric_key, value) if not np.isnan(value) else "N/A")
        raw_matrix.append(raw_row)
        annotations.append(ann_row)

    raw_matrix = np.array(raw_matrix, dtype=float)
    score_matrix = np.zeros_like(raw_matrix)

    for col, (_, _, direction) in enumerate(HEATMAP_METRICS):
        column = raw_matrix[:, col]
        cmin = np.nanmin(column)
        cmax = np.nanmax(column)
        if np.isclose(cmax, cmin):
            score_matrix[:, col] = 0.5
            continue
        normalized = (column - cmin) / (cmax - cmin)
        if direction == "higher":
            score_matrix[:, col] = normalized
        else:
            score_matrix[:, col] = 1.0 - normalized

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    im = ax.imshow(score_matrix, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(HEATMAP_METRICS)))
    ax.set_xticklabels([title for _, title, _ in HEATMAP_METRICS], rotation=15)
    ax.set_yticks(np.arange(len(display_labels)))
    ax.set_yticklabels(display_labels)
    ax.set_title("Family Breakdown Heatmap (Hetero-Ranker)", fontweight="bold")

    for i in range(score_matrix.shape[0]):
        for j in range(score_matrix.shape[1]):
            ax.text(
                j,
                i,
                annotations[i][j],
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized performance (higher is better)")
    fig.tight_layout()
    _save_figure(fig, save_path)


def build_error_groups(error_data):
    if not error_data:
        return None
    return {
        "main_methods": {"scheduler": error_data.get("scheduler", {}), **error_data.get("baselines", {})},
        "trap_methods": {"scheduler": error_data.get("trap_subset", {}), **error_data.get("trap_baselines", {})},
        "family_breakdown": error_data.get("family_breakdown", {}),
    }


def main():
    args = parse_args()
    json_path = Path(args.json) if args.json else choose_default_json()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(json_path)
    error_data = load_json(Path(args.error_json)) if args.error_json else None
    error_groups = build_error_groups(error_data)

    main_methods, trap_methods, family_breakdown = build_method_groups(data)

    plot_group(
        main_methods,
        MAIN_METRICS,
        None if error_groups is None else error_groups["main_methods"],
        out_dir / "ranker_main_results.png",
        "Overall Test Results",
    )
    plot_group(
        trap_methods,
        TRAP_METRICS,
        None if error_groups is None else error_groups["trap_methods"],
        out_dir / "ranker_trap_results.png",
        "Trap Subset Results",
    )
    plot_ranker_vs_teacher(main_methods, trap_methods, error_groups, out_dir / "ranker_vs_upfront_teacher.png")
    if family_breakdown:
        plot_family_heatmap(family_breakdown, out_dir / "ranker_family_heatmap.png")

    print("Saved figures to:", out_dir)


if __name__ == "__main__":
    main()
