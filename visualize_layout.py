#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def draw_layout(layout: dict, out_path: Path) -> None:
    chiplets = layout.get("chiplets") or []
    if not chiplets:
        raise ValueError("layout.json does not contain parsed chiplet coordinates")

    interposer = layout["interposer"]
    width = float(interposer["width"])
    height = float(interposer["height"])
    unit = layout.get("unit", "um")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.add_patch(plt.Rectangle((0, 0), width, height, fill=False, edgecolor="black", linewidth=1.5))

    for chiplet in chiplets:
        x = float(chiplet["x"])
        y = float(chiplet["y"])
        w = float(chiplet["width"])
        h = float(chiplet["height"])
        ax.add_patch(
            plt.Rectangle(
                (x - 0.5 * w, y - 0.5 * h),
                w,
                h,
                facecolor="#a7c7e7",
                edgecolor="#1f2933",
                linewidth=1.0,
                alpha=0.85,
            )
        )
        label = chiplet.get("name", "")
        power = chiplet.get("power_w")
        if power is not None:
            label = f"{label}\n{float(power):.1f} W"
        ax.text(x, y, label, ha="center", va="center", fontsize=8)

    ax.set_xlabel(f"x ({unit})")
    ax.set_ylabel(f"y ({unit})")
    ax.set_title(f"{layout.get('case', '')} {layout.get('mode', '')}".strip())
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("layout_json", help="Path to layout.json written by reproduce.py")
    parser.add_argument("--out", help="Output PNG path. Defaults to layout.png beside layout.json")
    args = parser.parse_args()

    layout_path = Path(args.layout_json).resolve()
    out_path = Path(args.out).resolve() if args.out else layout_path.with_name("layout.png")
    draw_layout(load_json(layout_path), out_path)
    print(out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
