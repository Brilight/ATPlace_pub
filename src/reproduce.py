import argparse
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

import torch

if not hasattr(torch, "concatenate"):
    torch.concatenate = torch.cat

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from Parser import read_data
from Thermal.ThermalSolver import HotSpot_solver
from ATPLACE.PlaceFlow import placeflow
from utils.Visualize import visualize_placement_results


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def _default_thermal_dir() -> str:
    return os.path.join(_repo_root(), "thermal") + os.sep


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _normalize_floorplan_stages(params, system) -> None:
    for stage in getattr(params, "floorplan_stages", []) or []:
        lr = stage.get("learning_rate", None)
        if isinstance(lr, (int, float)):
            stage["learning_rate"] = [float(lr) * float(system.grid_size), float(lr) * float(np.pi)]
        elif isinstance(lr, list) and len(lr) == 1:
            stage["learning_rate"] = [float(lr[0]) * float(system.grid_size), float(lr[0]) * float(np.pi)]


def _evaluate_hotspot(
    params,
    system,
    best_pos: Tuple[np.ndarray, np.ndarray, np.ndarray],
    chiplet_size: Tuple[np.ndarray, np.ndarray],
    out_dir: str,
) -> Dict[str, Any]:
    solver = HotSpot_solver(system)
    solver.amb = float(getattr(params, "ambient", 45))

    x_final, y_final, _ = best_pos
    size_x, size_y = chiplet_size
    solver.set_pos(system.powermap, (x_final, y_final), size_x, size_y)

    case_name = getattr(params, "prefix", "case")
    safe_name = os.path.basename(str(case_name))
    run_tag = time.strftime("%Y%m%d-%H%M%S")
    temp_file_name = f"{safe_name}_{run_tag}"
    solver.run(temp_file_name, default=1)
    temp_field = solver.getres(temp_file_name) - 273.15

    np.save(os.path.join(out_dir, "temp_field.npy"), temp_field.astype(np.float32))
    return {"temp_max": float(temp_field.max()), "temp_field": temp_field}


def run_case(
    case_dir: str,
    out_dir: Optional[str],
    seed: int = 1000,
    thermal: bool = True,
    plots: bool = True,
    thermal_dir: Optional[str] = None,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    case_dir = os.path.abspath(case_dir)
    case_name = os.path.basename(os.path.normpath(case_dir))
    if out_dir is None:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(_repo_root(), "results", case_name, ts)
    out_dir = _ensure_dir(out_dir)

    params, system, _ = read_data(param_file=None, spec_param_file=None, case_dir=case_dir)
    params.random_seed = int(seed)
    np.random.seed(int(seed))

    system.thermal_dir = (thermal_dir or _default_thermal_dir())
    params.thermal_dir = system.thermal_dir

    params.ILPsolver = "pulp"
    params.thermal_solver = "hotspot"
    params.temp_aware_opt = False
    params.warp_aware_opt = False
    params.plot_freq = -1

    if not hasattr(params, "overflow_init"):
        params.overflow_init = 0.3
    if not hasattr(params, "density_weight_init"):
        params.density_weight_init = float(getattr(params, "density_weight", 3.0))
    if not hasattr(params, "wl_weight"):
        params.wl_weight = 1e-2 / max(1, int(getattr(system, "num_nets", 1)))
    if not hasattr(params, "temp_weight_init"):
        params.temp_weight_init = 1e2
    if not hasattr(params, "warp_weight_init"):
        params.warp_weight_init = 1e2

    _normalize_floorplan_stages(params, system)

    hpwl, best_metric, best_fp = placeflow(params, system, tempmodel=None, temp_solver=None, temp_sim=False)

    num_chiplets = system.num_chiplets
    num_nodes = system.num_nodes
    pos_torch = best_fp[0]
    theta_torch = best_fp[1]
    x_final = pos_torch[:num_chiplets].detach().cpu().numpy()
    y_final = pos_torch[num_nodes : num_nodes + num_chiplets].detach().cpu().numpy()
    theta_final = theta_torch[:num_chiplets].detach().cpu().numpy()

    w0 = np.array([c.width for c in system.chiplets], dtype=np.float64)
    h0 = np.array([c.height for c in system.chiplets], dtype=np.float64)
    theta_q = (np.round(theta_final * 4).astype(np.int64) % 4)
    swap = (theta_q % 2) == 1
    chiplet_size_x = np.where(swap, h0, w0).astype(np.float64)
    chiplet_size_y = np.where(swap, w0, h0).astype(np.float64)

    result: Dict[str, Any] = {
        "case_dir": case_dir,
        "out_dir": out_dir,
        "hpwl": float(hpwl),
        "best_pos": (x_final, y_final, theta_final),
        "intp_width": float(getattr(system, "intp_width", 0.0)),
        "intp_height": float(getattr(system, "intp_height", 0.0)),
        "runtime_s": float(time.perf_counter() - t0),
    }

    if thermal:
        try:
            result.update(
                _evaluate_hotspot(params, system, result["best_pos"], (chiplet_size_x, chiplet_size_y), out_dir)
            )
        except Exception as e:
            result["thermal_error"] = str(e)

    np.savez(
        os.path.join(out_dir, "placement.npz"),
        x=x_final,
        y=y_final,
        theta=theta_final,
        chiplet_size_x=chiplet_size_x,
        chiplet_size_y=chiplet_size_y,
    )

    metrics = {k: v for k, v in result.items() if isinstance(v, (int, float, str))}
    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    try:
        params_json = params.toJson() if hasattr(params, "toJson") else dict(params.__dict__)
        with open(os.path.join(out_dir, "params_resolved.json"), "w", encoding="utf-8") as f:
            json.dump(params_json, f, indent=2, ensure_ascii=False)
    except Exception:
        pass

    if plots:
        visualize_placement_results(system, result, save_dir=out_dir, trial_id="best")

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True, help="Case directory, e.g. cases/Case1")
    parser.add_argument("--out", default=None, help="Output directory (default: results/<CaseName>/<timestamp>)")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--no-thermal", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--thermal-dir", default=None, help="Thermal directory (default: <repo>/thermal/)")
    args = parser.parse_args()

    case_dir = os.path.abspath(args.case)
    case_name = os.path.basename(os.path.normpath(case_dir))
    if args.out:
        out_dir = os.path.abspath(args.out)
    else:
        ts = time.strftime("%Y%m%d-%H%M%S")
        out_dir = os.path.join(_repo_root(), "results", case_name, ts)

    run_case(
        case_dir=case_dir,
        out_dir=out_dir,
        seed=args.seed,
        thermal=not args.no_thermal,
        plots=not args.no_plots,
        thermal_dir=args.thermal_dir,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
