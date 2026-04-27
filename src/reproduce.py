import argparse
import json
import os
import time
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import torch

if not hasattr(torch, "concatenate"):
    torch.concatenate = torch.cat

_mpl_dir = tempfile.mkdtemp(prefix="matplotlib-")
os.environ.setdefault("MPLCONFIGDIR", _mpl_dir)
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

from Parser import read_data
from Thermal.ThermalSolver import HotSpot_solver
from Thermal.TModelTrain import Datagen, train_model
from ATPLACE.PlaceFlow import placeflow
from utils.Visualize import visualize_placement_results


def _repo_root() -> str:
    return str(Path(__file__).resolve().parents[1])


def _default_thermal_dir() -> str:
    return os.path.join(_repo_root(), "thermal") + os.sep


def _ensure_dir(path: str) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return path


def _case_cfg_path(case_dir: str, config_name: str = "reproduce.json") -> str:
    return os.path.join(os.path.abspath(case_dir), config_name)


def _load_case_config(case_cfg_path: str) -> Dict[str, Any]:
    with open(case_cfg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"case config is not a json object: {case_cfg_path}")
    if data.get("schema_version", None) != 1:
        raise ValueError(f"Unsupported schema_version in {case_cfg_path}: {data.get('schema_version', None)}")
    if "defaults" in data and not isinstance(data["defaults"], dict):
        raise ValueError(f"Invalid 'defaults' in {case_cfg_path}: must be an object")
    for mode in ("wl", "thermal"):
        if mode not in data or not isinstance(data[mode], dict):
            raise ValueError(f"Missing or invalid '{mode}' in {case_cfg_path}")
    return data


def _select_mode_cfg(case_cfg: Dict[str, Any], mode: str, case_cfg_path: str) -> Dict[str, Any]:
    cfg = case_cfg.get(mode)
    if not isinstance(cfg, dict):
        raise ValueError(f"Invalid mode config '{mode}' in {case_cfg_path}: must be an object")
    banned_keys = {"flip_opt", "dis_bet_chips"}
    found = sorted(k for k in banned_keys if k in cfg)
    if found:
        raise ValueError(f"Unsupported keys in {case_cfg_path} for mode '{mode}': {', '.join(found)}")
    return cfg


def _apply_cfg(params, system, defaults: Dict[str, Any], cfg: Dict[str, Any]) -> Dict[str, Any]:
    applied: Dict[str, Any] = {}
    if "ILPsolver" in defaults:
        params.ILPsolver = defaults["ILPsolver"]
        applied["ILPsolver"] = params.ILPsolver
    if "thermal_solver" in defaults:
        params.thermal_solver = defaults["thermal_solver"]
        applied["thermal_solver"] = params.thermal_solver

    params.warp_aware_opt = False
    params.plot_freq = -1
    applied["warp_aware_opt"] = bool(getattr(params, "warp_aware_opt", False))
    applied["plot_freq"] = int(getattr(params, "plot_freq", -1))

    if "floorplan_stages" in cfg:
        if not isinstance(cfg["floorplan_stages"], list) or len(cfg["floorplan_stages"]) == 0:
            raise ValueError("Invalid floorplan_stages in best_params (must be a non-empty list)")
        params.floorplan_stages = cfg["floorplan_stages"]
        applied["floorplan_stages"] = cfg["floorplan_stages"]

    parser_keys = {
        "interposer_size",
        "fence_width",
        "fence_height",
        "num_bins_x",
        "num_bins_y",
        "num_grid_x",
        "num_grid_y",
        "reso_interposer",
    }
    for k, v in cfg.items():
        if k in {"floorplan_stages"} or k in parser_keys:
            continue
        setattr(params, k, v)
        applied[k] = v

    if "wl_weight" in cfg:
        wl_raw = float(cfg["wl_weight"])
        num_nets = max(1, int(getattr(system, "num_nets", 1)))
        params.wl_weight = wl_raw / num_nets
        applied["wl_weight_raw"] = wl_raw
        applied["wl_weight"] = float(params.wl_weight)

    if not hasattr(params, "overflow_init"):
        params.overflow_init = 0.3
        applied["overflow_init"] = float(params.overflow_init)
    if not hasattr(params, "density_weight_init"):
        params.density_weight_init = float(getattr(params, "density_weight", 3.0))
        applied["density_weight_init"] = float(params.density_weight_init)
    if not hasattr(params, "wl_weight"):
        params.wl_weight = 1e-2 / max(1, int(getattr(system, "num_nets", 1)))
        applied["wl_weight"] = float(params.wl_weight)
    if not hasattr(params, "temp_weight_init"):
        params.temp_weight_init = 1e2
        applied["temp_weight_init"] = float(params.temp_weight_init)
    if not hasattr(params, "warp_weight_init"):
        params.warp_weight_init = 1e2
        applied["warp_weight_init"] = float(params.warp_weight_init)

    _normalize_floorplan_stages(params, system)
    return applied


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


def _train_temp_model(params, system, thermal_dir: str) -> Any:
    system.thermal_dir = thermal_dir
    params.thermal_dir = thermal_dir

    solver = HotSpot_solver(system)
    solver.amb = float(getattr(params, "ambient", 45))

    num_dataset = int(getattr(params, "num_thermal_dataset", 1))
    if num_dataset <= 0:
        raise ValueError(f"Invalid num_thermal_dataset: {num_dataset}")

    data_loader = Datagen(num_dataset, system, solver, plot_flag=False)
    lr = 1e-1
    num_iters = 800
    criterion = lambda out, label, mask: ((out - label).pow(2)).mean()
    tempmodel = train_model(
        system.intp_width,
        system.intp_height,
        system.num_chiplets,
        system.num_grid_x,
        system.num_grid_y,
        lr,
        data_loader,
        criterion,
        num_iters,
    )
    for p in tempmodel.parameters():
        p.requires_grad_(False)
    return tempmodel, solver


def _write_temp_params_json(overrides: Dict[str, Any]) -> str:
    p = Path(tempfile.mkdtemp(prefix="atplace-")) / "params_overrides.json"
    with open(str(p), "w", encoding="utf-8") as f:
        json.dump(overrides, f, indent=2, ensure_ascii=False)
    return str(p)


def _split_cfg_for_parser(cfg: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    parser_keys = {
        "interposer_size",
        "fence_width",
        "fence_height",
        "num_bins_x",
        "num_bins_y",
        "num_grid_x",
        "num_grid_y",
        "reso_interposer",
    }
    parser_cfg: Dict[str, Any] = {}
    runtime_cfg: Dict[str, Any] = {}
    for k, v in cfg.items():
        if k in parser_keys:
            parser_cfg[k] = v
        else:
            runtime_cfg[k] = v
    return parser_cfg, runtime_cfg


def run_case(
    case_dir: str,
    out_dir: Optional[str],
    seed: int = 1000,
    mode: str = "wl",
    thermal: bool = True,
    plots: bool = True,
    thermal_dir: Optional[str] = None,
    config_name: str = "reproduce.json",
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    case_dir = os.path.abspath(case_dir)
    case_name = os.path.basename(os.path.normpath(case_dir))
    if out_dir is None:
        out_dir = os.path.join(_repo_root(), "results", case_name, mode)
    out_dir = _ensure_dir(out_dir)

    case_cfg_path = _case_cfg_path(case_dir, config_name)
    case_cfg = _load_case_config(case_cfg_path)
    cfg = _select_mode_cfg(case_cfg, mode, case_cfg_path)
    parser_cfg, runtime_cfg = _split_cfg_for_parser(cfg)
    defaults = case_cfg.get("defaults", {}) or {}

    if mode not in {"wl", "thermal"}:
        raise ValueError(f"Unsupported mode: {mode}")

    if "interposer_size" in parser_cfg:
        size = parser_cfg["interposer_size"]
        if not (isinstance(size, (list, tuple)) and len(size) == 2):
            raise ValueError(f"Invalid interposer_size in {case_cfg_path}: {size}")

    spec_param_file = _write_temp_params_json(parser_cfg) if parser_cfg else None
    params, system, _ = read_data(param_file=None, spec_param_file=spec_param_file, case_dir=case_dir)
    params.random_seed = int(seed)
    np.random.seed(int(seed))

    system.thermal_dir = (thermal_dir or _default_thermal_dir())
    params.thermal_dir = system.thermal_dir

    applied_cfg = _apply_cfg(params, system, defaults, runtime_cfg)

    if mode == "thermal":
        params.temp_aware_opt = True
        tempmodel, temp_solver = _train_temp_model(params, system, system.thermal_dir)
    else:
        params.temp_aware_opt = False
        tempmodel, temp_solver = None, None

    hpwl, best_metric, best_fp = placeflow(params, system, tempmodel=tempmodel, temp_solver=temp_solver, temp_sim=False)

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
        "mode": mode,
        "hpwl": float(hpwl),
        "best_pos": (x_final, y_final, theta_final),
        "intp_width": float(getattr(system, "intp_width", 0.0)),
        "intp_height": float(getattr(system, "intp_height", 0.0)),
        "runtime_s": float(time.perf_counter() - t0),
        "config_path": case_cfg_path,
    }
    result["applied_cfg"] = applied_cfg

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


def _iter_case_dirs(cases_root: str) -> List[str]:
    cases_root = os.path.abspath(cases_root)
    if not os.path.isdir(cases_root):
        raise ValueError(f"cases_root is not a directory: {cases_root}")
    out: List[str] = []
    for p in sorted(Path(cases_root).iterdir()):
        if p.is_dir() and p.name.startswith("Case"):
            out.append(str(p))
    if not out:
        raise ValueError(f"No case directories found under: {cases_root}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default=None, help="Case directory, e.g. cases/Case1")
    parser.add_argument("--cases-root", default=os.path.join(_repo_root(), "cases"), help="Cases root directory")
    parser.add_argument("--all", action="store_true", help="Run all Case* under --cases-root")
    parser.add_argument("--mode", choices=["wl", "thermal", "both"], default="both")
    parser.add_argument("--config-name", default="reproduce.json", help="Per-case config file name")
    parser.add_argument("--out", default=None, help="Output directory (default: results/<CaseName>/<mode>)")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--no-thermal", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--skip-failed", action="store_true", help="Skip cases that fail to run")
    parser.add_argument("--thermal-dir", default=None, help="Thermal directory (default: <repo>/thermal/)")
    args = parser.parse_args()

    if not args.all and not args.case:
        raise SystemExit("Need --case <cases/CaseX> or --all")

    case_dirs = _iter_case_dirs(args.cases_root) if args.all else [os.path.abspath(args.case)]

    outputs: List[str] = []
    failures: List[str] = []
    for case_dir in case_dirs:
        case_name = os.path.basename(os.path.normpath(case_dir))
        modes = ["wl", "thermal"] if args.mode == "both" else [args.mode]
        for mode in modes:
            out_dir = os.path.abspath(args.out) if args.out else os.path.join(_repo_root(), "results", case_name, mode)
            try:
                run_case(
                    case_dir=case_dir,
                    out_dir=out_dir,
                    seed=args.seed,
                    mode=mode,
                    thermal=not args.no_thermal,
                    plots=not args.no_plots,
                    thermal_dir=args.thermal_dir,
                    config_name=args.config_name,
                )
                outputs.append(out_dir)
            except Exception as e:
                msg = f"{case_name}/{mode}: {e}"
                if args.skip_failed:
                    failures.append(msg)
                    print(f"SKIP {msg}")
                    continue
                raise

    for item in outputs:
        print(item)
    if failures:
        print("FAILED_SUMMARY")
        for item in failures:
            print(item)


if __name__ == "__main__":
    main()
