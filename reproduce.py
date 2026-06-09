#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(ROOT))

import Params  # noqa: E402
from Chiplet import Chiplet  # noqa: E402
from Interposer import Passive_Interposer  # noqa: E402
from System import System_25D  # noqa: E402
from ATPLACE.PlaceFlow import placeflow_core  # noqa: E402
from utils.blocks_parser import parse_blocks  # noqa: E402
from utils.nets_parser import parse_nets  # noqa: E402
from utils.pl_parser import parse_pls  # noqa: E402


CASE_INTERPOSER_SIZE = {
    "Case1": [42000.0, 42000.0],
    "Case2": [32000.0, 32000.0],
    "Case3": [39000.0, 39000.0],
    "Case4": [37000.0, 37000.0],
    "Case5": [57000.0, 59000.0],
    "Case6": [49000.0, 53000.0],
    "Case7": [30000.0, 25000.0],
    "Case8": [26000.0, 23000.0],
    "Case9": [59000.0, 61000.0],
    "Case10": [47000.0, 47000.0],
}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def normalize_stage(params: Params.Params, data: dict) -> None:
    default_stage = (Params.Params(SRC / "params.json").floorplan_stages or [{}])[0]
    stages = data.get("floorplan_stages") or params.floorplan_stages or [default_stage]
    merged = []
    for stage in stages:
        item = dict(default_stage)
        item.update(stage)
        merged.append(item)
    params.floorplan_stages = merged


def load_params(param_file: Path, case_name: str, out_dir: Path) -> Params.Params:
    params = Params.Params(SRC / "params.json")
    data = load_json(param_file)
    params.fromJson(data)
    normalize_stage(params, data)
    params.interposer_size = data.get("interposer_size") or CASE_INTERPOSER_SIZE[case_name]
    params.fence_width = getattr(params, "fence_width", 0.0)
    params.fence_height = getattr(params, "fence_height", 0.0)
    params.result_dir = str(out_dir)
    params.thermal_dir = str(ROOT / "thermal") + os.sep
    params.ILPsolver = getattr(params, "ILPsolver", "grb")
    params.thermal_solver = getattr(params, "thermal_solver", "hotspot")
    return params


def build_system(case_dir: Path, case_name: str, params: Params.Params) -> System_25D:
    options = {
        "filename_blocks": str(case_dir / f"{case_name}.blocks"),
        "filename_nets": str(case_dir / f"{case_name}.nets"),
        "filename_pl": str(case_dir / f"{case_name}.pl"),
    }
    modules, block_headers = parse_blocks(options)
    locations = parse_pls(options)
    nets, net_headers = parse_nets(options)

    num_chiplets = int(block_headers["Headers"]["NumHardRectilinearBlocks"])
    num_terminals = int(block_headers["Headers"]["NumTerminals"])
    system = System_25D(num_chiplets, num_terminals)
    interposer = Passive_Interposer()

    for module_name, module in modules["Modules"].items():
        if "rectangles" in module:
            chiplet = Chiplet(module_name)
            chiplet.set_chiplet_size(*module["rectangles"][0][-2:])
            chiplet.set_chiplet_loc(*module["rectangles"][0][:2])
            system.append_chiplet(module_name, chiplet)
        elif "terminal" in module:
            center = locations["Modules"][module_name]["center"]
            interposer.append_terminal(module_name, center)
            system.append_terminal(module_name, center)

    system.num_nets = int(net_headers["Headers"]["NumNets"])
    system.num_pins = int(net_headers["Headers"]["NumPins"]) - system.num_nodes + num_chiplets

    pin_id = 0
    for net_idx, net in enumerate(nets["Nets"]):
        system.net_id.append(net_idx)
        system.net_weights.append(1.0)
        system.net2pin_map.append([])
        for pin in net:
            node_id = system.node_name2id_map[pin[0]]
            if len(pin) < 3 or pin[2] is None:
                continue
            pin_offset_x = float(pin[1])
            pin_offset_y = float(pin[2])
            existing_pin = None
            for old_pin_id in system.node2pin_map[node_id]:
                if (
                    system.pin_offset_x[old_pin_id] == pin_offset_x
                    and system.pin_offset_y[old_pin_id] == pin_offset_y
                ):
                    existing_pin = old_pin_id
                    break
            if existing_pin is not None:
                system.net2pin_map[net_idx].append(existing_pin)
                system.pin2net_map[existing_pin].append(net_idx)
            else:
                system.net2pin_map[net_idx].append(pin_id)
                system.pin2net_map.append([net_idx])
                system.node2pin_map[node_id].append(pin_id)
                system.pin2node_map.append(node_id)
                system.pin_offset_x.append(pin_offset_x)
                system.pin_offset_y.append(pin_offset_y)
                pin_id += 1

    interposer.set_interposer_size(params.interposer_size)
    fence = [
        params.fence_width,
        interposer.width - params.fence_width,
        params.fence_height,
        interposer.height - params.fence_height,
    ]
    system.set_interposer_size(fence, interposer)
    system.set_bins(params)
    system.initialize()
    system.set_granularity(params.reso_interposer)
    system.area_cplt = (np.array(system.node_size_x) * np.array(system.node_size_y)).sum()

    system.powermap = np.zeros(num_chiplets)
    power_file = case_dir / f"{case_name}.power"
    if power_file.exists():
        with power_file.open(encoding="utf-8") as handle:
            for line in handle:
                parts = line.split()
                if len(parts) != 2:
                    continue
                name, power = parts
                if name in system.node_name2id_map:
                    system.powermap[system.node_names.index(name)] = float(power)
    return system


def unpack_result(result):
    if isinstance(result, dict):
        hpwl = result["hpwl"]
        best_fp_values = result.get("best_fp_pos", [])
        best_fp = best_fp_values[0] if best_fp_values else None
        return hpwl, best_fp
    hpwl, _best_metric, best_fp = result[:3]
    return hpwl, best_fp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", required=True)
    parser.add_argument("--mode", required=True, choices=["wl", "thermal"])
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--param-file", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    case_dir = Path(args.case_dir).resolve()
    param_file = Path(args.param_file).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    params = load_params(param_file, args.case, out_dir)
    system = build_system(case_dir, args.case, params)

    start = time.time()
    result = placeflow_core(params, system, None)
    hpwl, best_fp = unpack_result(result)
    summary = {
        "case": args.case,
        "mode": args.mode,
        "case_dir": str(case_dir),
        "param_file": str(param_file),
        "out_dir": str(out_dir),
        "hpwl": float(hpwl),
        "twl_m": float(hpwl) / 1e6,
        "runtime_s": time.time() - start,
        "has_best_fp": best_fp is not None,
    }
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
