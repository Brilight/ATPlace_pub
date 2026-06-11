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


def build_compact_model(params: Params.Params, system: System_25D):
    if not params.temp_aware_opt:
        return None

    import torch
    import torch.nn as nn

    class AnalyticThermalModel(nn.Module):
        def __init__(self, width, height, num_chiplets, num_grid_x, num_grid_y):
            super().__init__()
            self.width = width
            self.height = height
            self.num_chiplets = num_chiplets
            xgrid = (torch.arange(num_grid_x) + 0.5) / num_grid_x * width
            ygrid = (torch.arange(num_grid_y) + 0.5) / num_grid_y * height
            xgrid, ygrid = torch.meshgrid(xgrid, ygrid, indexing="ij")
            self.register_buffer("xgrid", xgrid[None, None])
            self.register_buffer("ygrid", ygrid[None, None])
            self.amp = nn.Parameter(torch.ones(1) * 1e3)
            self.bias = nn.Parameter(torch.zeros(1))
            self.heff = nn.Parameter(torch.ones(1))
            self.decay = nn.Parameter(torch.ones(1, num_chiplets, 1, 2))

        def forward(self, input_data):
            x, y, length, width, power = input_data
            chips = self.num_chiplets
            batch = x.shape[0]
            xc = x.view(-1, chips, 1, 1)
            yc = y.view(-1, chips, 1, 1)
            lc = length.view(-1, chips, 1, 1)
            wc = width.view(-1, chips, 1, 1)
            xgrid = self.xgrid.expand(batch, chips, -1, -1)
            ygrid = self.ygrid.expand(batch, chips, -1, -1)
            power = power.reshape(-1, chips, 1, 1)
            val = self._main_term(xgrid - xc, ygrid - yc, lc, wc)
            return (power * (self.amp * val + self.bias)).sum(dim=1, keepdim=True)

        def _main_term(self, xdist, ydist, length, width):
            decay = self.decay
            ax = decay[..., :1]
            ay = decay[..., 1:2]
            val = (
                self._fabc(self.heff, ax * (length / 2 - xdist), ay * (width / 2 - ydist))
                + self._fabc(self.heff, ax * (length / 2 - xdist), ay * (width / 2 + ydist))
                + self._fabc(self.heff, ax * (length / 2 + xdist), ay * (width / 2 - ydist))
                + self._fabc(self.heff, ax * (length / 2 + xdist), ay * (width / 2 + ydist))
            )
            return val / length / width

        @staticmethod
        def _fabc(a, b, c):
            a = a.double()
            b = b.double()
            c = c.double()
            delta = torch.sqrt(a**2 + b**2 + c**2)
            val = (
                b * torch.log((c + delta) / (a**2 + b**2) ** 0.5)
                + c * torch.log((b + delta) / (a**2 + c**2) ** 0.5)
                - a * torch.arctan(b * c / a / delta)
            )
            return val.float()

    thermal = AnalyticThermalModel(
        system.intp_width,
        system.intp_height,
        system.num_chiplets,
        system.num_grid_x,
        system.num_grid_y,
    )
    return {"Thermal": thermal}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def to_plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "numpy"):
        return value.detach().cpu().numpy().tolist()
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(item) for item in value]
    return value


def flatten_numbers(value):
    if isinstance(value, (int, float, np.number)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        values = []
        for item in value:
            values.extend(flatten_numbers(item))
        return values
    return []


def export_layout(best_fp, system: System_25D, params: Params.Params, case_name: str, mode: str) -> dict:
    raw = to_plain(best_fp)
    pos = raw
    if isinstance(raw, dict):
        for key in ("pos", "position", "best_fp", "best_fp_pos"):
            if key in raw:
                pos = raw[key]
                break

    chiplets = []
    if isinstance(pos, list) and len(pos) >= 2:
        xy = flatten_numbers(pos[0])
        angles = flatten_numbers(pos[1])
        if len(xy) >= 2 * system.num_nodes:
            for idx in range(system.num_chiplets):
                angle = angles[idx] if idx < len(angles) else 0.0
                chiplets.append({
                    "name": system.node_names[idx],
                    "x": xy[idx],
                    "y": xy[idx + system.num_nodes],
                    "width": float(system.node_size_x[idx]),
                    "height": float(system.node_size_y[idx]),
                    "angle_rad": angle,
                    "power_w": float(system.powermap[idx]) if idx < len(system.powermap) else 0.0,
                })

    return {
        "case": case_name,
        "mode": mode,
        "unit": "um",
        "interposer": {
            "width": float(system.intp_width),
            "height": float(system.intp_height),
            "fence": [float(system.xlow), float(system.xhigh), float(system.ylow), float(system.yhigh)],
        },
        "thermal": {
            "temp_aware_opt": bool(params.temp_aware_opt),
            "thermal_solver": str(params.thermal_solver),
            "thermal_dir": str(params.thermal_dir),
            "num_grid_x": int(params.num_grid_x),
            "num_grid_y": int(params.num_grid_y),
        },
        "chiplets": chiplets,
        "raw_best_fp": raw,
    }


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
    params.thermal_dir = os.environ.get("ATPLACE_THERMAL_DIR", str(ROOT / "thermal")) + os.sep
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
    system.num_grid_x = params.num_grid_x
    system.num_grid_y = params.num_grid_y
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
    compact_model = build_compact_model(params, system)
    if params.temp_aware_opt:
        print(
            "thermal_model=public_analytic_wrapper; thermal_helper_imported=false; "
            "visible_training_step=false"
        )
    else:
        print("thermal_model=disabled")

    start = time.time()
    result = placeflow_core(params, system, compact_model)
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
        "thermal_model": "public_analytic_wrapper" if compact_model is not None else "disabled",
        "thermal_helper_imported": False,
        "visible_training_step": False,
    }
    if best_fp is not None:
        layout = export_layout(best_fp, system, params, args.case, args.mode)
        layout_path = out_dir / "layout.json"
        write_json(layout_path, layout)
        summary["layout_json"] = str(layout_path)
        summary["layout_chiplets"] = len(layout["chiplets"])
    write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
