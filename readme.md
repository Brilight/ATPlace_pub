# ATPlace2.5D Public Package

This repository contains a compact public package for ATPlace2.5D.

## Contents

- `cases/`: ten clean case directories. Each case contains Bookshelf-style input files, `WL-driven.json`, and `Thermal-aware.json`.
- `src/`: encrypted ATPlace layout kernel and its runtime.
- `thermal/`: HotSpot files used by thermal evaluation.
- `utils/`: parsers for case input files.
- `Thermal.py`: optional readable HotSpot helper for users who want to inspect or reuse the floorplan/layer/power-trace generation logic.
- `reproduce.sh`: shell entry for selecting a case and placement mode and starting the encrypted layout kernel.

## Usage

Create a Python environment with the package dependencies used by the encrypted kernel. `gurobipy` is the recommended solver interface, and the parameter files set `ILPsolver` to `grb`.

Each case directory contains Bookshelf input files and two layout parameter files.

- `WL-driven.json` is for wirelength-driven placement.
- `Thermal-aware.json` is for thermal-aware placement.

`pulp` may work in some environments, but it is not guaranteed to produce strictly identical legal placement or numerical values. Use `gurobipy` when reproducing reported layouts.

On the 231 server, activate the reproduction environment first:

```bash
conda activate ATPlan
```

Run one case as:

```bash
bash reproduce.sh Case1 wl
bash reproduce.sh Case1 thermal
```

The first argument is one of `Case1` through `Case10`. The second argument is `wl` or `thermal`.

The command selects:

- input files from `cases/<CaseX>/`;
- `WL-driven.json` for `wl`;
- `Thermal-aware.json` for `thermal`;
- output directory `results/<CaseX>_<mode>_<timestamp>/` unless `ATPLACE_OUT_DIR` is set.

Then it calls `reproduce.py`, which loads the selected Bookshelf files and parameter file and invokes the encrypted `ATPLACE.PlaceFlow.placeflow_core` kernel. A successful execution writes:

- `summary.json`: selected case, mode, parameter file, output directory, HPWL/TWL, runtime, and whether a final layout was returned;
- `layout.json`: final chiplet coordinates and chiplet sizes when the encrypted kernel returns a final layout.

Optional environment variables:

```bash
PYTHON=/path/to/python bash reproduce.sh Case1 wl
ATPLACE_OUT_DIR=/tmp/case1_wl bash reproduce.sh Case1 wl
ATPLACE_THERMAL_DIR=/path/to/thermal bash reproduce.sh Case1 thermal
ATPLACE_DRY_RUN=1 bash reproduce.sh Case1 wl
```

The case parameter files are self-contained layout settings. Do not change interposer geometry unless the target design itself changes.

## Thermal-Aware Mode

Thermal-aware placement is enabled by `temp_aware_opt` in `Thermal-aware.json`.

`reproduce.py` builds the case, creates a compact analytic thermal-model object, and passes that object into the encrypted `ATPLACE.PlaceFlow.placeflow_core` kernel. The public wrapper does not run a separate visible thermal-model training script and does not write a thermal-model checkpoint before placement. It is therefore expected that a thermal run may print the same early placement messages as a wirelength run after a short initialization step.

`Thermal.py` is not imported by `reproduce.py` in the current public wrapper. Prints added to `Thermal.py` will not appear during `bash reproduce.sh Case1 thermal` unless a user imports and calls that helper directly. The encrypted kernel may call its own internal thermal evaluation path. The readable `Thermal.py` file is kept as an optional reference implementation for generating HotSpot `.flp`, `.lcf`, `.ptrace`, and configuration files from a chiplet layout.

`thermal/` contains the HotSpot executable and default HotSpot configuration used by public thermal evaluation resources. Runtime thermal directory selection is controlled by `ATPLACE_THERMAL_DIR`.

The main thermal-aware layout controls are in `cases/<CaseX>/Thermal-aware.json`:

- `temp_aware_opt`: enables the thermal-aware objective;
- `temp_weight_init`, `temp_threshold`, and `gamma`: control the thermal penalty;
- `wl_weight`: controls the wirelength term in the thermal-aware objective;
- `target_density`, `density_weight_init`, `overflow_init`, and `eta`: control placement density and optimization behavior;
- `floorplan_stages[0].learning_rate`: position and angle learning rates.

Use these JSON files for ordinary parameter changes. Keep each case's `interposer_size` unchanged unless the physical case definition changes.

## Changing Thermal Configuration

For a new thermal configuration, copy the thermal directory and point the run to it:

```bash
cp -R thermal thermal_custom
# edit thermal_custom/hotspot.config
ATPLACE_THERMAL_DIR="$PWD/thermal_custom" bash reproduce.sh Case1 thermal
```

Editing `Thermal.py` changes only that readable helper. It does not change `reproduce.py` or the encrypted placement kernel unless those files are modified to import and call the helper.

Changing layer material constants, layer thicknesses, bump parameters, or generated `.flp`/`.lcf`/`.ptrace` content requires a wrapper or evaluation script that explicitly uses `Thermal.py`.

## Running the Full Reproduction Matrix

To run all ten cases in both modes, with three repeats per case-mode element and at most five cases active at the same time:

```bash
python reproduce_matrix.py --repeats 3 --case-workers 5
```

The script writes:

- `repro_runs/<timestamp>/commands.json`: all commands launched;
- `repro_runs/<timestamp>/<CaseX>/<mode>/rep*/stdout.log` and `stderr.log`;
- `repro_runs/<timestamp>/summary.csv`: all completed repeats and metrics parsed from `summary.json`;
- `repro_runs/<timestamp>/best.csv`: best repeat per case-mode element, ranked by the smallest `twl_m`.

For a new case, create `cases/<NewCase>/` with matching Bookshelf-style files:

```text
<NewCase>.blocks
<NewCase>.nets
<NewCase>.pl
<NewCase>.power
WL-driven.json
Thermal-aware.json
```

Then either add the case interposer size to `CASE_INTERPOSER_SIZE` in `reproduce.py`, or include `interposer_size` in the case parameter JSON.

## Layout Visualization

After a successful run, generate a PNG from the exported layout:

```bash
python visualize_layout.py results/Case1_wl_YYYYMMDD_HHMMSS/layout.json
python visualize_layout.py results/Case1_thermal_YYYYMMDD_HHMMSS/layout.json --out case1_thermal.png
```

The visualizer reads only `layout.json`. It does not call the encrypted placement kernel.

## Citation

```bibtex
@inproceedings{wang2024atplace2,
  title={ATPlace2. 5D: Analytical thermal-aware chiplet placement framework for large-scale 2.5 D-IC},
  author={Wang, Qipan and Li, Xueqing and Jia, Tianyu and Lin, Yibo and Wang, Runsheng and Huang, Ru},
  booktitle={Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design},
  pages={1--9},
  year={2024}
}
```
