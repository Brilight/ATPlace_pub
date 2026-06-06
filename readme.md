# ATPlace2.5D Public Package

This repository contains a compact public package for ATPlace2.5D.

## Contents

- `cases/`: ten clean case directories. Each case contains Bookshelf-style input files, `WL-driven.json`, and `Thermal-aware.json`.
- `src/`: encrypted ATPlace layout kernel and its runtime.
- `thermal/`: HotSpot files used by thermal evaluation.
- `utils/`: parsers for case input files.
- `Thermal.py`: thermal helper entry file.
- `reproduce.sh`: minimal shell wrapper for selecting a case and placement mode.

## Usage

Create a Python environment with the package dependencies used by the encrypted kernel. `gurobipy` is the recommended solver interface, and the parameter files set `ILPsolver` to `grb`.

Each case directory contains Bookshelf input files and two layout parameter files.

- `WL-driven.json` is for wirelength-driven placement.
- `Thermal-aware.json` is for thermal-aware placement.

`pulp` may work in some environments, but it is not guaranteed to produce strictly identical legal placement or numerical values. Use `gurobipy` when reproducing reported layouts.

Run the wrapper as:

```bash
bash reproduce.sh Case1 wl
bash reproduce.sh Case1 thermal
```

The first argument is one of `Case1` through `Case10`. The second argument is `wl` or `thermal`. The wrapper prints the selected case directory and parameter file, which can be passed to the encrypted layout kernel together with the case input files.

The case parameter files are self-contained layout settings. Do not change interposer geometry unless the target design itself changes.

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
