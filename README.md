# ATPlace2.5D

This repository contains the public materials for the ICCAD 2024 paper **ATPlace2.5D**. Due to intellectual property restrictions, we are unable to directly provide the core code of ATPlace 🔒. However, this repository includes the following resources:

1. **Test Cases**: A set of test cases used in our experiments.
2. **Thermal Configuration**: The thermal setup used in our work, which is processed using [HotSpot](https://github.com/uvahotspot/HotSpot.git). You can compile HotSpot-7.0 on your own 💻.
3. **Partial Code**: A subset of the TAP2.5D codebase 🧩.

In the future, we plan to release a compiled version of the ATPlace code to further support research in this area 🚀.

---

## Repository Contents

### Test Cases 
The test cases are organized into **10 folders** (`Case1` to `Case10`), each containing the following files:
- **Chiplet Geometric Sizes**: `*.blocks` files that define the geometric sizes of chiplets.
- **Nets**: `*.nets` files that describe the netlist connectivity.
- **Pin Locations**: `*.pl` files that specify the pin locations.
- **Chiplet Power**: `*.power` files that define the power consumption of chiplets.

These files are formatted based on the **Bookshelf format**, a widely-used standard in placement research. The parsers can be fount in `utils'.

---

### Thermal Configuration
This repo vendors a HotSpot binary and default configs under `thermal/`. The public runner uses it for thermal evaluation and dataset generation in thermal-aware mode.

---

## Future Plans 🤝
We are committed to open-sourcing more components of ATPlace in the future. Specifically, we plan to release a compiled version of the ATPlace code to enable researchers to reproduce and build upon our work.

---

## How to Use This Repository
1. Clone the repository:
   ```bash
   git clone https://github.com/Brilight/ATPlace_pub.git
2. Explore the test cases in the cases directory.
4. Results can be compared with the published data in our paper.

### Reproduce (Public Runner)
Each case stores its own reproducible parameters in `cases/CaseX/reproduce.json`.

```bash
python -m pip install -r src/requirements.txt
python src/reproduce.py --case cases/Case1 --mode both
```

- The command reads `cases/Case1/reproduce.json`.
- Outputs are written to `results/<CaseName>/{wl,thermal}/`, including `metrics.json`, `placement.npz`, and `params_resolved.json`.

### Batch Run
```bash
python src/reproduce.py --all --mode both --no-plots --skip-failed
```

- `--skip-failed` keeps batch reproduction running when a case has no feasible floorplan under its current parameters.
- No Jupyter notebook is required for reproduction; use the CLI runner and per-case configs only.

## Citation 📋
If you find this repository useful for your research, please consider citing our ICCAD 2024 paper:
    
```
@inproceedings{wang2024atplace2,
  title={ATPlace2. 5D: Analytical thermal-aware chiplet placement framework for large-scale 2.5 D-IC},
  author={Wang, Qipan and Li, Xueqing and Jia, Tianyu and Lin, Yibo and Wang, Runsheng and Huang, Ru},
  booktitle={Proceedings of the 43rd IEEE/ACM International Conference on Computer-Aided Design},
  pages={1--9},
  year={2024}
}
```

## Contact
For questions or feedback, feel free to reach out:

Email: qpwang@pku.edu.cn
GitHub Issues: Open an issue in this repository.
