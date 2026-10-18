# Classification and Signatures of Dissipative Quantum Hall–Superconductor Junctions

Requires Python 3.12 and LaTeX (`latex`, `dvipng`, Computer Modern, `type1cm`/`type1ec`).

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python phase_diagram_pipeline.py
python conductance_pipeline.py
python plot_conductance.py
python -m unittest
```

PDFs and PNGs go to `paper_figures/` (override with `--output-dir PATH`).
The phase pipeline makes Figs. 3–4; `plot_conductance.py` makes Figs. 6, 7, 9, and 10.
`conductance_pipeline.py` reads `conductance_parameters.json` and caches numerical data
in `generated_data_finite/` (`--data-dir PATH` in both scripts). Use `--recompute-data`
to rebuild it; plotting only reads saved data.

Custom transport: `conductance_pipeline.build_model(K, U, vectors, coefficients, ...)`
accepts dense `U` and explicit boundary matrices; its docstring lists all parameters.
Presets use this builder; dimensions, spins, and velocities follow from `K` and `U`.