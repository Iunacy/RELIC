# Executed validation commands

Working directory: `/home/iris/RELIC`. Starting branch/commit: `feature/adaptive-v0` / `323183dd3cd773384c4d9eb4217e659005c3b99f`. Revision branch: `feature/adaptive-v0-validation`.

Baseline before modification:

```sh
ctest --test-dir build-adaptive --output-on-failure -j 4
git switch -c feature/adaptive-v0-validation
```

Build and release checks:

```sh
cmake -S . -B build-adaptive-validation -DRELIC_ADAPTIVE=ON -DMUJOCO_ROOT=/home/iris/.local/opt/mujoco-3.16.0 -DCMAKE_BUILD_TYPE=Release
cmake --build build-adaptive-validation -j 6
ctest --test-dir build-adaptive-validation --output-on-failure -j 4
ctest --test-dir build-adaptive-validation -R 'adaptive_learning|adaptive_revision' --output-on-failure
```

An initial configure used the unused `RELIC_BUILD_ADAPTIVE` flag; this was corrected to `RELIC_ADAPTIVE` before building/testing the adaptive executable. Targeted checks and the full suite were repeated after implementation changes. The final full-suite log is `adaptive/runs/revision-20261022/tests-completion-latest.log`.

Executed experiment stages (independent collection, development tuning, baseline evaluation and convergence work overlapped):

```sh
python adaptive/revision.py qualify
python adaptive/revision.py teacher-qualify
python adaptive/revision.py collect
python adaptive/revision.py tune
python adaptive/revision.py convergence
python adaptive/revision/convergence_replication.py
python adaptive/revision.py evaluate-baselines
python adaptive/revision/historical_probe.py
/home/iris/.local/share/venvs/robotics-ml/bin/python adaptive/revision/training_probe.py
/home/iris/.local/share/venvs/robotics-ml/bin/python adaptive/revision.py prepare
/home/iris/.local/share/venvs/robotics-ml/bin/python adaptive/revision.py train
/home/iris/.local/share/venvs/robotics-ml/bin/python adaptive/revision.py evaluate
/home/iris/.local/share/venvs/robotics-ml/bin/python adaptive/revision.py report
/home/iris/.local/share/venvs/robotics-ml/bin/python adaptive/revision/convergence_report.py
/home/iris/.local/share/venvs/robotics-ml/bin/python adaptive/revision/analytical_robustness.py
python adaptive/revision/repeatability.py
python adaptive/revision_demo.py
python adaptive/revision_demo.py --fixtures
/home/iris/.local/share/venvs/robotics-ml/bin/python adaptive/revision_plot.py adaptive/runs/revision-20261022
/home/iris/.local/share/venvs/robotics-ml/bin/python adaptive/revision/export_results.py
python adaptive/revision/check_history.py
```

The first serial `prepare` process was terminated with SIGTERM to replace it with four independent readers. Its log remains `preparation-serial.log`; completed arrays were reused after exact equality checks. The parallel run rechecked the same acceptance gates and verification hashes. No physical campaign run or rejected proposal was removed.

Sanitizer checks:

```sh
cmake -S . -B build-adaptive-validation-sanitize -DRELIC_ADAPTIVE=ON -DMUJOCO_ROOT=/home/iris/.local/opt/mujoco-3.16.0 -DCMAKE_BUILD_TYPE=Debug '-DCMAKE_CXX_FLAGS=-fsanitize=address,undefined -fno-omit-frame-pointer' '-DCMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined'
cmake --build build-adaptive-validation-sanitize -j 2
ctest --test-dir build-adaptive-validation-sanitize -R adaptive --output-on-failure -j 1
```

The pinned native executable SHA is in `run_inventory.json`. Every physical execution retains its exact argument vector in `metadata.json` under `native_argv`; controller configurations, proposal tables, weights, trace hashes and physical parameters are retained beside the traces. `environment.json`, `split_protocol.json`, `analytical_robustness_protocol.json` and `training_results.json` pin the experiment inputs and software versions.

Full reproduction is `bash adaptive/revise.sh`. Quick inference/motion reproduction after the native build is `python adaptive/revision_demo.py --fixtures`; no bulk data or PyTorch runtime is needed for this short demo.

# Changed files and purpose

- `adaptive/residual.hpp`, `adaptive/episode.cpp`: explicit stopped result, rate-limited latched withdrawal, shared command/observed-motion checks, gravity/velocity and anti-windup integral controllers, failure injection, zero/copy controls, common-joint transfer and per-joint error metrics.
- `adaptive/learning.py`, `adaptive/pipeline.py`: controller-config provenance, causal stride-efficient features, restored_nominal default, distinct withdrawal termination and per-joint metric consistency validation.
- `adaptive/revision.py`, `adaptive/revision_learning.py`, `adaptive/revision_geometry.py`: frozen deterministic split, qualified/resumable teacher collection, gate/hash audit, four-reader preparation, matched and full model fits, paired family bootstrap and complete comparison matrix checks.
- `adaptive/revision_plot.py`, `adaptive/revision_demo.py`, `adaptive/revise.sh`, root `README.md`: compact reproduction, scientific plots and a runnable actual-motion demo using small committed weights.
- `adaptive/tests.cpp`, `adaptive/test_learning.py`, `adaptive/test_revision.py`, `adaptive/CMakeLists.txt`: nonzero invalid-output withdrawal, both signs/dimensions, infeasible stop, causality, split/duplicate checks and interrupted resumption.
- `adaptive/revision/*.py`: historical preservation/replay, small training probe, broader ILC convergence replication, separate stronger-PI robustness check, repeatability and result export.
- `adaptive/revision/models`, bounds/configuration fixtures and JSON/Markdown/PNG reports: small versioned runtime/reproducibility artifacts. Historical evidence/fixtures remain unchanged.

Pre-existing root `.gitignore` and `CMakeLists.txt` modifications, untracked experiment files and older adaptive Markdown/status files were not committed by this revision. Production `src/`, `include/`, model configurations, Ruckig and double-S experiment files were not modified.
