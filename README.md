# RELIC

RELIC is a C++20 and MuJoCo robotic manipulation research system for synthetic 3-DOF and 4-DOF arms. It provides analytical inverse kinematics, motion planning, sampled collision checks, position-servo execution, and execution telemetry.

The optional adaptive pipeline generates seeded physical-degradation episodes, validates and splits datasets by trajectory family, trains bounded offline residual models, and compares conventional, analytical, and learned control in closed-loop simulation. The conventional controller remains the default. This project does not actuate hardware or require Isaac Lab, GR00T, Cosmos, or cloud services.

Run the tested qualification, training, evaluation, and motion demonstration workflow with:

```sh
bash adaptive/revise.sh
```

Set `MUJOCO_ROOT` to a MuJoCo 3.16.0 SDK and `RELIC_ML_PYTHON` to a Python environment with the packages in `adaptive/requirements.txt`. Small fixtures and measured reproducibility reports are retained under `adaptive/fixtures` and `adaptive/evidence`. Bulk datasets, generated builds, and temporary runs are excluded from Git.

The validation revision uses paired held-out families, stronger analytical competitors and rate-limited correction withdrawal. The original campaign remains reproducible with `adaptive/reproduce.sh`; its results are retained as historical development evidence.

After building the adaptive executable, `python adaptive/revision_demo.py --fixtures` runs a short four-controller demonstration using the committed small model, without collecting the campaign.
