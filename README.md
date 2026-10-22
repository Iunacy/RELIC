# RELIC

RELIC is a C++20 and MuJoCo robotic manipulation research system for synthetic 3-DOF and 4-DOF arms. It provides analytical inverse kinematics, motion planning, sampled collision checks, position-servo execution, and execution telemetry.

The optional adaptive pipeline generates seeded physical-degradation episodes, validates and splits datasets by trajectory family, trains bounded offline residual models, and compares conventional, analytical, and learned control in closed-loop simulation. The conventional controller remains the default. This project does not actuate hardware or require Isaac Lab, GR00T, Cosmos, or cloud services.

Run the tested qualification, training, evaluation, and motion demonstration workflow with:

```sh
bash adaptive/revise.sh
```

Set `MUJOCO_ROOT` to a MuJoCo 3.16.0 SDK and `RELIC_ML_PYTHON` to a Python environment with the packages in `adaptive/requirements.txt`. Small fixtures and measured reproducibility reports are retained under `adaptive/fixtures` and `adaptive/evidence`. Bulk datasets, generated builds, and temporary runs are excluded from Git.

The validation revision uses paired held-out families, stronger analytical competitors and rate-limited correction withdrawal. The original campaign remains reproducible with `adaptive/reproduce.sh`; its results are retained as historical development evidence.

After building the adaptive executable, `python adaptive/revision_demo.py --fixtures` runs a short five-controller demonstration using the committed small model, without collecting the campaign.

The interactive Physical AI demo adds the original Menagerie Koch arm, known Cartesian targets, measured actuator execution and a local browser interface. Run `bash interactive/setup.sh` once, then `bash interactive/run.sh` and open `http://127.0.0.1:8765`. Choose direct target selection for offline use. Synthetic-arm mode compares conventional, analytical PI and the existing bounded learned residual; learned weights are disabled for the imported arm.

NVIDIA Nemotron task selection is implemented through Nebius Token Factory; successful live verification is pending the API key. See [setup, safety boundaries and recording instructions](interactive/README.md). Historical adaptive results and original synthetic model configurations are preserved.
