# VIPER

*From intent to motion.*

![VIPER: OpenMANIPULATOR-X and Franka Panda perform the same pick-and-place task](assets/simulation.gif)

<sub>Recorded PhysX execution · MuJoCo state rendering · classical control · 4×</sub>

VIPER is a robotic manipulation research system exploring how intelligence takes shape across different bodies.

Its current foundation combines analytical kinematics, trajectory planning, bounded control, and verified pick-and-place execution with OpenMANIPULATOR-X and Franka Panda. A shared experiment viewer exposes their motion, timing, and physical outcomes, alongside controlled robustness comparisons.

Interfaces for language-conditioned GR00T policies and Nemotron planning are in place. Learned manipulation and cross-embodiment transfer remain unvalidated.

VIPER was formerly named RELIC. Historical commits, run schemas, package namespaces, datasets, and `RELIC_*` settings retain that identity for attribution and compatibility. The current repository is [Iunacy/viper](https://github.com/Iunacy/viper).

Run the recorded dual-robot evidence viewer with `bash interactive/run.sh`. It displays classical PhysX execution, synchronized playback, applied actions, next-tick tracking error, and intervention provenance. Learned transfer remains unavailable. The C++/Ruckig foundation is separate from Isaac control; `viper_cli` and the compatible `relic_cli` target run the same native implementation.

The shared policy preserves six OpenMANIPULATOR-X and nine Panda joint channels, yielding 16- and 22-dimensional native states. Both predict world-frame Cartesian increments and an open-positive gripper command, with robot-specific admissible grasp orientation. Model-capacity padding happens only in the pinned upstream processor. No joint channel is reinterpreted as another robot's joint.

The transfer preparation restores target projector slot 10 from the same local pretrained checkpoint for E2 and E3, inherits shared weights from the appropriate donor, and recomputes Panda statistics with the pinned upstream training-data reader. Synthetic checkpoint tests verify conversion rules; actual GR00T-weight loading, optimizer steps, and learned closed-loop execution remain unverified.

On an approved GPU host with locally cached model **and backbone** weights, validated physical LeRobot datasets, and the pinned upstream environment:

```bash
python -m embodiment.cloud --source "$GROOT_SOURCE" --pretrained "$PRETRAINED" --volume "$RUN_VOLUME"
python -m embodiment.cross_smoke --source "$GROOT_SOURCE" --dataset-a "$DATASET_A" --dataset-b "$DATASET_B" --pretrained "$PRETRAINED" --output "$RUN_VOLUME/g3"
# Explicit compute opt-in, after the dry run and separate spending approval:
python -m embodiment.cross_smoke --source "$GROOT_SOURCE" --dataset-a "$DATASET_A" --dataset-b "$DATASET_B" --pretrained "$PRETRAINED" --output "$RUN_VOLUME/g3" --execute
```

The dry run launches no training or simulator and downloads no model. Completed training stages are reused only after checkpoint integrity checks; failed attempts remain on disk and retries use new attempt directories. Optimizer recovery checks saved model, optimizer, scheduler and RNG integrity. The instrumented upstream worker records actual trainable parameter names, wall time and peak allocated GPU memory when execution is authorized. An optimizer/inference pass is not a full G3 pass: execute the target policy in the separate `embodiment.policy_service` and `embodiment.evaluate` paths before claiming closed-loop control. Training checkpoint preparation records upstream runtime gates explicitly.

Freeze a bounded pilot with `python -m embodiment.pilot --root "$RUN_VOLUME/pilot" --episodes 12`. Its manifest contains paired scene specifications and explicit collection commands; it is not a dataset. Collection receipts preserve failed attempts and accept only physically successful camera episodes. Existing qualifications remain historical evidence. Generate a common-revision command with `python -m embodiment.qualification_pair --root "$RUN_VOLUME/qualification"`; add `--execute` only in the qualified Isaac runtime on a clean committed revision.

New training plans use 50 frozen held-out scenes. `python -m embodiment.paired --baseline baseline.json --transfer transfer.json --output paired.json` computes deterministic 10,000-resample scene-paired intervals. The primary rule remains +20 percentage points at 100 B demonstrations with a 95% interval excluding zero. Legacy 20-scene reports remain readable and cannot satisfy that primary rule. Report-level validation does not replace auditing raw episode hashes. Without E5, any benefit is attributable to related-robot initialization, not exclusively cross-embodiment transfer.

Offline checks: `python -m unittest discover -s embodiment/tests -v`. Set `RELIC_GROOT_SOURCE` to enable pinned upstream reader, normalization, serialization, and projector checks. TorchCodec may need an isolated FFmpeg 7 library path; keep the system encoder's library environment separate. Export dated attribution evidence with `python -m embodiment.provenance --output /tmp/viper-provenance.json`.

Open gates: G0 registration, license review, dated attribution review and common-revision qualification; G1 physical pilot and upstream-reader qualification on the final revision; G2 rental quote and approved budget; G3 real A-to-B optimizer/save/reload/inference and closed-loop execution. M0 requires learned source control on held-out scenes; M1 requires matched target experiments. No paid compute or large model download is performed by these preparation commands.
