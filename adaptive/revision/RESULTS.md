# Adaptive-v0 validation results

Learned correction improves substantially over conventional control, but the stronger analytical PI baseline has lower tracking RMSE. No learned-controller advantage over that baseline is claimed. The full-range PI study is exploratory: it was separately registered after one predeclared severe-case demo; all settings were selected using development data only. The original primary comparison is retained unchanged.

## Collection and split

Collected 1024 teacher scenarios across 64 independent base families and 128 embodied trajectory fingerprints: 512 episodes per embodiment, 128 per condition. Collection took 1602.10 s (0.6392 teacher scenarios/s), including proposal replays, verification replays and concurrent development/testing work. There were 3,068 accepted proposals and four rejected proposals out of 3,072; all were retained. The campaign comprises 7,164 actual native executions. Category/severity rates and exact physical coverage are in teacher_summary.json.

Eight of the 64 collected families were reserved for validation; 56 contributed fitting examples. Preparation produced 1,409,328 causal feature/target rows and rechecked every saved acceptance decision and verification hash. Four-reader preparation took 279.14 s. Serial preparation was interrupted for this performance revision; its log and prepared arrays were retained and checked for exact equality.

The final evaluation has 40 new base families, 80 embodied trajectory fingerprints, both embodiments in every partition, and 640 conditions. Primary aggregates use 560 conditions, excluding the duplicated restored_nominal condition. Thirteen full-cohort methods plus a four-joint-only common-joint transfer probe produced 8,640 executions; the extra PI comparison adds 640. Bootstrap intervals use complete base-family clusters, 4,000 replicates and a fixed seed. No final runs were incomplete; no continuous pairs were omitted. There are no cross-partition trajectory fingerprints or shared seeds. The nearest audited endpoint max-norm distance is 0.090419 rad (near-duplicate rule: below .05 rad and duration difference below .1 s).

restored_nominal is exactly nominal physics after a fresh reset. All 128 teacher nominal/restored pairs have identical physics and metrics. It is not evidence of degradation-then-recovery. Historical recovery-labelled records and the original 64-episode cohort remain unchanged as development evidence.

## Closed-loop primary results

- conventional: RMSE 0.0301632 (95% CI 0.0274568–0.0326611) rad; success 482/560; final Cartesian error 3.66666 (95% CI 3.50257–3.84153) mm.
- analytical: RMSE 0.00651319 (95% CI 0.00565992–0.00745618) rad; success 483/560; final Cartesian error 3.6686 (95% CI 3.50292–3.84609) mm.
- gravity: RMSE 0.00535377 (95% CI 0.00450492–0.00629369) rad; success 534/560; final Cartesian error 0.9737 (95% CI 0.821891–1.13094) mm.
- integral: RMSE 0.00339775 (95% CI 0.00284032–0.00401187) rad; success 512/560; final Cartesian error 1.22086 (95% CI 0.981988–1.47093) mm.
- shared_matched: RMSE 0.00344419 (95% CI 0.00282073–0.00412215) rad; success 560/560; final Cartesian error 1.21391 (95% CI 1.18711–1.24203) mm.
- per_matched: RMSE 0.00335841 (95% CI 0.00270041–0.00407734) rad; success 551/560; final Cartesian error 0.339741 (95% CI 0.323365–0.35655) mm.
- shared_full: RMSE 0.00245727 (95% CI 0.00190177–0.003056) rad; success 560/560; final Cartesian error 0.115162 (95% CI 0.108845–0.121206) mm.
- per_full: RMSE 0.00242821 (95% CI 0.00187271–0.00302559) rad; success 560/560; final Cartesian error 0.0803472 (95% CI 0.0742243–0.0866248) mm.
- full-range PI (exploratory): RMSE 0.00222636 (95% CI 0.00170953–0.00279483) rad; success 560/560; final Cartesian error 0.142742 (95% CI 0.120223–0.16511) mm.

The paired PI-minus-shared RMSE difference is -0.000230906 (95% CI -0.000400935–-7.38675e-05) rad. Both succeed on every primary condition. PI also has lower mean tracking error than the full expert bank. The learned model's slightly smaller final Cartesian error does not establish an overall learned advantage. Peak and command-end Cartesian errors, per-condition outcomes and paired intervals are retained in comparison.json and analytical_robustness.json.

## Generalization and model comparisons

- partition:combination_test: conventional/shared/full-range-PI RMSE 0.0280038/0.00188976/0.00165457 rad; successes 140/160, 160/160, 160/160. Partition aggregates include the nominal duplicate; their full intervals are in the JSON reports.
- partition:severity_test: conventional/shared/full-range-PI RMSE 0.0305376/0.00391035/0.00387717 rad; successes 125/160, 160/160, 160/160. Partition aggregates include the nominal duplicate; their full intervals are in the JSON reports.

Matched training uses 60,000 rows per embodiment, 960 shared updates versus 480 per expert, and exactly the same actual minibatch exposures. The bank has 3,266 parameters versus 1,633 shared. Full-data training uses the same union of fitting examples for 24 epochs (14,592 shared updates and 14,592 total expert updates). These are separate comparisons. model_bank_comparison.json contains directly paired shared-minus-bank intervals, including separate embodiment results. No zero-shot transfer claim follows from shared training on both embodiments.

- matched: shared-minus-bank tracking RMSE 8.57823e-05 (95% CI 6.40485e-06–0.000161876) rad; success-rate difference 0.0160714 (95% CI 0.00714286–0.0267857).
- full: shared-minus-bank tracking RMSE 2.90561e-05 (95% CI -9.95161e-06–7.39958e-05) rad; success-rate difference 0 (95% CI 0–0).

Learning curves hold eight validation families fixed. Fitting-family counts are 7, 14, 28 and 56; total model-selection family exposure is 15, 22, 36 and 64.

- 7 fitting families: RMSE 0.00339636 (95% CI 0.0026824–0.00413506) rad; success 558/560.
- 14 fitting families: RMSE 0.00278613 (95% CI 0.00215856–0.00346391) rad; success 560/560.
- 28 fitting families: RMSE 0.00268186 (95% CI 0.00212136–0.00328518) rad; success 560/560.
- 56 fitting families: RMSE 0.00245727 (95% CI 0.00190177–0.003056) rad; success 560/560.

The explicit 3→4 probe predicts only joints 0–2 and leaves the wrist conventional. Common-joint RMSE is 0.0311336 (95% CI 0.0289315–0.0332847) rad; full-arm success is 0/280 primary conditions (0/320 including restored_nominal). This is a transfer failure. The reverse direction was not tested.

Zero and copy-previous execution traces are byte-identical to conventional control on every condition. For the full shared model, normalized validation MSE is 0.000052 versus zero 0.054658 and copy-previous 0.014154. All raw rad² MSEs, per-embodiment comparisons, model hashes, fitting counts, curves and training times are in training_results.json. Offline MSE is not used as a claim of closed-loop superiority.

## ILC convergence and safety

The original two-family convergence probe is retained. The expanded replication has 32 training families and 64 paired conditions per lead setting, eight fixed iterations, no evaluation labels and no use of extra iterations in model fitting. At iteration eight, zero-lead mean proposed RMSE is 0.00127717 rad and 40-ms lead is 0.00110987 rad. The paired lead-minus-zero difference is -0.000167297 rad (95% CI -0.000293910 to -0.0000711579). All 128 iteration-eight runs complete successfully. At that iteration, zero lead rejects six proposals and lead rejects ten; reasons include final-error regression and insufficient improvement. Proposal magnitude saturation is approximately 0.99%/0.96% and rate saturation 1.21%/1.17%. Every iteration, rejection and saturation count is retained in convergence_comparison.json. Accepted-best monotonicity follows the gate and is not a convergence proof.

All primary and extra-PI evaluation runs complete; no fallback, collision or invalid-command episodes were observed. Qualification includes 16 injected invalid-output runs at nonzero correction, all withdrawing to zero through validated rate-limited commands. Unit tests cover both correction signs, bad dimensions/nonfinite values, infeasible withdrawal and the distinct stop reason. Unsafe observed motion uses common stopping rules for every controller. These are sampled collision checks, not continuous guarantees.

## Tests, costs and preservation

The initial 34-test suite passed. The final release suite passes 35/35 (6.70 s); five adaptive-related checks pass under AddressSanitizer/UBSan (32.24 s). Eighty final controller executions reproduce byte-for-byte, and 16 historical conventional executions reproduce byte-for-byte with unchanged metrics. All 17 protected historical evidence/fixture files and the Ruckig status/diff hashes are unchanged. No existing test was removed or tolerance weakened.

Nine CPU model fits take approximately 147 s in total, excluding data preparation/import overhead. Inference cost and total execution wall time are measured in comparison.json/evaluation_timing.json; these are loaded-workstation measurements, not an isolated latency guarantee. Bulk datasets, builds and run directories remain ignored; only small models and reports are versioned. Native executable identity and the compiled dirty-base commit are recorded in run_inventory.json, rather than presenting the pinned executable as a later clean commit.

## Reproduction and limitations

Full campaign: `bash adaptive/revise.sh`. Quick actual-motion demonstration after building: `python adaptive/revision_demo.py --fixtures`. The model is a small committed text-weight MLP; the inference demo needs only the native executable and Python standard library. It uses the predeclared first severity-test family, four-joint combined-fault condition.

Exact commands and source changes are recorded in COMMANDS.md and the Git history. The run directory retains all proposal/repeat traces, rejected outcomes, the two-epoch integration probe (including its NonConvergence outcomes), the interrupted serial preparation log, primary comparisons and exploratory PI tuning. No unsuccessful run was silently removed.

Limitations: two synthetic arms, one training seed, finite family sample, no hardware, no continuous collision guarantee, no true recovery experiment, and failed one-direction common-joint transfer. Family intervals do not include training-seed uncertainty or multiplicity correction. Some numerical degradation ranges overlap even though the severity factor is held out; the combination test holds out parameter values, not new fault mechanisms. The research harness generates/executes quintic joint trajectories independently of production execute_cartesian_motion and uses its own long-form recorder rather than TraceWriter; production IK/planner selection and dwell semantics are not validated by these results. This recorder integration remains unfinished and is explicitly documented in PROTOCOL.md.

Recommended next action: review the stronger analytical PI result and the failed transfer before increasing model complexity or collecting more data. Keep the production controller unchanged. Any production integration or changed learning target should receive architectural review; the current results do not justify promoting the learned controller over the strongest tested analytical feedback. No Isaac Lab, Cosmos, GR00T or NVIDIA/Nebius integration files were changed.
