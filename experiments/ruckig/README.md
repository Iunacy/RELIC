# ruckig experiment

optional, isolated 3- and 4-dof trajectory benchmark using ruckig v0.19.4 at
`a8db97a4e9c55e5160a3855f739fa3b270df8e4c`, c++20 and mujoco 3.16.0.
production motion stays unchanged; the dependency is disabled by default.

run from the repository root:

```sh
sh experiments/ruckig/reproduce.sh
```

baseline tests: 31/31. optional tests: 34/34. the campaign recorded 104
comparisons, 96 actuator executions and 78 settled runs. failures are included.

at matched 1.9 s duration, nominal 3-dof command-end error fell from 19.72 mm
to 3.95 mm, but peak tracking error rose from 0.03316 to 0.05247 rad.
final settled accuracy was similar. see `results/` for metrics and timings;
wall-clock timings vary, while repeated physics reports matched byte-for-byte.

tiny moves can retain about 1e-8 rad of endpoint error. collision checks are
sampled, reference bounds do not bound measured joint dynamics, and leak
checking was unavailable. address and undefined-behavior sanitizer tests passed.
upstream provenance and license details are in `UPSTREAM.txt` and `vendor/`.
