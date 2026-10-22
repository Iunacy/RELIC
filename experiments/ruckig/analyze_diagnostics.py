#!/usr/bin/env python3
"""Validate and summarize diagnostics CSVs. stdlib only unless --plots is requested."""
import argparse
import csv
import gzip
import io
import hashlib
import math
from pathlib import Path


def read(path):
    return list(csv.DictReader(io.StringIO(trace_bytes(path).decode())))


def trace_bytes(path):
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_suffix(path.suffix+".gz").read_bytes())


def correlation(xs, ys):
    mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
    xx = sum((x-mx)**2 for x in xs)
    yy = sum((y-my)**2 for y in ys)
    return sum((x-mx)*(y-my) for x, y in zip(xs, ys))/math.sqrt(xx*yy) if xx*yy else 0.0


def analyze(root, baseline=None, plots=False):
    runs = read(root/"runs.csv")
    checks = read(root/"checks.csv")
    assert len(runs) == len(checks) == 24
    assert all(c["result"] == "PASS" for c in checks), "C++ diagnostic checks failed"
    summaries, peaks, digests = [], [], []
    plot_data = {}
    historical = read(baseline) if baseline else []
    for run in runs:
        n, case, method, repeat = int(run["dof"]), run["case"], run["method"], int(run["repeat"])
        key = f"{n}_{case}_{method}"
        path = root/f"{key}_{repeat}.csv"
        raw = trace_bytes(path)
        digests.append(dict(file=path.name, sha256=hashlib.sha256(raw).hexdigest()))
        if repeat:
            assert raw == trace_bytes(root/f"{key}_0.csv"), key+" repeat differs"
            continue
        rows = read(path)
        assert len(rows) == int(run["rows"])
        assert run["status"] == "Success" and run["settled"] == "1", key+" failed"
        groups = {}
        for r in rows:
            for col, val in r.items():
                if col not in ("stage", "phase", "joint_name"):
                    r[col] = float(val)
                    assert math.isfinite(r[col]), (key, col)
            groups.setdefault((int(r["tick"]), r["stage"]), []).append(r)
            k = r["tick"]
            expected_time = (k-(r["stage"] == "pre"))*.002
            assert abs(r["sim_time_s"]-expected_time) < 1e-10
            assert abs(r["reference_time_s"]-k*.002) < 1e-12
            assert abs(r["error_rad"]-(r["q_des_rad"]-r["q_actual_rad"])) < 1e-14
            assert r["ctrl_rad"] == r["q_des_rad"]
            assert abs(r["qfrc_actuator_Nm"]-r["actuator_force_Nm"]) < 1e-10
            assert abs(r["actuator_force_Nm"]-max(-20, min(20, r["servo_unclipped_Nm"]))) < 1e-10
            saturated = abs(r["actuator_force_Nm"]) >= 20-2e-7
            assert saturated == bool(r["force_saturated"])
            if not saturated:
                assert abs(r["error_rad"]-.1*r["v_actual_rad_s"]-r["actuator_force_Nm"]/200) < 1e-12
        last_tick = max(k for k, stage in groups)
        assert len(groups) == 1+2*last_tick
        assert [r["joint"] for r in groups[0, "initial"]] == list(range(n))
        previous, previous_v = groups[0, "initial"], [0.0]*n
        for k in range(1, last_tick+1):
            pre, post = groups[k, "pre"], groups[k, "post"]
            assert len(pre) == len(post) == n
            for j, (before, after, old) in enumerate(zip(pre, post, previous)):
                assert before["joint"] == after["joint"] == j
                assert before["q_actual_rad"] == old["q_actual_rad"]
                assert before["v_actual_rad_s"] == old["v_actual_rad_s"]
                assert before["q_des_rad"] == after["q_des_rad"]
                v = (after["q_des_rad"]-old["q_des_rad"])/.002
                a = (v-previous_v[j])/.002
                assert abs(v-after["v_des_fd_rad_s"]) < 1e-12
                assert abs(a-after["a_des_fd_rad_s2"]) < 1e-10
                previous_v[j] = v
            previous = post
        assert abs(max(abs(r["error_rad"]) for r in rows)-float(run["legacy_peak_rad"])) < 1e-14
        for old in historical:
            if old["dof"] == str(n) and old["case"] == case and old["method"] == method and case != "fast":
                for newcol, oldcol in [("legacy_peak_rad", "peak_tracking_error_rad"),
                                      ("command_end_cartesian_m", "command_end_cartesian_error_m")]:
                    assert abs(float(run[newcol])-float(old[oldcol])) < 1e-13, "historical baseline mismatch"
        post = [r for r in rows if r["stage"] == "post"]
        command = [r for r in post if r["phase"] == "command"]
        for j in range(n):
            joint_rows = [r for r in rows if r["joint"] == j]
            for stage in ("pre", "post"):
                candidates = [r for r in joint_rows if r["stage"] == stage]
                peak = max(candidates, key=lambda r: abs(r["error_rad"]))
                peaks.append(dict(dof=n, case=case, method=method, stage=stage, joint=j,
                                  **{c: peak[c] for c in ("joint_name", "phase", "sim_time_s", "reference_time_s",
                                    "error_rad", "v_des_fd_rad_s", "a_des_fd_rad_s2", "v_actual_rad_s", "actuator_force_Nm")}))
        yaw = [r for r in command if r["joint"] == 0]
        best = max(post, key=lambda r: abs(r["error_rad"]))
        velocity_term = [.1*r["v_actual_rad_s"] for r in yaw]
        residual = [r["error_rad"]-term for r, term in zip(yaw, velocity_term)]
        summaries.append(dict(dof=n, case=case, method=method, command_s=run["command_s"],
            command_end_mm=1000*float(run["command_end_cartesian_m"]),
            legacy_peak_rad=run["legacy_peak_rad"], aligned_post_peak_rad=abs(best["error_rad"]),
            peak_time_s=best["sim_time_s"], peak_joint=best["joint_name"],
            v_des_peak_rad_s=max(abs(r["v_des_fd_rad_s"]) for r in post),
            a_des_fd_peak_rad_s2=max(abs(r["a_des_fd_rad_s2"]) for r in post),
            torque_peak_Nm=max(abs(r["actuator_force_Nm"]) for r in rows),
            saturated_pre=sum(r["force_saturated"] for r in rows if r["stage"] == "pre"),
            saturated_post=sum(r["force_saturated"] for r in post),
            ctrl_at_limit=sum(r["ctrl_at_limit"] for r in rows),
            yaw_error_vdes_corr=correlation([r["error_rad"] for r in yaw], [r["v_des_fd_rad_s"] for r in yaw]) if yaw else 0,
            yaw_error_ades_corr=correlation([r["error_rad"] for r in yaw], [r["a_des_fd_rad_s2"] for r in yaw]) if yaw else 0,
            yaw_velocity_term_rms_rad=math.sqrt(sum(x*x for x in velocity_term)/len(yaw)) if yaw else 0,
            yaw_residual_rms_rad=math.sqrt(sum(x*x for x in residual)/len(yaw)) if yaw else 0))
        if n == 3 and case == "nominal":
            plot_data[method] = post
    for filename, data in [("comparison.csv", summaries), ("peaks.csv", peaks), ("sha256.csv", digests)]:
        with (root/filename).open("w", newline="") as f:
            writer = csv.DictWriter(f, list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    if plots:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        for method, data in plot_data.items():
            yaw = [r for r in data if r["joint"] == 0]
            ts = [r["sim_time_s"] for r in yaw]
            for ax, col in zip(axes, ("error_rad", "v_des_fd_rad_s", "a_des_fd_rad_s2", "actuator_force_Nm")):
                ax.plot(ts, [r[col] for r in yaw], label=method)
                ax.set_ylabel(col.replace("_", " "))
                ax.axvline(1.9, color="gray", linestyle="--", linewidth=.8)
                ax.grid(alpha=.3)
            axes[0].plot(ts, [.1*r["v_actual_rad_s"] for r in yaw], linestyle=":", label=method+" 0.1 × actual velocity")
        axes[0].legend(fontsize=8)
        axes[2].set_yscale("symlog", linthresh=1)
        axes[-1].set_xlabel("Simulation time (s); dashed line = command end")
        fig.suptitle("3-DOF nominal, yaw: post-step observations (acceleration uses finite differences)")
        fig.tight_layout()
        fig.savefig(root/"nominal.png", dpi=150)
        plt.close(fig)
    print(f"Validated {len(runs)} runs; wrote comparison, per-joint peaks and SHA256 CSVs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--plots", action="store_true")
    parser.add_argument("--compress", action="store_true", help="losslessly gzip per-run CSVs after validation")
    args = parser.parse_args()
    analyze(args.output, args.baseline, args.plots)

    if args.compress:
        for run in read(args.output/"runs.csv"):
            path = args.output/("{dof}_{case}_{method}_{repeat}.csv".format(**run))
            if path.exists():
                path.with_suffix(".csv.gz").write_bytes(gzip.compress(path.read_bytes(), mtime=0))
                path.unlink()
