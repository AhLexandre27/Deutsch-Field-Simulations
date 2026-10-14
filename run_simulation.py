"""
Parallel run with checkpointing, CSV output, and per-alpha folders.

Output layout:
    simulation_output_<alpha_tag>/
        checkpoints/
            angle_<X>_main.pkl
        csv/
            results_angle_<X>.csv
            summary_angle_<X>.json
            results_all_angles.csv
"""

import numpy as np
import pandas as pd
import os
import json
import time
import argparse
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# Flat import — simulation.py lives in the same folder
from simulation import (
    simulate_single_particle,
    R, r_lc, t_max, om,
    alpha as ALPHA_RUN,
    ALPHA_TAG,
)

# ==========================================
# OUTPUT DIRECTORIES (per-alpha)
# ==========================================
OUTPUT_DIR     = f"simulation_output_{ALPHA_TAG}"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
CSV_DIR        = os.path.join(OUTPUT_DIR, "csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


# ==========================================
# CHECKPOINT HELPERS
# ==========================================
def checkpoint_path(ang_deg, tag="main"):
    return os.path.join(CHECKPOINT_DIR, f"angle_{ang_deg}_{tag}.pkl")


def save_checkpoint(ang_deg, results, completed_indices):
    state = {
        "angle_deg": ang_deg,
        "alpha": ALPHA_RUN,
        "alpha_tag": ALPHA_TAG,
        "results": results,
        "completed_indices": list(completed_indices),
        "timestamp": time.time(),
    }
    path = checkpoint_path(ang_deg)
    tmp  = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, path)          # atomic on POSIX and Windows


def load_checkpoint(ang_deg):
    path = checkpoint_path(ang_deg)
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        state = pickle.load(f)
    # Safety: refuse to resume from a checkpoint belonging to a different alpha
    if state.get("alpha") != ALPHA_RUN:
        print(f"[WARN] Checkpoint alpha mismatch "
              f"({state.get('alpha')} vs {ALPHA_RUN}). Ignoring.")
        return None
    return state


def clear_checkpoint(ang_deg):
    path = checkpoint_path(ang_deg)
    if os.path.exists(path):
        os.remove(path)


# ==========================================
# CSV OUTPUT
# ==========================================
def save_results_to_csv(ang_deg, results):
    """Write one CSV per angle, with an `alpha` column on every row."""
    rows = []
    for status in ("Crashed", "Ejected", "Trapped"):
        n = len(results[status]["x"])
        for i in range(n):
            rows.append({
                "alpha":     ALPHA_RUN,
                "angle_deg": ang_deg,
                "status":    status,
                "x0":        results[status]["x"][i],
                "y0":        results[status]["y"][i],
                "z0":        results[status]["z"][i],
                "xf":        results[status]["xf"][i],
                "yf":        results[status]["yf"][i],
                "zf":        results[status]["zf"][i],
                "gamma":     results[status]["gamas"][i],
            })
    df = pd.DataFrame(rows)
    df = df[["alpha", "angle_deg", "status",
             "x0", "y0", "z0", "xf", "yf", "zf", "gamma"]]
    csv_path = os.path.join(CSV_DIR, f"results_angle_{ang_deg}.csv")
    df.to_csv(csv_path, index=False)

    summary = {"alpha": ALPHA_RUN, "angle_deg": ang_deg, "counts": {}}
    for status in ("Crashed", "Ejected", "Trapped"):
        g = results[status]["gamas"]
        summary["counts"][status] = {
            "n":          len(g),
            "gamma_min":  float(min(g)) if g else None,
            "gamma_max":  float(max(g)) if g else None,
            "gamma_mean": float(np.mean(g)) if g else None,
        }
    json_path = os.path.join(CSV_DIR, f"summary_angle_{ang_deg}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path


def save_combined_csv(angles_done):
    dfs = []
    for ang in angles_done:
        p = os.path.join(CSV_DIR, f"results_angle_{ang}.csv")
        if os.path.exists(p):
            dfs.append(pd.read_csv(p))
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out = os.path.join(CSV_DIR, "results_all_angles.csv")
        combined.to_csv(out, index=False)
        print(f"[CSV] Combined: {out}  ({len(combined)} rows)")


# ==========================================
# PARALLEL SIMULATION
# ==========================================
def run_simulation_parallel(angles_to_simulate, num_particles,
                            r0_min, r0_max, n_cores=6,
                            batch_size=50, resume=True):
    print(f"Alpha = {ALPHA_RUN}  (tag = {ALPHA_TAG})")
    print(f"Output dir : {OUTPUT_DIR}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"CSV files  : {CSV_DIR}")
    print(f"Cores      : {n_cores}")

    angles_done = []

    for ang_deg in angles_to_simulate:
        loop_start = time.time()

        safe_ang = 0.0000001 if ang_deg == 0 else ang_deg
        X_tilt = float(np.radians(safe_ang))

        print(f"\n{'='*60}")
        print(f">>> Angle: {ang_deg}°  (X_tilt = {X_tilt:.6f} rad)")
        print(f"{'='*60}")

        # ---- Load or init ----
        state = load_checkpoint(ang_deg) if resume else None
        if state is not None:
            results = state["results"]
            completed_indices = set(state["completed_indices"])
            print(f"[RESUME] {len(completed_indices)}/{num_particles} particles done")
        else:
            results = {s: {"x": [], "y": [], "z": [],
                           "xf": [], "yf": [], "zf": [], "gamas": []}
                       for s in ("Crashed", "Ejected", "Trapped")}
            completed_indices = set()

        # ---- Initial positions (deterministic per angle) ----
        seed = 42 + int(ang_deg * 1000)
        rng = np.random.default_rng(seed)

        u_r = rng.uniform(0, 1, num_particles)
        r_uniform = (u_r * (r0_max**3 - r0_min**3) + r0_min**3) ** (1/3)

        v_theta = rng.uniform(-1, 1, num_particles)
        theta_uniform = np.arccos(v_theta)
        phi_uniform = rng.uniform(0, 2*np.pi, num_particles)

        x0_arr = r_uniform * np.sin(theta_uniform) * np.cos(phi_uniform)
        y0_arr = r_uniform * np.sin(theta_uniform) * np.sin(phi_uniform)
        z0_arr = r_uniform * np.cos(theta_uniform)

        remaining = [i for i in range(num_particles) if i not in completed_indices]
        particle_data = [(i, x0_arr[i], y0_arr[i], z0_arr[i], X_tilt)
                         for i in remaining]
        print(f"[INFO] {len(particle_data)} particles remaining")

        # ---- Parallel loop with per-batch checkpoints ----
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            pbar = tqdm(total=len(particle_data),
                        desc=f"Angle {ang_deg}°", unit="p")
            for bstart in range(0, len(particle_data), batch_size):
                batch = particle_data[bstart:bstart + batch_size]
                futures = {executor.submit(simulate_single_particle, p): p
                           for p in batch}
                for fut in as_completed(futures):
                    p = futures[fut]
                    idx = p[0]
                    try:
                        (status, x0, y0, z0, xf, yf, zf, g) = fut.result()
                    except Exception as e:
                        status, x0, y0, z0 = "Trapped", p[1], p[2], p[3]
                        xf, yf, zf, g = x0, y0, z0, 1.0
                        print(f"[WARN] Particle {idx}: {e}")

                    results[status]["x"].append(x0)
                    results[status]["y"].append(y0)
                    results[status]["z"].append(z0)
                    results[status]["xf"].append(xf)
                    results[status]["yf"].append(yf)
                    results[status]["zf"].append(zf)
                    results[status]["gamas"].append(g)
                    completed_indices.add(idx)
                    pbar.update(1)

                save_checkpoint(ang_deg, results, completed_indices)

            pbar.close()

        # ---- Write CSV and clear checkpoint ----
        csv_path, _ = save_results_to_csv(ang_deg, results)
        clear_checkpoint(ang_deg)
        angles_done.append(ang_deg)

        elapsed = (time.time() - loop_start) / 60
        n_c = len(results["Crashed"]["x"])
        n_e = len(results["Ejected"]["x"])
        n_t = len(results["Trapped"]["x"])
        print(f"  Crashed: {n_c} | Ejected: {n_e} | Trapped: {n_t} "
              f"| Time: {elapsed:.1f} min")
        print(f"  CSV: {csv_path}")
        for status, lab in [("Ejected", "Ej"),
                            ("Crashed", "Cr"),
                            ("Trapped", "Tr")]:
            g = results[status]["gamas"]
            if g:
                print(f"  {lab}: γ_min={min(g):.2f}, "
                      f"γ_max={max(g):.2f}, γ_mean={np.mean(g):.2f}")

    save_combined_csv(angles_done)


# ==========================================
# CLI
# ==========================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cores",     type=int, default=6)
    p.add_argument("--particles", type=int, default=1000)
    p.add_argument("--angles",    type=float, nargs="+",
                   default=[0, 15, 30, 45, 60, 75,
                            90, 105, 120, 135, 150, 165, 180])
    p.add_argument("--r0-min-factor", type=float, default=1.1)
    p.add_argument("--r0-max-factor", type=float, default=0.9)
    p.add_argument("--batch-size",    type=int, default=50)
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_simulation_parallel(
        angles_to_simulate=args.angles,
        num_particles=args.particles,
        r0_min=args.r0_min_factor * R,
        r0_max=args.r0_max_factor * r_lc,
        n_cores=args.cores,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )