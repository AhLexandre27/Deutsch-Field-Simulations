"""
MAIN EXECUTION SCRIPT - PARTICLE SIMULATION WITH TIME-DEPENDENT FIELDS

Features:
  - Saves results to CSV (one file per tilt angle, plus a combined summary)
  - Checkpointing: saves progress after each angle and after each batch
  - Configurable number of CPU cores
  - Resumable: if interrupted, re-running skips completed work
  - Progress bar via tqdm

Usage:
    python run_simulation.py --cores 6 --particles 1000 --angles 0 15 30 45 60 75 90
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

# ==========================================
# IMPORT SIMULATION MODULE
# ==========================================
from simulation import (
    simulate_single_particle, R, r_lc, t_max, om
)

# ==========================================
# CONFIGURATION
# ==========================================
OUTPUT_DIR = "simulation_output"
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
CSV_DIR = os.path.join(OUTPUT_DIR, "csv")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)


# ==========================================
# CHECKPOINT HELPERS
# ==========================================
def checkpoint_path(ang_deg, tag="main"):
    """Path for a checkpoint file for a given angle."""
    return os.path.join(CHECKPOINT_DIR, f"angle_{ang_deg}_{tag}.pkl")


def save_checkpoint(ang_deg, results, completed_indices, tag="main"):
    """Save current state of a simulation to a checkpoint file."""
    state = {
        "angle_deg": ang_deg,
        "results": results,
        "completed_indices": list(completed_indices),
        "timestamp": time.time(),
    }
    tmp = checkpoint_path(ang_deg, tag) + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(state, f)
    os.replace(tmp, checkpoint_path(ang_deg, tag))  # atomic write


def load_checkpoint(ang_deg, tag="main"):
    """Load a checkpoint if it exists. Returns None otherwise."""
    path = checkpoint_path(ang_deg, tag)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def clear_checkpoint(ang_deg, tag="main"):
    """Delete a checkpoint file."""
    path = checkpoint_path(ang_deg, tag)
    if os.path.exists(path):
        os.remove(path)


# ==========================================
# CSV OUTPUT
# ==========================================
def save_results_to_csv(ang_deg, results):
    """
    Save results for one angle to CSV files.

    Creates:
      - results_angle_{ang}.csv        : all particles, one row each
      - summary_angle_{ang}.json       : counts and gamma statistics
    """
    rows = []
    for status in ("Crashed", "Ejected", "Trapped"):
        n = len(results[status]["x"])
        for i in range(n):
            rows.append({
                "status": status,
                "x0": results[status]["x"][i],
                "y0": results[status]["y"][i],
                "z0": results[status]["z"][i],
                "xf": results[status]["xf"][i],
                "yf": results[status]["yf"][i],
                "zf": results[status]["zf"][i],
                "gamma": results[status]["gamas"][i],
                "angle_deg": ang_deg,
            })

    df = pd.DataFrame(rows)
    # Order columns nicely
    cols = ["angle_deg", "status", "x0", "y0", "z0",
            "xf", "yf", "zf", "gamma"]
    df = df[cols]
    csv_path = os.path.join(CSV_DIR, f"results_angle_{ang_deg}.csv")
    df.to_csv(csv_path, index=False)

    # Summary JSON
    summary = {"angle_deg": ang_deg, "counts": {}}
    for status in ("Crashed", "Ejected", "Trapped"):
        gammas = results[status]["gamas"]
        summary["counts"][status] = {
            "n": len(gammas),
            "gamma_min": float(min(gammas)) if gammas else None,
            "gamma_max": float(max(gammas)) if gammas else None,
            "gamma_mean": float(np.mean(gammas)) if gammas else None,
        }
    json_path = os.path.join(CSV_DIR, f"summary_angle_{ang_deg}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return csv_path, json_path


def save_combined_csv(angles_done):
    """Concatenate all per-angle CSVs into one master CSV."""
    dfs = []
    for ang in angles_done:
        path = os.path.join(CSV_DIR, f"results_angle_{ang}.csv")
        if os.path.exists(path):
            dfs.append(pd.read_csv(path))
    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        out = os.path.join(CSV_DIR, "results_all_angles.csv")
        combined.to_csv(out, index=False)
        print(f"[CSV] Combined file: {out}  ({len(combined)} rows)")


# ==========================================
# PARALLEL SIMULATION WITH CHECKPOINTING
# ==========================================
def run_simulation_parallel(angles_to_simulate, num_particles,
                            r0_min, r0_max, n_cores=6,
                            batch_size=50, resume=True):
    """
    Run particle simulations in parallel for multiple tilt angles.

    Parameters
    ----------
    angles_to_simulate : list of float
        Tilt angles in degrees.
    num_particles : int
        Particles per angle.
    r0_min, r0_max : float
        Initial radial shell (in units of R).
    n_cores : int
        Number of CPU cores.
    batch_size : int
        Number of particles per checkpoint write.
    resume : bool
        If True, resume from existing checkpoints.
    """
    print(f"Using {n_cores} cores for parallel processing")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Checkpoints: {CHECKPOINT_DIR}")
    print(f"CSV files:   {CSV_DIR}")

    angles_done = []

    for ang_deg in angles_to_simulate:
        loop_start = time.time()

        # Handle aligned case
        safe_ang = 0.0000001 if ang_deg == 0 else ang_deg
        X_tilt = float(np.radians(safe_ang))

        print(f"\n{'='*60}")
        print(f">>> Angle: {ang_deg}° (X_tilt = {X_tilt:.6f} rad)")
        print(f"{'='*60}")

        # ----------------------------------------
        # Try to load checkpoint
        # ----------------------------------------
        state = load_checkpoint(ang_deg) if resume else None
        if state is not None:
            results = state["results"]
            completed_indices = set(state["completed_indices"])
            print(f"[RESUME] Found checkpoint with "
                  f"{len(completed_indices)}/{num_particles} particles done.")
        else:
            results = {
                "Crashed": {"x": [], "y": [], "z": [],
                            "xf": [], "yf": [], "zf": [], "gamas": []},
                "Ejected": {"x": [], "y": [], "z": [],
                            "xf": [], "yf": [], "zf": [], "gamas": []},
                "Trapped": {"x": [], "y": [], "z": [],
                            "xf": [], "yf": [], "zf": [], "gamas": []},
            }
            completed_indices = set()

        # ----------------------------------------
        # Generate initial positions (deterministic seed)
        # ----------------------------------------
        # Use a seed that depends on the angle so different angles
        # get different (but reproducible) initial conditions.
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

        # ----------------------------------------
        # Build list of particles still needing simulation
        # ----------------------------------------
        remaining = [i for i in range(num_particles)
                     if i not in completed_indices]
        particle_data = [(i, x0_arr[i], y0_arr[i], z0_arr[i], X_tilt)
                         for i in remaining]

        print(f"[INFO] {len(particle_data)} particles remaining to simulate.")

        # ----------------------------------------
        # Run in batches, checkpointing after each batch
        # ----------------------------------------
        with ProcessPoolExecutor(max_workers=n_cores) as executor:
            pbar = tqdm(total=len(particle_data),
                        desc=f"Angle {ang_deg}°", unit="particle")

            # Process in chunks to allow checkpointing
            for batch_start in range(0, len(particle_data), batch_size):
                batch = particle_data[batch_start:batch_start + batch_size]

                futures = {
                    executor.submit(simulate_single_particle, p): p
                    for p in batch
                }

                for fut in as_completed(futures):
                    p = futures[fut]
                    idx = p[0]
                    try:
                        (status, x0, y0, z0,
                         xf, yf, zf, gamma_final) = fut.result()
                    except Exception as e:
                        # If a particle fails catastrophically,
                        # record it as Trapped at initial position.
                        status, x0, y0, z0 = "Trapped", p[1], p[2], p[3]
                        xf, yf, zf, gamma_final = x0, y0, z0, 1.0
                        print(f"[WARN] Particle {idx} failed: {e}")

                    results[status]["x"].append(x0)
                    results[status]["y"].append(y0)
                    results[status]["z"].append(z0)
                    results[status]["xf"].append(xf)
                    results[status]["yf"].append(yf)
                    results[status]["zf"].append(zf)
                    results[status]["gamas"].append(gamma_final)

                    completed_indices.add(idx)
                    pbar.update(1)

                # ---- Checkpoint after each batch ----
                save_checkpoint(ang_deg, results, completed_indices)
                pbar.set_postfix({
                    "ckpt": len(completed_indices),
                })

            pbar.close()

        # ----------------------------------------
        # Save final CSV for this angle
        # ----------------------------------------
        csv_path, json_path = save_results_to_csv(ang_deg, results)
        print(f"[CSV] Saved: {csv_path}")
        print(f"[JSON] Summary: {json_path}")

        # Clear checkpoint (work complete for this angle)
        clear_checkpoint(ang_deg)

        angles_done.append(ang_deg)

        # ----------------------------------------
        # Print summary
        # ----------------------------------------
        elapsed = (time.time() - loop_start) / 60
        n_c = len(results["Crashed"]["x"])
        n_e = len(results["Ejected"]["x"])
        n_t = len(results["Trapped"]["x"])
        print(f"  Crashed: {n_c} | Ejected: {n_e} | Trapped: {n_t} "
              f"| Time: {elapsed:.1f} min")

        for status, label in [("Ejected", "Ej"),
                              ("Crashed", "Cr"),
                              ("Trapped", "Tr")]:
            g = results[status]["gamas"]
            if g:
                print(f"  {label}: γ_min={min(g):.2f}, "
                      f"γ_max={max(g):.2f}, γ_mean={np.mean(g):.2f}")

    # ----------------------------------------
    # Save combined CSV across all angles
    # ----------------------------------------
    save_combined_csv(angles_done)


# ==========================================
# MAIN
# ==========================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Pulsar test-particle simulation (Deutsch field)"
    )
    p.add_argument("--cores", type=int, default=6,
                   help="Number of CPU cores to use.")
    p.add_argument("--particles", type=int, default=1000,
                   help="Number of particles per tilt angle.")
    p.add_argument("--angles", type=float, nargs="+",
                   default=[0, 15, 30, 45, 60, 75, 90],
                   help="Tilt angles in degrees.")
    p.add_argument("--r0-min-factor", type=float, default=1.1,
                   help="Minimum initial radius as multiple of R.")
    p.add_argument("--r0-max-factor", type=float, default=0.9,
                   help="Maximum initial radius as multiple of r_lc.")
    p.add_argument("--batch-size", type=int, default=50,
                   help="Particles per checkpoint write.")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore existing checkpoints and start fresh.")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    num_particles = args.particles
    r0_min = args.r0_min_factor * R
    r0_max = args.r0_max_factor * r_lc
    angles_to_simulate = args.angles

    print("=" * 60)
    print("TIME-DEPENDENT FIELD SIMULATION")
    print("psi = phi - om*t (corotating pattern)")
    print("=" * 60)
    print(f"Tilt angles: {angles_to_simulate}°")
    print(f"Particles per angle: {num_particles}")
    print(f"Initial radial range: {r0_min:.2f} to {r0_max:.2f} R")
    print(f"Max integration time: {t_max:.2f} "
          f"(≈ {t_max*om/(2*np.pi):.1f} periods)")
    print(f"Angular frequency: om = {om:.3f}")
    print(f"Light cylinder radius: r_lc = {r_lc:.2f}")
    print(f"Cores: {args.cores}")
    print(f"Batch size (checkpoint): {args.batch_size}")
    print(f"Resume: {not args.no_resume}")
    print("=" * 60)

    run_simulation_parallel(
        angles_to_simulate=angles_to_simulate,
        num_particles=num_particles,
        r0_min=r0_min,
        r0_max=r0_max,
        n_cores=args.cores,
        batch_size=args.batch_size,
        resume=not args.no_resume,
    )

    print("\n" + "=" * 60)
    print("SIMULATION COMPLETED!")
    print("=" * 60)