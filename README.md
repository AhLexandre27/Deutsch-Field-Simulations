## What it does
- Computes the time-dependent Deutsch field solution (rotating dipole fields)
- Integrates particle trajectories using the relativistic Lorentz force plus radiation reaction
- Classifies each particle as Crashed (hit the star), Ejected (escaped), or Trapped (still orbiting)
- Runs batches of particles in parallel across multiple tilt angles

## Key parameters

- **alpha** — charge-to-mass ratio (qBR_0/mc^2). Controls how strongly particles accelerate (You can artificially choose other numbers e.g 1,100,...)
- **om** — stellar rotation frequency. Sets the light cylinder radius (r_lc = 1/om)
- **X_tilt** — angle between magnetic and rotation axes (0 = aligned, 90 = orthogonal)

## Files

- `simulation.py` — all the physics: fields, equations of motion, integration
- `run_code.ipynb` — runs the simulation: set parameters, choose angles, execute and plots

## How to run

1. Open `run_code.ipynb`
2. Set your parameters:
   - `num_particles` — how many per angle
   - `angles_to_simulate` — list of tilt angles in degrees
3. Run all cells
4. Results are saved as pickle files (`results_angle_*.pkl`)
