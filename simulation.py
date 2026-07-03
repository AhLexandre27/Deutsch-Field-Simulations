import numpy as np
from scipy.integrate import solve_ivp
from numba import njit

# ==========================================
# PHYSICAL PARAMETERS FOR THE SIMULATION
# ==========================================

# Neutron star parameters
R = 1.0        # Star radius (normalized)
c = 1.0        # Speed of light (normalized to 1)
B0 = 1.0       # Surface magnetic field strength (normalized)

# Particle parameters
alpha = 10             # Charge-to-mass ratio (qB_0R_0/mc^2)
eta = 1.88e-19           # eta=k_rad*alpha**2
k_rad = eta * alpha**2   # Radiation reaction force coefficient

# Rotation parameters
om = 0.209               # Angular velocity of the star (normalized for P=0.001s)
r_lc = 1.0 / om          # Light cylinder radius
r_esc = 3.0 * r_lc       # Escape radius (we consider particle has escaped)

# Flat spacetime metric (Minkowski)
ETA = np.diag([-1.0, 1.0, 1.0, 1.0])

# Maximum simulation time (5 rotation periods)
t_max = 5.0 * 2.0 * np.pi / om

# ==========================================
# SPHERICAL BESSEL FUNCTIONS
# ==========================================
# These functions are required for the Deutsch solution:
# electromagnetic fields of a rotating magnetized sphere

@njit  # Decorator that compiles the function for faster execution
def h1(z):
    """Spherical Hankel function h₁(z) = j₁(z) + i·y₁(z)"""
    # For very small z, use series expansion to avoid division by zero
    if abs(z) < 1e-6:
        inv_z2 = 1.0 / (z * z)
        return -1j * inv_z2 - 0.5j + z / 3.0
    
    # Standard calculation for non-tiny z
    j1 = np.sin(z) / (z**2) - np.cos(z) / z
    y1 = -np.cos(z) / (z**2) - np.sin(z) / z
    return j1 + 1j * y1

@njit
def h2(z):
    """Spherical Hankel function h₂(z) = j₂(z) + i·y₂(z)"""
    if abs(z) < 1e-6:
        inv_z3 = 1.0 / (z * z * z)
        inv_z = 1.0 / z
        return -3j * inv_z3 - 0.5j * inv_z
    
    j2 = (3.0 / (z**3) - 1.0 / z) * np.sin(z) - (3.0 / (z**2)) * np.cos(z)
    y2 = -(3.0 / (z**3) - 1.0 / z) * np.cos(z) - (3.0 / (z**2)) * np.sin(z)
    return j2 + 1j * y2

@njit
def h3(z):
    """Spherical Hankel function h₃(z) = j₃(z) + i·y₃(z)"""
    if abs(z) < 1e-6:
        inv_z4 = 1.0 / (z * z * z * z)
        return -15j * inv_z4
    
    j3 = (15.0 / (z**4) - 6.0 / (z**2)) * np.sin(z) - (15.0 / (z**3) - 1.0 / z) * np.cos(z)
    y3 = -(15.0 / (z**4) - 6.0 / (z**2)) * np.cos(z) - (15.0 / (z**3) - 1.0 / z) * np.sin(z)
    return j3 + 1j * y3

# Pre-calculate constants that depend only on radius and frequency
# This avoids recalculating the same Bessel functions repeatedly
H_1_const = h1(om * R)    # h₁(ωR) - constant evaluated at the surface
H_2_const = h2(om * R)    # h₂(ωR)
H_3_const = h3(om * R)    # h₃(ωR)
D_H2_const = 3.0 * H_2_const - H_3_const * om * R  # Useful combination

# ==========================================
# ELECTROMAGNETIC FIELD CALCULATIONS
# ==========================================

@njit
def EM_Deutsch(x, y, z, t, X_tilt):
    """
    Calculate E and B fields from the Deutsch solution.
    
    Parameters:
    - x, y, z: position in space
    - t: cosmic time 
    - X_tilt: magnetic axis tilt angle
    
    Returns:
    - Ex, Ey, Ez: electric field components
    - Bx, By, Bz: magnetic field components
    """
    # Distance from star center
    r = np.sqrt(x**2 + y**2 + z**2)
    r = max(r, 1e-6)  # Prevent division by zero

    # Calculate theta angle (relative to z-axis)
    cos_theta = z / r
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    
    # Azimuthal angle phi
    phi = np.arctan2(y, x)
    
    # Time-dependent phase for the rotating magnetic field (cosmic time)
    psi = phi - om * t
    
    # Calculate Hankel functions at current position
    h_1 = h1(om * r)
    h_2 = h2(om * r)
    h_3 = h3(om * r)
    
    # Useful combinations (derivatives of Hankel functions)
    D_h1 = 2.0 * h_1 - h_2 * om * r
    D_h2 = 3.0 * h_2 - h_3 * om * r
    
    # Rotation term: contains time dependence and tilt angle
    rot = np.sin(X_tilt) * np.exp(1j * psi)
    cos_X = np.cos(X_tilt)
    
    # ===== MAGNETIC FIELD GEOMETRIC FACTORS =====
    
    # Radial component
    geo_1r = (R**3) / (r**3) * cos_theta * cos_X
    geo_2r = (R / r) * (h_1 / H_1_const) * sin_theta
    
    # Theta component
    geo_1th = (R**3) / (r**3) * sin_theta * cos_X
    geo_2th = ((R / r) * (D_h1 / H_1_const) + 
               (R**2) / (r_lc**2) * (h_2 / D_H2_const)) * cos_theta
    
    # Phi component
    geo_1phi = ((R / r) * (D_h1 / H_1_const) + 
                (R**2) / (r_lc**2) * (h_2 / D_H2_const) * np.cos(2.0 * theta))
    
    # Complex magnetic field components
    B_r_im = 2.0 * B0 * (geo_1r + geo_2r * rot)
    B_theta_im = B0 * (geo_1th + geo_2th * rot)
    B_phi_im = 1j * B0 * (geo_1phi * rot)
    
    # ===== ELECTRIC FIELD GEOMETRIC FACTORS =====
    
    geo_1Er = ((R**2) / (r**2) * cos_X) * (2.0 / 3.0 - 
              (R**2) / (r**2) * (3.0 * cos_theta**2 - 1.0))
    geo_2Er = (R / r) * (3.0 * np.sin(2.0 * theta) * h_2 / D_H2_const)
    
    geo_1Eth = -(R**4) / (r**4) * np.sin(2.0 * theta) * cos_X
    geo_2Eth = (R / r) * (D_h2 / D_H2_const) * np.cos(2.0 * theta) - h_1 / H_1_const
    
    geo_1Ephi = (R / r) * (D_h2 / D_H2_const) - h_1 / H_1_const
    
    # Complex electric field components
    E_r_im = om * B0 * R / c * (geo_1Er + geo_2Er * rot)
    E_theta_im = om * B0 * R / c * (geo_1Eth + geo_2Eth * rot)
    E_phi_im = 1j * om * B0 * R / c * (geo_1Ephi * cos_theta * rot)
    
    # Take the REAL part (physical fields are real-valued)
    Br = np.real(B_r_im)
    Btheta = np.real(B_theta_im)
    Bphi = np.real(B_phi_im)
    Er = np.real(E_r_im)
    Etheta = np.real(E_theta_im)
    Ephi = np.real(E_phi_im)
    
    # ===== CONVERSION: spherical → Cartesian coordinates =====
    cp_phi = np.cos(phi)
    sp_phi = np.sin(phi)
    
    Bx = Br * sin_theta * cp_phi + Btheta * cos_theta * cp_phi - Bphi * sp_phi
    By = Br * sin_theta * sp_phi + Btheta * cos_theta * sp_phi + Bphi * cp_phi
    Bz = Br * cos_theta - Btheta * sin_theta
    
    Ex = Er * sin_theta * cp_phi + Etheta * cos_theta * cp_phi - Ephi * sp_phi
    Ey = Er * sin_theta * sp_phi + Etheta * cos_theta * sp_phi + Ephi * cp_phi
    Ez = Er * cos_theta - Etheta * sin_theta
    
    return Ex, Ey, Ez, Bx, By, Bz

@njit
def Faraday_tensor(x, y, z, t, X_tilt):
    """
    Construct the electromagnetic field tensor F^{μν} at a given point and time.
    
    The electromagnetic tensor organizes E and B into a 4×4 matrix:
         [ 0    Ex   Ey   Ez ]
    F =  [-Ex   0    Bz  -By]
         [-Ey  -Bz   0    Bx]
         [-Ez   By  -Bx   0 ]
    
    This makes relativity easier: the Lorentz force becomes
    simply a matrix multiplication.
    """
    Ex, Ey, Ez, Bx, By, Bz = EM_Deutsch(x, y, z, t, X_tilt)
    return np.array([
        [0.0, Ex, Ey, Ez],
        [-Ex, 0.0, Bz, -By],
        [-Ey, -Bz, 0.0, Bx],
        [-Ez, By, -Bx, 0.0]
    ])

@njit
def back_reaction(u, F):
    """
    Calculate the radiation reaction 4-force (Landau-Lifshitz formula).
    This function calculates the force using the covariant (4-dimensional) version.
    """
    # Convert to covariant vector (lowered indices)
    u_cov = ETA @ u
    
    # Electric field in the particle's rest frame
    E_mu = F @ u_cov
    E_cov = ETA @ E_mu

    # Scalar product E² = E^μ E_μ
    E_sq = np.dot(E_mu, E_cov)
    
    # Radiation force: k_rad * (F·E_cov - E²·u)
    return k_rad * (F @ E_cov - E_sq * u)

@njit
def lorentz_condition(u):
    """
    Correct the 4-velocity to enforce u·u = -1.
    Numerical errors can violate this constraint, so we fix it.
    """
    # Calculate u·u = -ut² + ux² + uy² + uz²
    u_sq = -u[0]**2 + u[1]**2 + u[2]**2 + u[3]**2
    
    # If off the mass shell, correct it
    if abs(u_sq + 1.0) > 1e-10:
        gamma = u[0]  # Lorentz factor
        v_sq = u[1]**2 + u[2]**2 + u[3]**2  # Spatial velocity squared
        
        if v_sq > 0:
            # Correction factor to maintain the correct gamma
            correction = np.sqrt((gamma**2 - 1.0) / v_sq)
            u[1] *= correction
            u[2] *= correction
            u[3] *= correction
    
    return u

@njit
def Equations_of_Motion(t, a, X_tilt):
    """
    ODE SYSTEM: Equations of motion for the particle.
    
    This is the main function that gets numerically integrated.
    It returns the time derivatives of all variables.
    
    State y = [x, y, z, ut, ux, uy, uz] where:
    - x, y, z: particle position
    - ut, ux, uy, uz: 4-velocity (ut = gamma = Lorentz factor)
    
    The parameter X_tilt is passed through to the EM field calculations.
    """
    # Unpack the state vector
    x, y, z, ut, ux, uy, uz = a
    u = np.array([ut, ux, uy, uz])
    
    # STEP 1: Ensure the 4-velocity is coherent
    u = lorentz_condition(u)
    ut, ux, uy, uz = u
    
    # STEP 2: Calculate electromagnetic tensor at current position
    F = Faraday_tensor(x, y, z, t, X_tilt)

    # STEP 3: Calculate forces (***with respect to proper time***)
    
    # Lorentz force
    u_cov = ETA @ u
    lorentz_4force = alpha * (F @ u_cov)
    
    # Radiation reaction force
    rad_4force = back_reaction(u, F)

    # Total acceleration (***derivative with respect to proper time***)
    dU_dtau = lorentz_4force + rad_4force

    
    # STEP 4: Convert from proper time (τ) derivatives to coordinate time (t) derivatives
    # dx/dt = (dx/dτ) / (dt/dτ) = ux / ut
    inv_ut = 1.0 / ut
    
    dx_dt = ux * inv_ut
    dy_dt = uy * inv_ut
    dz_dt = uz * inv_ut
    
    dut_dt = dU_dtau[0] * inv_ut
    dux_dt = dU_dtau[1] * inv_ut
    duy_dt = dU_dtau[2] * inv_ut
    duz_dt = dU_dtau[3] * inv_ut
    
    return [dx_dt, dy_dt, dz_dt, dut_dt, dux_dt, duy_dt, duz_dt]

# ==========================================
# EVENT DETECTORS
# ==========================================
# These detect when the particle reaches certain conditions
# (hitting the star, escaping)

def event_crashed(t, a, X_tilt):
    """Stop when particle hits the star surface."""
    r = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2) #This "a" is the state vector [x, y, z, ut, ux, uy, uz]
    return r - R
event_crashed.terminal = True
event_crashed.direction = -1  # Only when approaching the star

def event_ejected(t, a, X_tilt):
    """Stop when particle escapes (spherical boundary)."""
    r = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    return r - r_esc
event_ejected.terminal = True
event_ejected.direction = 1  # Only when moving outward

# ==========================================
# MAIN SIMULATION FUNCTION
# ==========================================


def simulate_single_particle(particle_data):
    """
    Simulate the trajectory of a single particle.
    
    Parameters:
        particle_data: tuple (index, x0, y0, z0, X_tilt)
    
    Returns:
        status, x0, y0, z0, xf, yf, zf, gamma_final
    """
    # Unpack the input tuple
    i, x0, y0, z0, X_tilt = particle_data
    
    # BUILD THE INITIAL STATE VECTOR
    # [x, y, z, ut, ux, uy, uz] — particle starts at rest (ut=1, ux=uy=uz=0)
    y0 = [x0, y0, z0, 1.0, 0.0, 0.0, 0.0]
    
    # TIME INTERVAL for integration (t=0 to t=t_max)
    t_span = (0.0, t_max)
    
    # EVENTS that can stop the integration early
    events = [event_crashed, event_ejected]
    
    # NUMERICAL INTEGRATION
    sol = solve_ivp(
        Equations_of_Motion,   # ODE system
        t_span,                # (t_start, t_end)
        y0,                    # Initial state
        method='Radau',        # Implicit method for stiff systems
        events=events,         # Terminal events
        args=(X_tilt,),        # Extra argument for ODE and events
        rtol=1e-10,            # Relative tolerance
        atol=1e-12,            # Absolute tolerance
        max_step=np.inf        # No step size limit
    )
    
    # CLASSIFY THE PARTICLE'S FATE
    # sol.t_events[0] = times when event_crashed fired
    # sol.t_events[1] = times when event_ejected fired
    # An array with .size > 0 means the event triggered
    if sol.t_events[0].size > 0:      # Hit the star
        status = 'Crashed'
    elif sol.t_events[1].size > 0:    # Escaped
        status = 'Ejected'
    else:                             # Neither: still orbiting at t_max
        status = 'Trapped'
    
    # EXTRACT FINAL STATE
    # sol.y has shape (7, n_steps): [variable, time_step]
    # sol.y[3, -1] = final gamma (ut = Lorentz factor)
    # sol.y.shape[1] = number of time steps taken
    gamma_final = sol.y[3, -1] if sol.y.shape[1] > 0 else 1.0
    
    # Final positions (last value of x, y, z)
    xf = sol.y[0, -1] if sol.y.shape[1] > 0 else x0
    yf = sol.y[1, -1] if sol.y.shape[1] > 0 else y0
    zf = sol.y[2, -1] if sol.y.shape[1] > 0 else z0
    
    return status, x0, y0, z0, xf, yf, zf, gamma_final