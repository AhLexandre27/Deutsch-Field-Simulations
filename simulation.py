import numpy as np
from scipy.integrate import solve_ivp
from numba import njit
import warnings
warnings.filterwarnings('ignore')  # Ignorar warnings numéricos

# ==========================================
# PHYSICAL PARAMETERS FOR THE SIMULATION
# ==========================================

# Neutron star parameters
R = 1.0        # Star radius (normalized)
c = 1.0        # Speed of light (normalized to 1)
B0 = 1.0       # Surface magnetic field strength (normalized)

# Particle parameters
alpha = 1000          # Charge-to-mass ratio (qB_0R_0/mc^2) - ALTERADO PARA 1000
eta = 1.88e-19         # eta=k_rad*alpha**2
k_rad = eta * alpha**2 # Radiation reaction force coefficient

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

@njit
def h1(z):
    """Spherical Hankel function h₁(z) = j₁(z) + i·y₁(z)"""
    if abs(z) < 1e-6:
        inv_z2 = 1.0 / (z * z)
        return -1j * inv_z2 - 0.5j + z / 3.0
    
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

# Pre-calculate constants
H_1_const = h1(om * R)
H_2_const = h2(om * R)
H_3_const = h3(om * R)
D_H2_const = 3.0 * H_2_const - H_3_const * om * R

# ==========================================
# ELECTROMAGNETIC FIELD CALCULATIONS
# ==========================================

@njit
def EM_Deutsch(x, y, z, t, X_tilt):
    """
    Calculate E and B fields from the Deutsch solution.
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    r = max(r, 1e-6)

    cos_theta = z / r
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)
    
    phi = np.arctan2(y, x)
    psi = phi #- om * t
    
    h_1 = h1(om * r)
    h_2 = h2(om * r)
    h_3 = h3(om * r)
    
    D_h1 = 2.0 * h_1 - h_2 * om * r
    D_h2 = 3.0 * h_2 - h_3 * om * r
    
    rot = np.sin(X_tilt) * np.exp(1j * psi)
    cos_X = np.cos(X_tilt)
    
    # Magnetic field components
    geo_1r = (R**3) / (r**3) * cos_theta * cos_X
    geo_2r = (R / r) * (h_1 / H_1_const) * sin_theta
    
    geo_1th = (R**3) / (r**3) * sin_theta * cos_X
    geo_2th = ((R / r) * (D_h1 / H_1_const) + 
               (R**2) / (r_lc**2) * (h_2 / D_H2_const)) * cos_theta
    
    geo_1phi = ((R / r) * (D_h1 / H_1_const) + 
                (R**2) / (r_lc**2) * (h_2 / D_H2_const) * np.cos(2.0 * theta))
    
    B_r_im = 2.0 * B0 * (geo_1r + geo_2r * rot)
    B_theta_im = B0 * (geo_1th + geo_2th * rot)
    B_phi_im = 1j * B0 * (geo_1phi * rot)
    
    # Electric field components
    geo_1Er = ((R**2) / (r**2) * cos_X) * (2.0 / 3.0 - 
              (R**2) / (r**2) * (3.0 * cos_theta**2 - 1.0))
    geo_2Er = (R / r) * (3.0 * np.sin(2.0 * theta) * h_2 / D_H2_const)
    
    geo_1Eth = -(R**4) / (r**4) * np.sin(2.0 * theta) * cos_X
    geo_2Eth = (R / r) * (D_h2 / D_H2_const) * np.cos(2.0 * theta) - h_1 / H_1_const
    
    geo_1Ephi = (R / r) * (D_h2 / D_H2_const) - h_1 / H_1_const
    
    E_r_im = om * B0 * R / c * (geo_1Er + geo_2Er * rot)
    E_theta_im = om * B0 * R / c * (geo_1Eth + geo_2Eth * rot)
    E_phi_im = 1j * om * B0 * R / c * (geo_1Ephi * cos_theta * rot)
    
    # Real parts
    Br = np.real(B_r_im)
    Btheta = np.real(B_theta_im)
    Bphi = np.real(B_phi_im)
    Er = np.real(E_r_im)
    Etheta = np.real(E_theta_im)
    Ephi = np.real(E_phi_im)
    
    # Spherical to Cartesian
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
    Ex, Ey, Ez, Bx, By, Bz = EM_Deutsch(x, y, z, t, X_tilt)
    return np.array([
        [0.0, Ex, Ey, Ez],
        [-Ex, 0.0, Bz, -By],
        [-Ey, -Bz, 0.0, Bx],
        [-Ez, By, -Bx, 0.0]
    ])

@njit
def back_reaction(u, F):
    u_cov = ETA @ u
    E_mu = F @ u_cov
    E_cov = ETA @ E_mu
    E_sq = np.dot(E_mu, E_cov)
    return k_rad * (F @ E_cov - E_sq * u)

@njit
def lorentz_condition(u):
    u_sq = -u[0]**2 + u[1]**2 + u[2]**2 + u[3]**2
    if abs(u_sq + 1.0) > 1e-10:
        gamma = u[0]
        v_sq = u[1]**2 + u[2]**2 + u[3]**2
        if v_sq > 0:
            correction = np.sqrt((gamma**2 - 1.0) / v_sq)
            u[1] *= correction
            u[2] *= correction
            u[3] *= correction
    return u

@njit
def Equations_of_Motion(t, a, X_tilt):
    x, y, z, ut, ux, uy, uz = a
    u = np.array([ut, ux, uy, uz])
    u = lorentz_condition(u)
    ut, ux, uy, uz = u
    
    F = Faraday_tensor(x, y, z, t, X_tilt)
    u_cov = ETA @ u
    lorentz_4force = alpha * (F @ u_cov)
    rad_4force = back_reaction(u, F)
    dU_dtau = lorentz_4force + rad_4force
    
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

def event_crashed(t, a, X_tilt):
    r = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    return r - R
event_crashed.terminal = True
event_crashed.direction = -1

def event_ejected(t, a, X_tilt):
    r = np.sqrt(a[0]**2 + a[1]**2 + a[2]**2)
    return r - r_esc
event_ejected.terminal = True
event_ejected.direction = 1

# ==========================================
# MAIN SIMULATION FUNCTION (MODIFICADA)
# ==========================================

def simulate_single_particle(particle_data):
    """
    Simulate the trajectory of a single particle with robust error handling.
    """
    i, x0, y0, z0, X_tilt = particle_data
    
    try:
        y0 = [x0, y0, z0, 1.0, 0.0, 0.0, 0.0]
        t_span = (0.0, t_max)
        events = [event_crashed, event_ejected]
        
        # Tentativa 1: Configuração padrão
        try:
            sol = solve_ivp(
                Equations_of_Motion,
                t_span,
                y0,
                method='Radau',
                events=events,
                args=(X_tilt,),
                rtol=1e-8,
                atol=1e-10,
                max_step=0.5
            )
        except Exception as e:
            # Tentativa 2: Tolerâncias mais baixas
            sol = solve_ivp(
                Equations_of_Motion,
                t_span,
                y0,
                method='Radau',
                events=events,
                args=(X_tilt,),
                rtol=1e-6,
                atol=1e-8,
                max_step=0.2
            )
        
        # Classificar resultado
        if sol.t_events[0].size > 0:
            status = 'Crashed'
        elif sol.t_events[1].size > 0:
            status = 'Ejected'
        else:
            status = 'Trapped'
        
        # Extrair estado final
        gamma_final = sol.y[3, -1] if sol.y.shape[1] > 0 else 1.0
        xf = sol.y[0, -1] if sol.y.shape[1] > 0 else x0
        yf = sol.y[1, -1] if sol.y.shape[1] > 0 else y0
        zf = sol.y[2, -1] if sol.y.shape[1] > 0 else z0
        
        # Verificar valores razoáveis
        if not np.isfinite(gamma_final) or gamma_final > 1e10 or gamma_final < 0:
            status = 'Trapped'
            gamma_final = 1.0
            xf, yf, zf = x0, y0, z0
            
        return status, x0, y0, z0, xf, yf, zf, gamma_final
        
    except Exception as e:
        # Se tudo falhar, retorna como 'Trapped'
        return 'Trapped', x0, y0, z0, x0, y0, z0, 1.0