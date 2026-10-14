# simulation.py
import numpy as np
from numba import njit
import warnings
warnings.filterwarnings('ignore')
# ==========================================
# PHYSICAL PARAMETERS
# ==========================================
R      = 1.0
c      = 1.0
B0     = 1.0

alpha  = 100000              # ← change this to change the run
eta    = 1.88e-19
k_rad  = eta * alpha**2

om     = 0.209
r_lc   = 1.0 / om
r_esc  = 3.0 * r_lc

ETA    = np.diag([-1.0, 1.0, 1.0, 1.0])

t_max  = 5.0 * 2.0 * np.pi / om

# HC pusher control
HC_SAFETY = 0.10
DT_MAX    = 5.0e-3
DT_MIN    = 1.0e-6

# Status codes
STATUS_TRAPPED = 0
STATUS_CRASHED = 1
STATUS_EJECTED = 2
STATUS_MAP = {0: "Trapped", 1: "Crashed", 2: "Ejected"}

# ==========================================
# RUN IDENTIFIER (used by run_simulation.py)
# ==========================================
def alpha_tag(a=None):
    """Return a filesystem-safe string uniquely identifying the alpha value.

    Examples:
        alpha = 1000.0    -> 'alpha1e3'
        alpha = 10000.0   -> 'alpha1e4'
        alpha = 2.5e5     -> 'alpha2.5e5'
    """
    if a is None:
        a = alpha
    # Scientific notation, strip trailing zeros in mantissa
    s = f"{a:.6e}"          # e.g. '1.000000e+04'
    mant, exp = s.split("e")
    mant = mant.rstrip("0").rstrip(".")
    exp  = int(exp)
    return f"alpha{mant}e{exp}"

ALPHA_TAG = alpha_tag()   # e.g. 'alpha1e4'

# ==========================================
# SPHERICAL HANKEL FUNCTIONS
# ==========================================
@njit(cache=True)
def h1(z):
    if abs(z) < 1e-6:
        return -1j / (z * z) - 0.5j + z / 3.0
    j1 = np.sin(z) / (z**2) - np.cos(z) / z
    y1 = -np.cos(z) / (z**2) - np.sin(z) / z
    return j1 + 1j * y1

@njit(cache=True)
def h2(z):
    if abs(z) < 1e-6:
        return -3j / (z**3) - 0.5j / z
    j2 = (3.0 / (z**3) - 1.0 / z) * np.sin(z) - (3.0 / (z**2)) * np.cos(z)
    y2 = -(3.0 / (z**3) - 1.0 / z) * np.cos(z) - (3.0 / (z**2)) * np.sin(z)
    return j2 + 1j * y2

@njit(cache=True)
def h3(z):
    if abs(z) < 1e-6:
        return -15j / (z**4)
    j3 = (15.0 / (z**4) - 6.0 / (z**2)) * np.sin(z) - (15.0 / (z**3) - 1.0 / z) * np.cos(z)
    y3 = -(15.0 / (z**4) - 6.0 / (z**2)) * np.cos(z) - (15.0 / (z**3) - 1.0 / z) * np.sin(z)
    return j3 + 1j * y3

H_1_const    = h1(om * R)
H_2_const    = h2(om * R)
H_3_const    = h3(om * R)
D_H2_const   = 3.0 * H_2_const - H_3_const * om * R


# ==========================================
# DEUTSCH FIELD (frozen snapshot, psi = phi)
# ==========================================
@njit(cache=True)
def EM_Deutsch(x, y, z, t, X_tilt):
    r = np.sqrt(x*x + y*y + z*z)
    if r < 1e-6:
        r = 1e-6

    cos_theta = z / r
    if cos_theta > 1.0:
        cos_theta = 1.0
    elif cos_theta < -1.0:
        cos_theta = -1.0
    theta = np.arccos(cos_theta)
    sin_theta = np.sin(theta)

    phi = np.arctan2(y, x)
    psi = phi                       # frozen snapshot

    h_1 = h1(om * r)
    h_2 = h2(om * r)
    h_3 = h3(om * r)

    D_h1 = 2.0 * h_1 - h_2 * om * r
    D_h2 = 3.0 * h_2 - h_3 * om * r

    rot   = np.sin(X_tilt) * np.exp(1j * psi)
    cos_X = np.cos(X_tilt)

    # ---- Magnetic field ----
    geo_1r  = (R**3) / (r**3) * cos_theta * cos_X
    geo_2r  = (R / r) * (h_1 / H_1_const) * sin_theta

    geo_1th = (R**3) / (r**3) * sin_theta * cos_X
    geo_2th = ((R / r) * (D_h1 / H_1_const) +
               (R**2) / (r_lc**2) * (h_2 / D_H2_const)) * cos_theta

    geo_1phi = ((R / r) * (D_h1 / H_1_const) +
                (R**2) / (r_lc**2) * (h_2 / D_H2_const) * np.cos(2.0 * theta))

    B_r_im     = 2.0 * B0 * (geo_1r + geo_2r * rot)
    B_theta_im = B0 * (geo_1th + geo_2th * rot)
    B_phi_im   = 1j * B0 * (geo_1phi * rot)

    # ---- Electric field ----
    geo_1Er = ((R**2) / (r**2) * cos_X) * (2.0/3.0 -
              (R**2) / (r**2) * (3.0 * cos_theta**2 - 1.0))
    geo_2Er = (R / r) * (3.0 * np.sin(2.0 * theta) * h_2 / D_H2_const)

    geo_1Eth = -(R**4) / (r**4) * np.sin(2.0 * theta) * cos_X
    geo_2Eth = (R / r) * (D_h2 / D_H2_const) * np.cos(2.0 * theta) - h_1 / H_1_const

    geo_1Ephi = (R / r) * (D_h2 / D_H2_const) - h_1 / H_1_const

    E_r_im     = om * B0 * R / c * (geo_1Er + geo_2Er * rot)
    E_theta_im = om * B0 * R / c * (geo_1Eth + geo_2Eth * rot)
    E_phi_im   = 1j * om * B0 * R / c * (geo_1Ephi * cos_theta * rot)

    Br = np.real(B_r_im);  Btheta = np.real(B_theta_im);  Bphi = np.real(B_phi_im)
    Er = np.real(E_r_im);  Etheta = np.real(E_theta_im);  Ephi = np.real(E_phi_im)

    cp = np.cos(phi)
    sp = np.sin(phi)

    Bx = Br * sin_theta * cp + Btheta * cos_theta * cp - Bphi * sp
    By = Br * sin_theta * sp + Btheta * cos_theta * sp + Bphi * cp
    Bz = Br * cos_theta - Btheta * sin_theta

    Ex = Er * sin_theta * cp + Etheta * cos_theta * cp - Ephi * sp
    Ey = Er * sin_theta * sp + Etheta * cos_theta * sp + Ephi * cp
    Ez = Er * cos_theta - Etheta * sin_theta

    return Ex, Ey, Ez, Bx, By, Bz


@njit(cache=True)
def build_Faraday(Ex, Ey, Ez, Bx, By, Bz):
    F = np.zeros((4, 4))
    F[0, 1] = Ex;  F[0, 2] = Ey;  F[0, 3] = Ez
    F[1, 0] = -Ex; F[1, 2] = Bz;  F[1, 3] = -By
    F[2, 0] = -Ey; F[2, 1] = -Bz; F[2, 3] = Bx
    F[3, 0] = -Ez; F[3, 1] = By;  F[3, 2] = -Bx
    return F


@njit(cache=True)
def back_reaction(u, F, k_rad_local):
    u_cov = ETA @ u
    E_mu  = F @ u_cov
    E_cov = ETA @ E_mu
    E_sq  = 0.0
    for i in range(4):
        E_sq += E_mu[i] * E_cov[i]
    out = np.zeros(4)
    tmp = F @ E_cov
    for i in range(4):
        out[i] = k_rad_local * (tmp[i] - E_sq * u[i])
    return out


# ==========================================
# HIGUERA-CARY STEP
# ==========================================
@njit(cache=True)
def hc_step(x, y, z, u3, alpha_local, dt, X_tilt, t):
    """
    One Higuera-Cary step (Lorentz only). Modifies u3 in place.
    Returns new (x, y, z).
    """
    Ex, Ey, Ez, Bx, By, Bz = EM_Deutsch(x, y, z, t, X_tilt)

    half_a_dt = 0.5 * alpha_local * dt

    # ---- Half electric push ----
    umx = u3[0] + half_a_dt * Ex
    umy = u3[1] + half_a_dt * Ey
    umz = u3[2] + half_a_dt * Ez

    # ---- HC magnetic rotation ----
    tx = half_a_dt * Bx
    ty = half_a_dt * By
    tz = half_a_dt * Bz

    tau_sq       = tx*tx + ty*ty + tz*tz
    um_sq        = umx*umx + umy*umy + umz*umz
    gamma_m_sq   = 1.0 + um_sq

    u_star = umx*tx + umy*ty + umz*tz
    sigma  = gamma_m_sq - tau_sq

    disc        = sigma*sigma + 4.0 * (tau_sq + u_star*u_star)
    gamma_bar   = np.sqrt(0.5 * (sigma + np.sqrt(disc)))

    # rotation vector
    rx = tx / gamma_bar
    ry = ty / gamma_bar
    rz = tz / gamma_bar
    r_sq = rx*rx + ry*ry + rz*rz
    fac  = 2.0 / (1.0 + r_sq)

    # (um x r)
    c1x = umy*rz - umz*ry
    c1y = umz*rx - umx*rz
    c1z = umx*ry - umy*rx

    # (um + c1) x (2r/(1+|r|^2))
    wx = umx + c1x
    wy = umy + c1y
    wz = umz + c1z

    upx = umx + (wy*rz - wz*ry) * fac
    upy = umy + (wz*rx - wx*rz) * fac
    upz = umz + (wx*ry - wy*rx) * fac

    # ---- Second half electric push ----
    unx = upx + half_a_dt * Ex
    uny = upy + half_a_dt * Ey
    unz = upz + half_a_dt * Ez

    # ---- Position update ----
    gamma_new = np.sqrt(1.0 + unx*unx + uny*uny + unz*unz)
    inv_g     = 1.0 / gamma_new

    x_new = x + dt * unx * inv_g
    y_new = y + dt * uny * inv_g
    z_new = z + dt * unz * inv_g

    u3[0] = unx
    u3[1] = uny
    u3[2] = unz

    return x_new, y_new, z_new


@njit(cache=True)
def rad_reaction_correction(x, y, z, u3, dt, X_tilt, t, k_rad_local):
    """
    Explicit Euler correction for the radiation reaction (operator split).
    Modifies u3 in place.
    """
    Ex, Ey, Ez, Bx, By, Bz = EM_Deutsch(x, y, z, t, X_tilt)
    F = build_Faraday(Ex, Ey, Ez, Bx, By, Bz)

    g = np.sqrt(1.0 + u3[0]*u3[0] + u3[1]*u3[1] + u3[2]*u3[2])
    u = np.array([g, u3[0], u3[1], u3[2]])

    f_rad = back_reaction(u, F, k_rad_local)

    inv_g = 1.0 / g
    u3[0] += dt * f_rad[1] * inv_g
    u3[1] += dt * f_rad[2] * inv_g
    u3[2] += dt * f_rad[3] * inv_g


# ==========================================
# CORE INTEGRATOR (returns endpoints only)
# ==========================================
@njit(cache=True)
def _hc_integrate(x0, y0, z0, X_tilt,
                  alpha_local, k_rad_local,
                  dt_max, dt_min, t_end,
                  r_crash, r_escape, safety):
    x, y, z = x0, y0, z0
    u3 = np.zeros(3)

    t = 0.0
    status = STATUS_TRAPPED
    max_steps = 50_000_000

    for _ in range(max_steps):
        if t >= t_end:
            break

        # Adaptive dt from local |B|
        Ex, Ey, Ez, Bx, By, Bz = EM_Deutsch(x, y, z, t, X_tilt)
        B_mag = np.sqrt(Bx*Bx + By*By + Bz*Bz)

        dt = safety / (alpha_local * B_mag + 1e-12)
        if dt > dt_max: dt = dt_max
        if dt < dt_min: dt = dt_min

        # Lorentz (HC)
        x, y, z = hc_step(x, y, z, u3, alpha_local, dt, X_tilt, t)

        # Radiation reaction correction
        rad_reaction_correction(x, y, z, u3, dt, X_tilt, t, k_rad_local)

        t += dt

        r = np.sqrt(x*x + y*y + z*z)
        if r <= r_crash:
            status = STATUS_CRASHED
            break
        if r >= r_escape:
            status = STATUS_EJECTED
            break

    gamma_final = np.sqrt(1.0 + u3[0]*u3[0] + u3[1]*u3[1] + u3[2]*u3[2])
    return status, x, y, z, gamma_final


# ==========================================
# PUBLIC SINGLE-PARTICLE WRAPPER
# ==========================================
def simulate_single_particle(particle_data):
    """
    Wrapper used by the parallel scheduler.
    particle_data = (index, x0, y0, z0, X_tilt)
    Returns (status, x0, y0, z0, xf, yf, zf, gamma_final)
    """
    i, x0, y0, z0, X_tilt = particle_data

    try:
        status_code, xf, yf, zf, gamma_final = _hc_integrate(
            float(x0), float(y0), float(z0), float(X_tilt),
            float(alpha), float(k_rad),
            float(DT_MAX), float(DT_MIN), float(t_max),
            float(R), float(r_esc), float(HC_SAFETY),
        )
        status = STATUS_MAP.get(int(status_code), "Trapped")

        if (not np.isfinite(gamma_final)
                or gamma_final < 1.0
                or gamma_final > 1e10):
            status = "Trapped"
            xf, yf, zf = x0, y0, z0
            gamma_final = 1.0

        return status, x0, y0, z0, xf, yf, zf, float(gamma_final)

    except Exception:
        return "Trapped", x0, y0, z0, x0, y0, z0, 1.0


# ==========================================
# FULL-TRAJECTORY VARIANT (for plotting)
# ==========================================
def integrate_trajectory_hc(x0, y0, z0, X_tilt, t_end=None):
    """
    Returns a dict with the full trajectory (x, y, z, t, gamma).
    Used only for plotting — not for the parallel run.
    """
    if t_end is None:
        t_end = t_max

    # Python loop calling njit primitives (slow but fine for 1 particle)
    x, y, z = float(x0), float(y0), float(z0)
    u3 = np.zeros(3)

    xs, ys, zs, ts, gs = [x], [y], [z], [0.0], [1.0]

    t = 0.0
    status = "Trapped"
    safety = HC_SAFETY

    for _ in range(50_000_000):
        if t >= t_end:
            break

        Ex, Ey, Ez, Bx, By, Bz = EM_Deutsch(x, y, z, t, X_tilt)
        B_mag = np.sqrt(Bx*Bx + By*By + Bz*Bz)

        dt = safety / (alpha * B_mag + 1e-12)
        if dt > DT_MAX: dt = DT_MAX
        if dt < DT_MIN: dt = DT_MIN

        x, y, z = hc_step(x, y, z, u3, alpha, dt, X_tilt, t)
        rad_reaction_correction(x, y, z, u3, dt, X_tilt, t, k_rad)
        t += dt

        g = float(np.sqrt(1.0 + u3[0]**2 + u3[1]**2 + u3[2]**2))
        xs.append(x); ys.append(y); zs.append(z); ts.append(t); gs.append(g)

        r = float(np.sqrt(x*x + y*y + z*z))
        if r <= R:
            status = "Crashed"; break
        if r >= r_esc:
            status = "Ejected"; break

    return {
        "x": np.array(xs), "y": np.array(ys), "z": np.array(zs),
        "t": np.array(ts), "gammas": np.array(gs),
        "status": status,
    }