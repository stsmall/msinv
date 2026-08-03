"""E[T] under moments exponential growth vs constant Ne=775k, for panmictic,
within-arrangement (single-origin I), within-S, and between-arrangement pairs."""
import numpy as np

N0, NANC, TGROW = 6_808_096.0, 547_928.0, 769_519.0
ALPHA = np.log(N0 / NANC) / TGROW
MU = 3e-9
P_I, P_S = 0.626, 0.374
NE_CONST = 775_000.0

def N_growth(t):
    return np.where(t <= TGROW, N0 * np.exp(-ALPHA * np.minimum(t, TGROW)), NANC)

def N_const(t):
    return np.full_like(np.asarray(t, dtype=float), NE_CONST)

def ET(hazard_fn, t_max, forced_at=None, dt=200.0):
    """E[T] by numerical integration. forced_at: mass point (single-origin cap)."""
    t = np.arange(0.0, t_max, dt) + dt / 2
    h = hazard_fn(t)
    S = np.exp(-np.cumsum(h * dt))
    e = float(np.sum(t * h * S * dt))
    if forced_at is not None:
        e += forced_at * float(S[-1])          # survivors forced to coalesce
    else:
        e += float(S[-1]) * (t_max + 0.0)      # negligible if t_max large
    return e

def suite(Nfn, t_inv, horizon=40e6):
    # panmictic
    pan = ET(lambda t: 1.0 / (2 * Nfn(t)), horizon)
    # I class: hazard scaled by p_I, forced coalescence at t_inv
    Ti = ET(lambda t: 1.0 / (2 * Nfn(t) * P_I), t_inv, forced_at=t_inv)
    # S class: p_S-scaled until t_inv, panmictic after
    def hS(t):
        return np.where(t < t_inv, 1.0 / (2 * Nfn(t) * P_S), 1.0 / (2 * Nfn(t)))
    Ts = ET(hS, horizon)
    # between: no coalescence until t_inv, then panmictic
    def hB(t):
        return np.where(t < t_inv, 0.0, 1.0 / (2 * Nfn(t)))
    Tb = ET(hB, horizon)
    return pan, Ti, Ts, Tb

print(f"alpha={ALPHA:.4e}  frac of panmictic pairs coalescing during growth phase="
      f"{1-np.exp(-np.trapezoid(1/(2*N_growth(np.linspace(0,TGROW,20000))), np.linspace(0,TGROW,20000))):.3f}")

for label, Nfn in (("GROWTH (moments)", N_growth), ("CONST Ne=775k", N_const)):
    pan, *_ = suite(Nfn, 1.1e6)
    print(f"\n{label}: panmictic E[T] = {pan:,.0f} gen -> pi = {2*MU*pan:.5f}  (observed 0.00930)")

print("\n" + "="*78)
print(f"{'t_inv':>10} | {'GROWTH pi_I/pi_S':>17} {'GROWTH dxy/pi_I':>16} | {'CONST pi_I/pi_S':>16} {'CONST dxy/pi_I':>15}")
print("="*78)
best = {}
for t_inv in [2e5, 4e5, 6e5, 8e5, 1.0e6, 1.1e6, 1.3e6, 1.6e6, 2.0e6, 2.5e6, 3.0e6, 4.0e6]:
    _, gi, gs, gb = suite(N_growth, t_inv)
    _, ci, cs, cb = suite(N_const, t_inv)
    gr, gd = gi/gs, gb/gi
    cr, cd = ci/cs, cb/ci
    for k, v in (("growth", gd), ("const", cd)):
        if k not in best or v < best[k][1]:
            best[k] = (t_inv, v)
    print(f"{t_inv:>10,.0f} | {gr:>17.3f} {gd:>16.3f} | {cr:>16.3f} {cd:>15.3f}")
print("="*78)
print(f"dxy/pi_I FLOOR  growth: {best['growth'][1]:.3f} at t_inv={best['growth'][0]:,.0f}"
      f"   |   const: {best['const'][1]:.3f} at t_inv={best['const'][0]:,.0f}")
print(f"OBSERVED dxy/pi_AA = 1.846   pi_AA/pi_BB = 0.744")

# --- refine: solve t_inv for observed pi_I/pi_S=0.744, and locate floors ---
from scipy.optimize import brentq, minimize_scalar
OBS_RATIO, OBS_DXY = 0.744, 1.846
print("\n" + "="*78)
for label, Nfn in (("GROWTH (moments)", N_growth), ("CONST Ne=775k", N_const)):
    f = lambda t: (lambda s: s[1]/s[2])(suite(Nfn, t)) - OBS_RATIO
    t_fit = brentq(f, 3e5, 3e6, xtol=2000)
    res = minimize_scalar(lambda t: (lambda s: s[3]/s[1])(suite(Nfn, t)),
                          bounds=(3e5, 4e6), method='bounded',
                          options={'xatol': 5000})
    _, Ti, Ts, Tb = suite(Nfn, t_fit)
    print(f"{label}:")
    print(f"   t_inv from pi_I/pi_S=0.744 -> {t_fit:,.0f} gen ({t_fit/1e6:.2f} My)")
    print(f"   dxy/pi_I predicted at that t_inv = {Tb/Ti:.3f}  (observed {OBS_DXY})")
    print(f"   dxy/pi_I FLOOR = {res.fun:.3f} at t_inv={res.x:,.0f}")
    print(f"   flux shortfall = {res.fun/OBS_DXY:.2f}x below floor")
