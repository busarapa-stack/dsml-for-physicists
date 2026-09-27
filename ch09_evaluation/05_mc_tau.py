"""Ch.9  Integrated autocorrelation time, block bootstrap and blocking for Metropolis energies of the 2D Ising model.

Book references: ssec:boot-mc-primary, def:tau-int, code:mc-tau (line for line), tab:mc-tau
init_lattice, total_energy and metropolis_step come from code:metropolis of Chapter 4 (executed from
../ch04_statistics, without its temperature scan).  Each temperature takes about 45 s in pure Python;
the script runs T = 2.3 (the listing) and then T = 2.0 and 3.0 for the table: about 2.5 minutes.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.chdir(HERE)
_s = (HERE.parent / 'ch04_statistics' / '07_metropolis_ising.py').read_text()
_s = _s[_s.index('\n# --- code:metropolis'):]
exec(_s[:_s.index('def simulate_T')])                                   # rng, init_lattice, total_energy, metropolis_step

# --- code:mc-tau -------------------------------------------------------------
# uses init_lattice, total_energy and metropolis_step from the Chapter 4 listing
def energy_series(L, T, n_thermalize, n_measure):
    lattice = init_lattice(L)
    beta = 1.0 / T
    for _ in range(n_thermalize):
        metropolis_step(lattice, beta)
    E = np.empty(n_measure)
    for k in range(n_measure):
        metropolis_step(lattice, beta)
        E[k] = total_energy(lattice)
    return E

def tau_int(x, c=5.0):
    """Integrated autocorrelation time with Sokal's automatic window."""
    x = np.asarray(x) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, 2 * n)                      # zero-padded FFT
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0]
    tau = 0.5
    for W in range(1, n):
        tau += acf[W]
        if W >= c * tau:
            break
    return tau, W

def block_bootstrap(x, stat, block, n_boot=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    n_blocks = len(x) // block
    blocks = np.asarray(x)[:n_blocks * block].reshape(n_blocks, block)
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, n_blocks, n_blocks)   # resample whole blocks
        out[b] = stat(blocks[pick].ravel())
    return out

rng = np.random.default_rng(seed=2026)             # restart the generator
L, T = 16, 2.3
e = energy_series(L, T, n_thermalize=2000, n_measure=20000) / L**2
tau, W = tau_int(e)
se_naive = e.std(ddof=1) / np.sqrt(len(e))
print(f"<e> = {e.mean():.4f}, tau_int = {tau:.1f} sweeps (window {W})")
print(f"SE naive = {se_naive:.5f}, SE with tau = {se_naive*np.sqrt(2*tau):.5f}")
cv = lambda x: x.var() * L**2 / T**2               # C_v per spin
for stat, name in [(np.mean, '<e>'), (cv, 'C_v/N')]:
    boot = block_bootstrap(e, stat, block=200)
    print(name, "block-bootstrap SE", boot.std().round(5),
          "95% CI", np.percentile(boot, [2.5, 97.5]).round(4))
# --- end of listings ---------------------------------------------------------


def blocking(x, min_blocks=16):
    """Flyvbjerg-Petersen: SE of the mean after repeatedly averaging neighbouring pairs."""
    x = np.asarray(x, float); out = []
    while len(x) >= min_blocks:
        out.append(np.sqrt(x.var(ddof=1) / len(x)))
        m = len(x) // 2 * 2
        x = 0.5 * (x[:m:2] + x[1:m:2])
    return np.array(out)


def row(e_series, T):
    tau_T, _ = tau_int(e_series)
    naive = e_series.std(ddof=1) / np.sqrt(len(e_series))
    cvT = lambda x: x.var() * L**2 / T**2
    return (e_series.mean(), tau_T, naive, naive * np.sqrt(2 * tau_T),
            block_bootstrap(e_series, np.mean, block=200).std(),
            block_bootstrap(e_series, cvT, block=1).std(), block_bootstrap(e_series, cvT, block=200).std(), cvT(e_series))


if __name__ == '__main__':
    print(f"N_eff = {len(e) / (2 * tau):.0f} of {len(e)}; corrected / naive SE = {np.sqrt(2 * tau):.1f}")
    print("blocking SE of <e> (levels 0, 1, 2, ...):", blocking(e).round(5))
    for blk in (50, 100, 200, 500, 1000):
        print(f"block bootstrap SE of <e>, blocks of {blk:4d}: {block_bootstrap(e, np.mean, block=blk).std():.5f}")
    print(f"C_v/N = {cv(e):.3f}; single-point bootstrap SE {block_bootstrap(e, cv, block=1).std():.4f}")

    rows = {2.3: row(e, 2.3)}
    for T_other in (2.0, 3.0):
        rng = np.random.default_rng(seed=2026)                          # same restart as in the listing
        e_T = energy_series(L, T_other, n_thermalize=2000, n_measure=20000) / L**2
        rows[T_other] = row(e_T, T_other)
    print("\ntab:mc-tau   T   <e>      tau_int  SE naive  SE tau   SE block | C_v/N  SE single  SE block")
    for T_val in (2.0, 2.3, 3.0):
        r = rows[T_val]
        print(f"           {T_val:.1f} {r[0]:8.4f} {r[1]:6.1f} {r[2]:9.5f} {r[3]:8.4f} {r[4]:8.4f} | {r[7]:5.3f} {r[5]:9.4f} {r[6]:9.4f}")
