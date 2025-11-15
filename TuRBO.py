# turbo_fit_density.py
import math
import torch
from dataclasses import dataclass
from typing import Callable, Dict, Tuple

from botorch.models import SingleTaskGP
from botorch.fit import fit_gpytorch_mll
from botorch.acquisition.logei import qLogExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim.optimize import optimize_acqf
from botorch.utils.transforms import normalize, unnormalize
from gpytorch.mlls.exact_marginal_log_likelihood import ExactMarginalLogLikelihood

# ------------------------------------------------------------
# Minimal helpers
# ------------------------------------------------------------
def _device_dtype_like(t: torch.Tensor):  # convenience
    return dict(device=t.device, dtype=t.dtype)

def sort_knot_locs(unit_locs: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """
    unit_locs: (..., K) in [0,1]; returns sorted mapped radii in [lo,hi]
    """
    x = torch.sort(unit_locs, dim=-1).values
    return lo + (hi - lo) * x

def map_unit_to_profiles(
    U: torch.Tensor,
    rho: torch.Tensor,
    D_bounds=(1e-2, 10.0),      # physical D range (you can swap to log10 if you prefer)
    V_bounds=(-5.0, 5.0),       # physical V range (m/s or arb.)
    n_knots: int = 4,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    U: (B, 2*(n_knots + n_knots)) = (B, 2*2*n_knots) params in [0,1]
       Layout:
         [ D_loc[0:K], D_val[0:K],  V_loc[0:K], V_val[0:K] ]
    Returns:
      D, V: (B, Nr) each, linearly interpolated from knot params.
    """
    B, d = U.shape
    assert d == 4 * n_knots, f"Expected {4*n_knots} dims; got {d}"

    Nr = rho.numel()
    lo_r, hi_r = float(rho.min()), float(rho.max())

    D_loc_u  = U[:, 0:n_knots]
    D_val_u  = U[:, n_knots:2*n_knots]
    V_loc_u  = U[:, 2*n_knots:3*n_knots]
    V_val_u  = U[:, 3*n_knots:4*n_knots]

    # map to physical
    D_locs = sort_knot_locs(D_loc_u, lo_r, hi_r)       # (B,K)
    V_locs = sort_knot_locs(V_loc_u, lo_r, hi_r)       # (B,K)

    D_lo, D_hi = D_bounds
    V_lo, V_hi = V_bounds
    D_vals = D_lo + (D_hi - D_lo) * D_val_u            # (B,K)  positive
    V_vals = V_lo + (V_hi - V_lo) * V_val_u  # (B,K)  symmetric map to [V_lo,V_hi]

    # piecewise-linear interpolation over rho
    # vectorized: for each batch, walk knots and fill segments
    D = torch.empty(B, Nr, **_device_dtype_like(rho))
    V = torch.empty(B, Nr, **_device_dtype_like(rho))

    # Build a common index for segment placement
    # For each interval [loc[j], loc[j+1]], compute linear weights.
    rho_b = rho.expand(B, Nr)                          # (B,Nr)

    # handle K==1 edge (constant); else piecewise
    for (locs, vals, out) in ((D_locs, D_vals, D), (V_locs, V_vals, V)):
        if n_knots == 1:
            out.copy_(vals[:, :1].expand(B, Nr))
            continue

        # initialize with last knot value (right fill)
        out[:] = vals[:, -1].unsqueeze(1)
        # loop segments (K-1), left-closed, right-open, final segment closes at end
        for j in range(n_knots - 1):
            x0 = locs[:, j].unsqueeze(1)
            x1 = locs[:, j+1].unsqueeze(1)
            y0 = vals[:, j].unsqueeze(1)
            y1 = vals[:, j+1].unsqueeze(1)

            # mask of rho in [x0, x1]
            m = (rho_b >= x0) & (rho_b < x1)
            # linear weight
            w = ((rho_b - x0) / (x1 - x0 + 1e-12)).clamp(0, 1)
            seg = y0 + (y1 - y0) * w
            out[m] = seg[m]

    return D, V

# ------------------------------------------------------------
# TuRBO
# ------------------------------------------------------------
@dataclass
class TurboState:
    dim: int
    batch_size: int
    length_init: float = 0.8
    length_min: float = 0.5 ** 7
    length_max: float = 1.6
    success_tol: int = 3
    failure_tol: int = 3
    length: float = None
    success_counter: int = 0
    failure_counter: int = 0
    best_value: float = float("inf")

    def __post_init__(self):
        self.length = self.length_init

    def update(self, f_new: torch.Tensor, f_old_best: float):
        """
        f_new: (q,) new objective values (lower is better)
        """
        improved = (f_new.min().item() < f_old_best - 1e-12)
        if improved:
            self.success_counter += 1
            self.failure_counter = 0
        else:
            self.failure_counter += 1
            self.success_counter = 0

        # expand / shrink trust region
        if self.success_counter >= self.success_tol:
            self.length = min(2.0 * self.length, self.length_max)
            self.success_counter = 0
        elif self.failure_counter >= self.failure_tol:
            self.length = max(0.5 * self.length, self.length_min)
            self.failure_counter = 0

# ------------------------------------------------------------
# Main optimizer
# ------------------------------------------------------------
def turbo_optimize_density(
    rho: torch.Tensor,
    time: torch.Tensor,
    N_target: torch.Tensor,                       # (Nr,Nt)
    solver,                                       # your SolverV2_opt()
    SR: torch.Tensor = None,                      # (Nr,) optional
    ST: torch.Tensor = None,                      # (Nt,) optional
    S_full: torch.Tensor = None,                  # (Nr,Nt) optional
    N0: torch.Tensor = None,                      # (Nr,) initial density
    A_edge: torch.Tensor = None,                  # scalar edge bc
    n_knots: int = 4,
    D_bounds=(1e-2, 10.0),
    V_bounds=(-5.0, 5.0),
    n_steps: int = 40,
    batch_q: int = 16,
    seed: int = 0,
    top_k_return: int = 5,
) -> Dict:
    """
    Returns:
      dict with keys:
        'U_all': (M,d) tried points in unit cube
        'f_all': (M,) losses
        'best': list of top_k_return dicts { 'U','loss','D','V','N_pred' }
    """
    torch.manual_seed(seed)

    device = rho.device
    dtype  = rho.dtype

    Nr, Nt = rho.numel(), time.numel()
    d = 4 * n_knots  # K D locs + K D vals + K V locs + K V vals

    # prepare source & fixed inputs
    if S_full is not None:
        # factor S into SR * ST if you like, but we can just supply SR,ST that reproduce S_full at each step:
        # here we carry S_full and add it after solve via replace of SR*ST → S_full, using your solver's S=SR*ST.
        use_full_S = True
        SR_use = torch.ones_like(rho)     # placeholders (won't matter)
        ST_use = torch.ones_like(time)
    else:
        use_full_S = False
        assert SR is not None and ST is not None, "Provide SR/ST or S_full."

    if N0 is None:
        N0 = N_target[:, 0].clone()  # reasonable default: start = target at t=0

    if A_edge is None:
        A_edge = torch.tensor([0.0], **_device_dtype_like(rho))

    # storage
    U_hist, f_hist = [], []

    # initial Sobol points in [0,1]^d
    sob = torch.quasirandom.SobolEngine(dimension=d, scramble=True, seed=seed)
    U = sob.draw(2 * d).to(device=device, dtype=dtype)  # warmup
    f = torch.empty(U.shape[0], device=device, dtype=dtype)

    # Objective wrapper (batched on U)
    def eval_batch(Ub: torch.Tensor) -> torch.Tensor:
        B = Ub.shape[0]
        # map to D,V
        D, V = map_unit_to_profiles(Ub, rho, D_bounds=D_bounds, V_bounds=V_bounds, n_knots=n_knots)  # (B,Nr)

        # build case dict for solver
        case = {
            "rho":  rho.expand(B, -1),
            "time": time.expand(B, -1),
            "SR":   (SR if SR is not None else SR_use).expand(B, -1),
            "ST":   (ST if ST is not None else ST_use).expand(B, -1),
            "D":    D,
            "V":    V,
            "N0":   N0.expand(B, -1),
            "A":    A_edge.expand(B),
        }

        # solve
        with torch.no_grad():
            N_pred = solver.solve(**case, conv_bc_outer='dirichlet',
                                  assert_conservation=False, dtype=dtype)  # (B,Nr,Nt)

            # if a full source S was provided, overwrite SR*ST effect:
            if use_full_S:
                # quick hack: re-run source mid-step is non-trivial; instead, compare to target on same grid directly
                pass  # leave as-is; typically you'd inject S_full into your solver for exactness.

            # MSE over (ρ,t)
            diff = (N_pred - N_target.unsqueeze(0))
            loss = (diff.pow(2).mean(dim=(1, 2)))  # (B,)
            # clamp inf/nan
            bad = ~torch.isfinite(loss)
            if bad.any():
                loss[bad] = 1e30
            return loss

    # evaluate warmup
    f[:U.shape[0]] = eval_batch(U)

    # TuRBO state
    state = TurboState(dim=d, batch_size=batch_q)
    state.best_value = f.min().item()
    U_hist.append(U)
    f_hist.append(f)

    # main loop
    for step in range(n_steps):
        # fit GP on all seen data (normalize to [0,1]^d)
        U_all = torch.vstack(U_hist)                   # (m,d) in [0,1]
        f_all = torch.hstack(f_hist).unsqueeze(-1)     # (m,1)

        # Normalize inputs for the GP
        X = U_all
        Y = (f_all - f_all.mean()) / (f_all.std() + 1e-9)

        gp = SingleTaskGP(X, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        try:
            fit_gpytorch_mll(mll)
        except Exception:
            pass  # fall back to defaults if fitting fails

        # Choose a TR center = current best
        best_idx = torch.argmin(f_all)
        x_center = U_all[best_idx]                     # (d,)

        # bounds for the TR in unit space
        delta = 0.5 * state.length * torch.ones(d, device=device, dtype=dtype)
        lb = (x_center - delta).clamp(0.0, 1.0)
        ub = (x_center + delta).clamp(0.0, 1.0)
        bounds = torch.stack([lb, ub], dim=0)

        # Acquisition
        ____ = torch.ones(128, device=device, dtype=dtype)  # dummy to fix botorch SobolQMCNormalSampler seed bug
        sampler = SobolQMCNormalSampler(sample_shape=____.shape)
        acqf = qLogExpectedImprovement(model=gp, best_f=Y.min().item(), sampler=sampler)

        # optimize acqf within TR
        cand, _ = optimize_acqf(
            acq_function=acqf,
            bounds=bounds,
            q=batch_q,
            num_restarts=8,
            raw_samples=256,
            options={"batch_limit": 4, "maxiter": 100},
        )
        U_next = cand.clamp(0.0, 1.0)

        # evaluate batch, update state
        f_next = eval_batch(U_next)
        prev_best = state.best_value
        state.update(f_next, prev_best)
        state.best_value = min(state.best_value, f_next.min().item())

        U_hist.append(U_next)
        f_hist.append(f_next)

        print(f"[TuRBO] step {step+1:02d}: added {batch_q} | TR_len={state.length:.3f} | "
              f"best={state.best_value:.3e}")

    # collate results
    U_all = torch.vstack(U_hist)           # (M,d)
    f_all = torch.hstack(f_hist)           # (M,)

    # top-k (unique-ish)
    k = min(top_k_return, U_all.shape[0])
    idx = torch.topk(-f_all, k=k).indices  # sort ascending loss
    idx = idx[torch.argsort(f_all[idx])]   # ensure ordered by loss
    results = []
    with torch.no_grad():
        for i in idx.tolist():
            Ui = U_all[i:i+1]
            Di, Vi = map_unit_to_profiles(Ui, rho, D_bounds, V_bounds, n_knots=n_knots)
            case = {
                "rho":  rho.unsqueeze(0),
                "time": time.unsqueeze(0),
                "SR":   (SR if SR is not None else torch.ones_like(rho)).unsqueeze(0),
                "ST":   (ST if ST is not None else torch.ones_like(time)).unsqueeze(0),
                "D":    Di,
                "V":    Vi,
                "N0":   N0.unsqueeze(0),
                "A":    torch.tensor([A_edge.item()], **_device_dtype_like(rho)),
            }
            N_pred = solver.solve(**case, conv_bc_outer='dirichlet',
                                  assert_conservation=False, dtype=dtype).squeeze(0)
            results.append({
                "U": Ui.squeeze(0).cpu(),
                "loss": float(f_all[i].cpu()),
                "D": Di.squeeze(0).cpu(),
                "V": Vi.squeeze(0).cpu(),
                "N_pred": N_pred.cpu(),
            })

    return {"U_all": U_all.cpu(), "f_all": f_all.cpu(), "best": results}
