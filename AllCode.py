import os
os.environ['LOKY_MAX_CPU_COUNT'] = '12' # Replace 4 with your desired core count
os.environ["TORCHINDUCTOR_MAX_AUTOTUNE"] = "0"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["CUDA_LAUNCH_BLOCKING"]="1"

import torch
import math
import numpy as np
import time
import matplotlib.pyplot as plt
import pickle
from pathlib import Path    
from scipy.interpolate import PchipInterpolator
from sklearn.mixture import GaussianMixture
from typing import Callable, Dict, List
from dataclasses import dataclass

from Spedup import PytorchPchipInterpolator


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}") 


##### ReadPickle / WritePickle ##### ReadPickle / WritePickle ##### ReadPickle / WritePickle ##### ReadPickle / WritePickle ##### ReadPickle / WritePickle ##### ReadPickle / WritePickle ##### ReadPickle / WritePickle ##### ReadPickle / WritePickle #####


def WritePickle(obj: object, filepath: str, safe=True) -> None:
    r"""
    Description
    -----------
    Writes a Python object to a .pickle file.
    
    Parameters
    ----------
    `obj`: object | Required
        The object to be pickled.
    
    `filepath`: str | Required
        The path to save the pickle file to.
    
    `safe`: bool | Optional
        If True, checks if the file already exists and asks to overwrite. Default is True.
        If False, overwrites the file without asking.
    """
    #Input Checks
    if (str(filepath).endswith('.pickle') == False) & (str(filepath).endswith('.pkl') == False):
        filepath += '.pickle'
        
    #Check If File Exists, Asks To Overwrite
    if safe == True:
        if Path(filepath).exists():
            overwrite = input(f"File {str(filepath)} already exists. Overwrite? (y/n): ")
            if overwrite.lower() != 'y':
                print("File not overwritten.")
                return
    #Write The File
    try:
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise IOError(f"Error writing to file {str(filepath)}: {e}") from e
    
def ReadPickle(filepath: str) -> object:
    r"""
    Description
    -----------
    Reads a .pickle file and returns the object contained within.
    
    Parameters
    ----------
    `filepath`: str | Required
        The path to the pickle file to be read.
    
    Returns
    -------
    `object`: object
        The object contained within the pickle file.
    """
    #Input Checks
    if (str(filepath).endswith('.pickle') == False) & (str(filepath).endswith('.pkl') == False):
        filepath = str(filepath) + '.pickle'
    
    #Read The File
    try:
        with open(filepath, 'rb') as f:
            return pickle.load(f)
        
    #Error Handling
    except FileNotFoundError:
        raise FileNotFoundError(f"File {str(filepath)} not found.")
    except Exception as e:
        raise IOError(f"Error reading file {str(filepath)}: {e}") from e
    

##### Solver ##### Solver ##### Solver ##### Solver ##### Solver ##### Solver ##### Solver ##### Solver ##### Solver ##### Solver ##### Solver ##### Solver ##### Solver ##### Solver ##### Solver #####


@torch.jit.script
def thomas_batched_jit(a, b, c, d):
        a = a.clone(); b = b.clone(); c = c.clone(); d = d.clone()
        B, N = b.shape
        for i in range(1, N):
            w = a[:, i] / b[:, i-1]
            b[:, i] -= w * c[:, i-1]
            d[:, i] -= w * d[:, i-1]
        x = d
        x[:, -1] = d[:, -1] / b[:, -1]
        for i in range(N-2, -1, -1):
            x[:, i] = (d[:, i] - c[:, i] * x[:, i+1]) / b[:, i]
        return x

class SolverV2_opt:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def _apply_tridiag(self, a, b, c, x):
        y = b*x
        y[:, 1:]  += a[:, 1:]*x[:, :-1]
        y[:, :-1] += c[:, :-1]*x[:, 1:]
        return y

    def _thomas_batched(self, a, b, c, d):
        return thomas_batched_jit(a, b, c, d) #! 492.5s / 4090.7s

    def solve(self, rho, time, SR, ST, D, V, N0, A, conv_bc_outer='dirichlet', assert_conservation=True, dtype=torch.float32):
        '''
        rho: torch tensor, shape (batch_size, nr)
        time: torch tensor, shape (batch_size, nt)
        SR: torch tensor, shape (batch_size, nr)
        ST: torch tensor, shape (batch_size, nt)
        D:  torch tensor, shape (batch_size, nr)
        V:  torch tensor, shape (batch_size, nr)
        N0: torch tensor, shape (batch_size, nr)
        A:  torch tensor, shape (batch_size,)
        conv_bc_outer: str, 'dirichlet' or 'extrapolate_A'
        assert_conservation: bool, whether to assert conservation on diffusion interiors
        dtype: torch dtype, default torch.float32
        
        self.S: torch tensor, shape (batch_size, nr, nt) = SR*ST
        '''
        #Assumptions/ Hyperparameters
        m = 1
        device = self.device
        tiny = torch.finfo(dtype).eps
        
        #Move to device, set dtype
        rho = rho.to(device, dtype=dtype)
        time = time.to(device, dtype=dtype)
        SR = SR.to(device, dtype=dtype)
        ST = ST.to(device, dtype=dtype)
        D = D.to(device, dtype=dtype)
        V = V.to(device, dtype=dtype)
        N0 = N0.to(device, dtype=dtype)
        A = A.to(device, dtype=dtype)
        
        #Dimensions
        batch_size, nr = SR.shape
        nt = ST.shape[1] #NOTE, nr is the number of cells, not faces
        buffer = torch.zeros((batch_size, 1)).to(device, dtype=dtype)
        big_buffer = torch.zeros((batch_size, nr)).to(device, dtype=dtype)
        N_buffer = torch.empty((batch_size, nr, nt- 1)).to(device, dtype=dtype)
        
        #Centers 
        p_i = rho
        
        #Faces, sets edge faces to 0 and rmax
        r_f0 = buffer if m > 0 else (p_i[:, 0] - 0.5*(p_i[:, 1] - p_i[:, 0])).unsqueeze(1)
        r_fi = 0.5*(p_i[:, :-1] + p_i[:, 1:])
        r_ff = p_i[:, -1].unsqueeze(1)
        r_f = torch.cat([r_f0, r_fi, r_ff], dim=1).to(device, dtype=dtype)
        #r_f = rho
        
        #Distances
        dr_neg = p_i - r_f[:, :-1]
        dr_pos = r_f[:, 1:] - p_i
        dr_c = dr_neg + dr_pos
        
        #Metrics
        r_im = torch.clamp(p_i, min=tiny)**m
        r_fm = torch.clamp(r_f, min=tiny)**m
        if m > 0:
            r_fm[:, 0] = buffer.squeeze()
        
        #Face-Averaged Diffusion Coefficients
        D_f0 =D[:, 0].unsqueeze(1)
        D_fi = 2*D[:, :-1]*D[:, 1:]/(D[:, :-1] + D[:, 1:] + tiny)
        D_ff = D[:, -1].unsqueeze(1)
        D_f = torch.cat([D_f0, D_fi, D_ff], dim=1).to(device, dtype=dtype)
        
        #Face-Averaged Convection Coefficients
        V_f0 = V[:, 0].unsqueeze(1)
        V_fi = 0.5*(V[:, :-1] + V[:, 1:])
        V_ff = V[:, -1].unsqueeze(1)
        V_f = torch.cat([V_f0, V_fi, V_ff], dim=1).to(device, dtype=dtype)
        
        #Diffusion Operator
        D_a0 = buffer
        D_ai = (r_fm[:, 1:-2]*D_f[:, 1:-2])/(r_im[:, 1:-1]*dr_c[:, 1:-1]*dr_neg[:, 1:-1])
        D_af = ((r_fm[:, -2]*D_f[:, -2])/(r_im[:, -1]*dr_c[:, -1]*dr_neg[:, -1])).unsqueeze(1)
        
        #!D_c0 = ((2*r_fm[:, 1]*D_f[:, 1])/(r_im[:, 0]*dr_c[:, 0]*dr_pos[:, 0])).unsqueeze(1)
        D_c0 = ((r_fm[:, 1]*D_f[:, 1])/(r_im[:, 0]*dr_c[:, 0]*dr_pos[:, 0])).unsqueeze(1)
        D_ci = (r_fm[:, 2:-1]*D_f[:, 2:-1])/(r_im[:, 1:-1]*dr_c[:, 1:-1]*dr_pos[:, 1:-1])
        D_cf = buffer
        
        D_b0 = -D_c0
        D_bi = -(D_ai + D_ci)
        D_bf = -D_af
        
        G_f = ((r_fm[:, -1]*D_f[:, -1]*A)/(r_im[:, -1]*dr_c[:, -1])).unsqueeze(1)
        
        D_a = torch.cat([D_a0, D_ai, D_af], dim=1)
        D_c = torch.cat([D_c0, D_ci, D_cf], dim=1)
        D_b = torch.cat([D_b0, D_bi, D_bf], dim=1)
        G = torch.cat([big_buffer[:, :-1], G_f], dim=1)
        
        #! Remove
        # Quick diagnostic: diffusion row sums (should be ~0 in interiors)
        #row_sum = (D_a + D_b + D_c)[:, 1:-1]
        #print("check row-sum max(abs) (diff interiors):", row_sum.abs().max().item())

        #! Remove
        
        #Convection Operator
        C_alpha_neg_0 = (r_fm[:, 0]*V_f[:, 0])/(r_im[:, 0]*dr_c[:, 0])
        C_alpha_neg_i = (r_fm[:, 1:-2]*V_f[:, 1:-2])/(r_im[:, 1:-1]*dr_c[:, 1:-1])
        C_alpha_neg_f = (r_fm[:, -2]*V_f[:, -2])/(r_im[:, -1]*dr_c[:, -1])
        
        C_alpha_pos_0 = (r_fm[:, 1]*V_f[:, 1])/(r_im[:, 0]*dr_c[:, 0])
        C_alpha_pos_i = (r_fm[:, 2:-1]*V_f[:, 2:-1])/(r_im[:, 1:-1]*dr_c[:, 1:-1])
        C_alpha_pos_f = (r_fm[:, -1]*V_f[:, -1])/(r_im[:, -1]*dr_c[:, -1])
        
        mask_neg = V_f[:, 1:-2] >= 0
        mask_pos = V_f[:, 2:-1] >= 0

        C_a0 = buffer
        C_ai = torch.where(mask_neg,  C_alpha_neg_i, big_buffer[:,2:])
        C_af = torch.where(V_f[:, -2] >= 0, C_alpha_neg_f, buffer.squeeze()).unsqueeze(1)
        
        C_c0 = torch.where(V_f[:, 1] < 0, -C_alpha_pos_0, buffer.squeeze()).unsqueeze(1)
        C_ci = torch.where(~mask_pos, -C_alpha_pos_i, big_buffer[:,2:])
        C_cf = buffer
        
        C_b0 = (torch.where(V_f[:, 0] < 0, C_alpha_neg_0, buffer.squeeze()) + torch.where(V_f[:, 1] >= 0, -C_alpha_pos_0, buffer.squeeze())).unsqueeze(1)
        C_bi = torch.where(~mask_neg, C_alpha_neg_i, big_buffer[:,2:]) + torch.where(mask_pos, -C_alpha_pos_i, big_buffer[:,2:])
        C_bf = (torch.where(V_f[:, -2] < 0, C_alpha_neg_f, buffer.squeeze()) + torch.where(V_f[:, -1] >= 0, -C_alpha_pos_f, buffer.squeeze())).unsqueeze(1)

        inflow_mask = V_f[:, -1] < 0
        
        if conv_bc_outer == 'dirichlet':
            #? = added
            N_in = N0[:, -1] #?
            H_f = torch.where(inflow_mask, -C_alpha_pos_f*N0[:, -1], buffer.squeeze()).unsqueeze(1)
            C_bf = C_bf + torch.where(inflow_mask, -C_alpha_pos_f, buffer.squeeze()).unsqueeze(1)
        elif conv_bc_outer == 'extrapolate_A':
            H_f = torch.where(inflow_mask, -C_alpha_pos_f*(A*dr_pos[:, -1]), buffer.squeeze()).unsqueeze(1)
            C_bf = C_bf + torch.where(inflow_mask, -C_alpha_pos_f, buffer.squeeze()).unsqueeze(1)
        
        C_a = torch.cat([C_a0, C_ai, C_af], dim=1)
        C_c = torch.cat([C_c0, C_ci, C_cf], dim=1)
        C_b = torch.cat([C_b0, C_bi, C_bf], dim=1)
        H = torch.cat([big_buffer[:, :-1], H_f], dim=1)
        
        #Setup for time-stepping
        theta = 0.5
        conv_theta = 1
        N = torch.cat([N0.unsqueeze(2), N_buffer], dim=2)  #shape (batch_size, nr, nt)
        
        #Construct Constant Matrices
        RHS_bc  = G + H
        Da_th = D_a*theta; Db_th = D_b*theta; Dc_th = D_c*theta
        Ca_th = C_a*conv_theta; Cb_th = C_b*conv_theta; Cc_th = C_c*conv_theta
        Da_oth = D_a*(1.0-theta); Db_oth = D_b*(1.0-theta); Dc_oth = D_c*(1.0-theta)
        Ca_tha = C_a*(1.0-conv_theta); Cb_tha = C_b*(1.0-conv_theta); Cc_tha = C_c*(1.0-conv_theta)

        #Time-stepping
        for n in range(1, nt):
            dt = time[:, n] - time[:, n-1]  #shape (batch_size,)
            dt = dt.view(-1,1)
            
            #Source term
            S_mid = SR*0.5*(ST[:, n] + ST[:, n-1]).unsqueeze(1)
            
            #Left-hand side matrix
            LHS_a = -dt*(Da_th + Ca_th)
            LHS_b = 1 - dt*(Db_th + Cb_th)
            LHS_c = -dt*(Dc_th + Cc_th)
            
            #Right-hand side matrix
            #RHS = self._apply_tridiag(Da_oth, Db_oth, Dc_oth, N[:, :, n-1])
            RHS = N[:, :, n-1] + dt*(self._apply_tridiag(Da_oth, Db_oth, Dc_oth, N[:, :, n-1]) + S_mid + RHS_bc)
            if conv_theta < 1.0:
                RHS = N[:, :, n-1] + dt*(self._apply_tridiag(Da_oth, Db_oth, Dc_oth, N[:, :, n-1]) + self._apply_tridiag(Ca_tha, Cb_tha, Cc_tha, N[:, :, n-1]) + S_mid + RHS_bc)

            #Solve
            N[:, :, n] = self._thomas_batched(LHS_a, LHS_b, LHS_c, RHS)
        
        #Validate conservation on diffusion interiors
        if assert_conservation:
            res = D_a[:, 1:-1] + D_b[:, 1:-1] + D_c[:, 1:-1]
            max_abs = res.abs().max().item()
            print("max |row sum| (diffusion interiors):", max_abs)
            assert torch.allclose(D_a[:,1:-1] + D_b[:,1:-1] + D_c[:,1:-1], torch.zeros_like(D_b[:,1:-1]), atol=1e-2)
        #Validate shapes
        #for T in (D_a, D_b, D_c, C_a, C_b, C_c, G, H):
            #assert T.shape == (batch_size, nr)

        return N


##### Param Space ##### Param Space ##### Param Space ##### Param Space ##### Param Space ##### Param Space ##### Param Space ##### Param Space ##### Param Space ##### Param Space ##### Param Space ##### Param Space ##### Param Space ##### Param Space #####


@dataclass
class ParamDef:
    name: str
    low: float              # PROVIDED IN PHYSICAL (LINEAR) UNITS
    high: float             # PROVIDED IN PHYSICAL (LINEAR) UNITS
    scale: str = "linear"   # "linear" | "log10"

class ParamSpace:
    """
    You pass physical (linear) low/high for every parameter.
    If scale == "log10", we internally store log10(low/high) for sampling, but
    still expose/accept physical values at the edges (unit<->phys and clamping).
    """
    def __init__(self, params: List[ParamDef]):
        self.dtype  = torch.float64
        self.params = params
        self.names  = [p.name for p in params]
        self._scale = [p.scale for p in params]

        # Physical (linear) bounds as provided
        low_phys  = torch.tensor([p.low  for p in params], device=device, dtype=self.dtype)
        high_phys = torch.tensor([p.high for p in params], device=device, dtype=self.dtype)

        # Validate log10 params are strictly positive in physical space
        for j, sc in enumerate(self._scale):
            if sc == "log10":
                if not (low_phys[j] > 0 and high_phys[j] > 0):
                    raise ValueError(
                        f"Param '{self.names[j]}' has nonpositive bounds for log10 scale: "
                        f"low={float(low_phys[j])}, high={float(high_phys[j])}"
                    )

        # Internal “scaled” bounds:
        # - linear: same as physical
        # - log10 : log10(physical)
        low_scaled  = low_phys.clone()
        high_scaled = high_phys.clone()
        with torch.no_grad():
            for j, sc in enumerate(self._scale):
                if sc == "log10":
                    low_scaled[j]  = torch.log10(low_phys[j])
                    high_scaled[j] = torch.log10(high_phys[j])

        self._low_phys   = low_phys
        self._high_phys  = high_phys
        self._low_scaled = low_scaled
        self._high_scaled= high_scaled

    # Convenience properties
    @property
    def lows_phys(self):   return self._low_phys
    @property
    def highs_phys(self):  return self._high_phys
    @property
    def lows(self):        return self._low_scaled      # scaled space (lin or log10)
    @property
    def highs(self):       return self._high_scaled     # scaled space (lin or log10)

    @property
    def dim(self) -> int:
        return len(self.params)

    def unit_to_phys(self, u: torch.Tensor) -> torch.Tensor:
        """
        Map u in [0,1]^d -> physical space.
        For log10 params, interpolate in log10-space and exponentiate back.
        Accepts shape (d,) or (q,d). Returns same leading shape.
        """
        u = u.to(device=device, dtype=self.dtype).clamp(0, 1)
        batched = (u.ndim == 2)
        if not batched:
            u = u.unsqueeze(0)  # (1,d)

        outs = []
        for j, sc in enumerate(self._scale):
            lo_s, hi_s = self._low_scaled[j], self._high_scaled[j]
            uj = u[:, j]
            if sc == "linear":
                x_phys = self._low_phys[j] + uj * (self._high_phys[j] - self._low_phys[j])
            elif sc == "log10":
                x_log = lo_s + uj * (hi_s - lo_s)      # interpolate in log10
                x_phys = torch.pow(10.0, x_log)        # back to physical
            else:
                raise ValueError(f"Unknown scale: {sc}")
            outs.append(x_phys.unsqueeze(1))

        X = torch.cat(outs, dim=1)  # (q,d)
        return X if batched else X.squeeze(0)

    def phys_to_unit(self, x: torch.Tensor) -> torch.Tensor:
        """
        Map physical x -> u in [0,1]^d.
        For log10 params, convert x to log10(x) then linearly scale.
        Accepts shape (d,) or (q,d). Returns same leading shape, clamped to [0,1].
        """
        x = x.to(device=device, dtype=self.dtype)
        batched = (x.ndim == 2)
        if not batched:
            x = x.unsqueeze(0)  # (1,d)

        us = []
        for j, sc in enumerate(self._scale):
            lo_s, hi_s = self._low_scaled[j], self._high_scaled[j]
            xj = x[:, j]
            if sc == "linear":
                uj = (xj - self._low_phys[j]) / (self._high_phys[j] - self._low_phys[j])
            elif sc == "log10":
                # guard for tiny/negative due to numerical noise
                xj = torch.clamp(xj, min=1e-300)
                x_log = torch.log10(xj)
                uj = (x_log - lo_s) / (hi_s - lo_s)
            else:
                raise ValueError(f"Unknown scale: {sc}")
            us.append(uj.unsqueeze(1))
        U = torch.cat(us, dim=1).clamp(0, 1)
        return U if batched else U.squeeze(0)

    def clamp_phys(self, x: torch.Tensor) -> torch.Tensor:
        """
        Clamp physical x to physical bounds (uses linear low/high even if scale=log10).
        Preserves leading batch dims.
        """
        x = x.to(device=device, dtype=self.dtype)
        # support (..., d)
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])  # (n,d)
        outs = []
        for j, _ in enumerate(self._scale):
            outs.append(torch.clamp(x2[:, j], min=self._low_phys[j], max=self._high_phys[j]).unsqueeze(1))
        X = torch.cat(outs, dim=1).reshape(*lead, x.shape[-1])
        return X

    def dict_from_vector(self, x: torch.Tensor) -> Dict[str, float]:
        # stay off the GPU here
        x = x.to(dtype=self.dtype).flatten()
        vals = x.detach().cpu().tolist()
        return dict(zip(self.names, vals))

    

##### Pareto Archive ##### Pareto Archive ##### Pareto Archive ##### Pareto Archive ##### Pareto Archive ##### Pareto Archive ##### Pareto Archive ##### Pareto Archive ##### Pareto Archive ##### Pareto Archive ##### Pareto Archive


class ParetoArchive:
    """
    Minimal nondominated archive for k-objective minimization.
    Stores physical parameter dicts (X_phys) and objective vectors (F).
    """
    def __init__(self, k: int):
        self.k = k
        self.tol = 1e-8
        self.X_phys_all = []   # list[dict]
        self.F_all = []        # list[np.ndarray], each shape (k,)
        
        # live Pareto front only
        self.X_front = []          # list[dict]
        self.F_front = np.empty((0, k), dtype=float)

    def _dom_by_front(self, f):
        # any front point dominates f ?
        F = self.F_front
        if F.size == 0: return False
        le = (F <= f + self.tol).all(axis=1)
        lt = (F <  f - self.tol).any(axis=1)
        return np.any(le & lt)

    def _who_front_dominated_by(self, f):
        # which current front points are dominated by f ?
        F = self.F_front
        if F.size == 0: return np.zeros(0, dtype=bool)
        le = (f <= F + self.tol).all(axis=1)
        lt = (f <  F - self.tol).any(axis=1)
        return le & lt

    # ----- public API -----
    def add(self, x_phys_dict: dict, f_vec_np: np.ndarray):
        f = np.asarray(f_vec_np, dtype=float)
        if not np.all(np.isfinite(f)):  # ignore non-finite
            return
        self.X_phys_all.append(x_phys_dict)
        self.F_all.append(f)

        # 1) if dominated by current front -> discard for front
        if self._dom_by_front(f):
            return

        # 2) drop any front points dominated by the new one
        kill = self._who_front_dominated_by(f)
        if kill.any():
            self.F_front = self.F_front[~kill]
            self.X_front = [x for x, keep in zip(self.X_front, ~kill) if keep]

        # 3) (optional) de-dup: if equal to an existing point, skip append
        #if self.F_front.size and np.any(np.all(np.isclose(self.F_front, f, atol=self.tol), axis=1)):
        #    return

        # 4) add to front
        self.F_front = np.vstack([self.F_front, f])
        self.X_front.append(x_phys_dict)

    def add_batch(self, X_list, F_mat):
        if not X_list: return
        F = np.asarray(F_mat, dtype=float)
        finite = np.isfinite(F).all(axis=1)
        for xd, fv in zip([X_list[i] for i in np.where(finite)[0]], F[finite]):
            self.add(xd, fv)

    def front(self):
        # already maintained
        return list(self.X_front), self.F_front.copy()

    def size_front(self):
        # O(1)
        return len(self.X_front) 
    

##### Build Scene ##### ##### Build Scene ##### Build Scene ##### Build Scene ##### Build Scene ##### Build Scene ##### Build Scene ##### Build Scene ##### Build Scene ##### Build Scene ##### Build Scene ##### Build Scene ##### Build Scene ##### Build Scene ##### Build Scene #####


class SceneBuilder:
    def __init__(self, rho: torch.Tensor, time: torch.Tensor):
        """
        rho: (nr,) tensor of spatial grid points
        time: (nt,) tensor of time grid points
        """
        self.rho = rho.to(device)
        self.time = time.to(device)
        self._rho_np  = rho.detach().cpu().numpy()
        self._time_np = time.detach().cpu().numpy()
        self.dtype = torch.float64

    def build_case(self, param_dict: Dict[str, float]) -> Dict[str, torch.Tensor]:
        """
        Given a parameter dictionary, build the case dict for solver.
        param_dict: dict with keys matching parameter names
        Returns case dict with keys 'rho', 'time', and any parameters.
        """
        #Spatial Source
        A_SR = param_dict['A_SR'];   Rc_SR = param_dict['Rc_SR'];   o_SR = param_dict['o_SR']
        SR = A_SR*torch.exp(-((self.rho - Rc_SR)/o_SR)**2) #! 72.9s / 4090.7s
  
        #Temporal Source
        A_ST = param_dict['A_ST'];   f_ST = param_dict['f_ST'];   p_ST = param_dict['p_ST']
        ST = A_ST*torch.sin(2*np.pi*f_ST*self.time + p_ST) + 1.0  # ensure non-negative  #! 59.2s / 4090.7s
  
        #Diffusion Coefficient
        A_D = param_dict['A_D'];   AL_D = param_dict['AL_D'];   A0_D = param_dict['A0_D']
        Rc_D = param_dict['Rc_D'];   RwL_D = param_dict['RwL_D'];   RwR_D = param_dict['RwR_D']
        Dknots = torch.tensor([0.0, Rc_D - RwL_D, Rc_D, Rc_D + RwR_D], device=device, dtype=self.dtype)
        Dvals = A_D*torch.tensor([AL_D, AL_D, A0_D, 1], device=device, dtype=self.dtype)
        D_interp = PchipInterpolator(Dknots.detach().cpu().numpy(), Dvals.detach().cpu().numpy(), extrapolate=True) #! 105.8s / 4090.7s
        D = torch.as_tensor(D_interp(self._rho_np), device=device, dtype=self.dtype)

        #Convection Profile (Flip inverts, default positive) (Bounce ensures all the same sign)
        A_V = param_dict['A_V'];   A1_V = param_dict['A1_V'];   A2_V = param_dict['A2_V']
        R1_V = param_dict['R1_V'];   R2_V = param_dict['R2_V'];   R3_V = param_dict['R3_V']
        Flip_V = param_dict['Flip_V']
        Bounce_V = param_dict['Bounce_V']
        Vknots = torch.tensor([0.0, R1_V, R2_V, R3_V], device=device, dtype=self.dtype)
        Vvals = A_V*torch.tensor([0, A1_V, A2_V, -1], device=device, dtype=self.dtype)
        V_interp = PchipInterpolator(Vknots.detach().cpu().numpy(), Vvals.detach().cpu().numpy(), extrapolate=True) #! 102.9s / 4090.7s
        V = torch.as_tensor(V_interp(self._rho_np), device=device, dtype=self.dtype)
        if Bounce_V > 0:
            V = V.abs()
        V = V*(-1 if Flip_V < 0 else 1)
        
        #Initial Density
        A_N0 = param_dict['A_N0'];   Xs_N0 = param_dict['Xs_N0'];   H_N0 = param_dict['H_N0']
        a_N0 = param_dict['a_N0'];   B_N0 = param_dict['B_N0']
        N0Funca = lambda r, a: ((1 + a*r)*torch.exp(r) - torch.exp(-r))/(torch.exp(r) + torch.exp(-r))
        N0Funcb = lambda r, xs, H: (xs - r)/H
        N0Func = lambda r, a, xs, H, A, B: A*N0Funca(N0Funcb(r,xs,H),a) + B
        N0 = N0Func(self.rho, a_N0, Xs_N0, H_N0, A_N0, B_N0) #! 187s / 4090.7s
        
        #Edge Boundary Condition
        if param_dict['A_sign'] >= 0:
            A_sign = 1.0
        else:
            A_sign = -1.0
        A_edge = torch.tensor([A_sign*param_dict['A_mag']], device=device, dtype=self.dtype)
        
        #Construct case dict
        case = {
            'rho': self.rho.unsqueeze(0),
            'time': self.time.unsqueeze(0),
            'SR': SR.unsqueeze(0),
            'ST': ST.unsqueeze(0),
            'D': D.unsqueeze(0),
            'V': V.unsqueeze(0),
            'N0': N0.unsqueeze(0),
            'A': A_edge
        }
        
        return case
    
    def build_batch(self, plist: List[Dict[str, float]]) -> Dict[str, torch.Tensor]:
        #Build to get shapes
        B = len(plist)
        # --- 1. Stack Parameters (CPU to GPU once) ---
        # Do this for all parameters you need
        A_SR = torch.tensor([p['A_SR'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        Rc_SR = torch.tensor([p['Rc_SR'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        o_SR = torch.tensor([p['o_SR'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        
        A_ST = torch.tensor([p['A_ST'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        f_ST = torch.tensor([p['f_ST'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        p_ST = torch.tensor([p['p_ST'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        
        # ... stack parameters for D ...
        A_D = torch.tensor([p['A_D'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        AL_D = torch.tensor([p['AL_D'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        A0_D = torch.tensor([p['A0_D'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        Rc_D = torch.tensor([p['Rc_D'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        RwL_D = torch.tensor([p['RwL_D'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        RwR_D = torch.tensor([p['RwR_D'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)

        # ... stack parameters for N0 ...
        A_N0 = torch.tensor([p['A_N0'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        Xs_N0 = torch.tensor([p['Xs_N0'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        H_N0 = torch.tensor([p['H_N0'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        a_N0 = torch.tensor([p['a_N0'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        B_N0 = torch.tensor([p['B_N0'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        
        # ... stack parameters for V (make sure to include Flip_V, Bounce_V) ...
        A_V = torch.tensor([p['A_V'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        A1_V = torch.tensor([p['A1_V'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        A2_V = torch.tensor([p['A2_V'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        R1_V = torch.tensor([p['R1_V'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        R2_V = torch.tensor([p['R2_V'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        R3_V = torch.tensor([p['R3_V'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        Flip_V = torch.tensor([p['Flip_V'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)   
        Bounce_V = torch.tensor([p['Bounce_V'] for p in plist], device=device, dtype=self.dtype).unsqueeze(1)
        
        # ... stack parameters for A (edge) ...
        A_sign = torch.sign(torch.tensor([p['A_sign'] for p in plist], device=device, dtype=self.dtype))
        A_mag = torch.tensor([p['A_mag'] for p in plist], device=device, dtype=self.dtype)
        A = A_sign * A_mag

        # --- 2. Vectorize Calculations (All on GPU) ---
        
        # `self.rho` (nr,) broadcasts with (B, 1) params to create (B, nr)
        SR = A_SR * torch.exp(-((self.rho - Rc_SR) / o_SR)**2)
        
        # `self.time` (nt,) broadcasts with (B, 1) params to create (B, nt)
        ST = A_ST * torch.sin(2 * np.pi * f_ST * self.time + p_ST) + 1.0

        # N0Func is already pure-torch, so it will broadcast
        N0Funca = lambda r, a: ((1 + a*r)*torch.exp(r) - torch.exp(-r))/(torch.exp(r) + torch.exp(-r))
        N0Funcb = lambda r, xs, H: (xs - r)/H
        N0Func = lambda r, a, xs, H, A, B: A*N0Funca(N0Funcb(r,xs,H),a) + B
        N0 = N0Func(self.rho, a_N0, Xs_N0, H_N0, A_N0, B_N0) # This is now (B, nr)

        # --- 3. Solve Interpolation (The Hard Part) ---
        # You MUST replace the SciPy interpolator.
        
        # Build batched knot and value tensors
        # Shape will be (B, 4)
        zeros = torch.zeros_like(Rc_D)
        ones = torch.ones_like(Rc_D)
        
        Dknots = torch.cat([zeros, Rc_D - RwL_D, Rc_D, Rc_D + RwR_D], dim=1)
        Dvals = A_D * torch.cat([AL_D, AL_D, A0_D, ones], dim=1)

        D_interp = PytorchPchipInterpolator(Dknots, Dvals)  # You need to implement this class
        D = D_interp(self.rho)  # interpolates self.rho for each B, output (B, nr)
        
        Vknots = torch.cat([zeros, R1_V, R2_V, R3_V], dim=1)
        Vvals = A_V * torch.cat([zeros, A1_V, A2_V, -ones], dim=1)
        
        V_interp = PytorchPchipInterpolator(Vknots, Vvals)  # You need to implement this class
        V = V_interp(self.rho)  # interpolates self.rho for each B, output (B, nr)

        V = torch.where(Bounce_V > 0, V.abs(), V)
        # Optimized flip logic (avoids creating large tensors)
        V = torch.where(Flip_V < 0, -V, V)
        
        # --- 3. Construct case dict ---
        case = {
            'rho': self.rho.expand(B, -1),
            'time': self.time.expand(B, -1),
            'SR': SR,
            'ST': ST,
            'D': D,
            'V': V,
            'N0': N0,
            'A': A
        }
        
        return case
    

##### Objectives ##### Objectives ##### Objectives ##### Objectives ##### Objectives ##### Objectives ##### Objectives ##### Objectives ##### Objectives ##### Objectives ##### Objectives ##### Objectives ##### Objectives #####


# ---- your metrics, batched over the leading dim ----
def make_eval_vector_batched(N_cap_all: float, N_cap_edge: float, N_min_core: float) -> Callable:
    @torch.no_grad()
    def eval_batched(case: Dict[str, torch.Tensor], solver) -> torch.Tensor:
        # case tensors should already have batch dim (B, ...)
        N = solver.solve(**case, conv_bc_outer='dirichlet',
                         assert_conservation=False, dtype=case["rho"].dtype)
        # Metrics:
        # f1: max density exceedance (>=0), 0 if not exceeding
        # f2: edge density exceedance (>=0), 0 if not exceeding
        # f3: total negative density magnitude (>=0)
        B = N.shape[0]
        # f1
        max_all = N.max(dim=2).values.max(dim=1).values  # (B,)
        f1 = torch.clamp(max_all - N_cap_all, min=0.0)
        # f2: last rho index assumed "edge"
        edge = N[:, -1, :]  # (B, T)
        max_edge = edge.max(dim=1).values                 # (B,)
        f2 = torch.clamp(max_edge - N_cap_edge, min=0.0)
        # f3: sum of negative mass (magnitude)
        neg = torch.clamp(-N, min=0.0)
        f3 = neg.sum(dim=(1, 2))  # (B,)
        # f4: core minimum density violation
        min_core = N[:, 0, :].min(dim=1).values  # (B,)
        f4 = torch.clamp(N_min_core - min_core, min=0.0)
        F = torch.stack([f1, f2, f3, f4], dim=1)  # (B, 3)
        
        # replace non-finite rows with a big penalty (keeps shape)
        bad = ~torch.isfinite(F).all(dim=1)
        if bad.any():
            F[bad] = torch.tensor([1e6, 1e6, 1e12], device=F.device, dtype=F.dtype)
        return F
    return eval_batched

# tiny adapter for single-case usage (Phase B / TuRBO):
def make_eval_vector_single_from_batched(batched_fn: Callable) -> Callable:
    @torch.no_grad()
    def eval_single(case: Dict[str, torch.Tensor], solver) -> torch.Tensor:
        # case from SceneBuilder already has batch dim 1 → shapes are (1, nr), (1, nt), ...
        F = batched_fn(case, solver)  # (1, K)
        return F.squeeze(0)           # (K,)
    return eval_single

def make_eval_vector(N_cap_all: float, N_cap_edge: float, N_min_core: float, std_max = 5e-3,
                     neg_norm_mode: str = "t0_mean",  # "t0_mean"|"cap_all"|"const"
                     neg_norm_const: float = 1.0):
    """
    Returns a placeholder callable for compatibility with per-sample evals,
    and carries caps/normalization settings as attributes for the batched path.
    """
    @torch.no_grad()
    def _dummy(case, solver):
        raise RuntimeError("Per-sample eval not used in Phase A; using batched evaluator.")
    _dummy.N_cap_all  = float(N_cap_all)
    _dummy.N_cap_edge = float(N_cap_edge)
    _dummy.N_min_core= float(N_min_core)
    _dummy.std_max    = float(std_max)
    _dummy.neg_norm_mode  = str(neg_norm_mode)
    _dummy.neg_norm_const = float(neg_norm_const)
    return _dummy

@torch.no_grad()
def evaluate_batch_batched(
    space, builder, solver, U_batch: torch.Tensor, eval_vec,
    batch_size: int = 128, penalty_val: float = 1e6, verbose: bool = False
):
    """
    Row-wise, batch-stable metrics (minimize all; dimensionless):
      f1 = max(0, (max_r,t N - N_cap_all)/N_cap_all)
      f2 = max(0, (max_t N(edge) - N_cap_edge)/N_cap_edge)
      f3 = sum_{r,t} max(0, -N) / (N_ref * nr * nt)
          with N_ref from eval_vec.neg_norm_mode ("t0_mean"|"cap_all"|"const")
    """
    # Pull settings off eval_vec
    N_cap_all  = float(getattr(eval_vec, "N_cap_all", 1e21))
    N_cap_edge = float(getattr(eval_vec, "N_cap_edge", 1e20))
    N_min_core= float(getattr(eval_vec, "N_min_core", 2e20))
    std_max    = float(getattr(eval_vec, "std_max", 5e-3))
    neg_norm_mode  = str(getattr(eval_vec, "neg_norm_mode", "t0_mean"))
    neg_norm_const = float(getattr(eval_vec, "neg_norm_const", 1.0))

    X = space.unit_to_phys(U_batch)  # (q,d)
    plist_all, F_all = [], []

    def metrics_from_N(N: torch.Tensor) -> torch.Tensor:
        B, nr, nt = N.shape
        N64 = N.to(torch.float64)
        finite_mask = torch.isfinite(N64).all(dim=2).all(dim=1)  # (B,)
        F = torch.full((B, 5), penalty_val, dtype=torch.float64, device=N.device)

        if finite_mask.any():
            Ns = N64[finite_mask]                           # (Bs,nr,nt)
            gmax = Ns.amax(dim=(1, 2))                      # (Bs,)
            f1 = torch.clamp((gmax - N_cap_all) / max(N_cap_all, 1e-30), min=0.0)

            emax = Ns[:, -1, :].amax(dim=1)                 # (Bs,)
            f2 = torch.clamp((emax - N_cap_edge) / max(N_cap_edge, 1e-30), min=0.0)

            neg_sum = (-torch.minimum(Ns, torch.zeros((), dtype=Ns.dtype, device=Ns.device))).sum(dim=(1, 2))

            if neg_norm_mode == "t0_mean":
                N_ref = Ns[:, :, 0].mean(dim=1).abs().clamp_min(1e-30)  # per-row
            elif neg_norm_mode == "cap_all":
                N_ref = torch.full_like(neg_sum, N_cap_all)
            elif neg_norm_mode == "const":
                N_ref = torch.full_like(neg_sum, neg_norm_const).clamp_min(1e-30)
            else:
                raise ValueError(f"Unknown neg_norm_mode: {neg_norm_mode}")

            f3 = neg_sum / (N_ref * (nr * nt))
            
            f4 = torch.clamp((N_min_core - Ns[:, 0, :].amin(dim=1)) / max(N_min_core, 1e-30), min=0.0)
            
            _ = Ns[:, -int(0.05*nr):, -1]/Ns[:, :, 0].amax(dim=1).unsqueeze(1)
            stds_torch = _.std(dim=1, unbiased=False)  # Add unbiased=False to match np.std
            f5 = torch.clamp((stds_torch - std_max) / max(std_max, 1e-30), min=0.0)
            
            F[finite_mask] = torch.stack([f1, f2, f3, f4, f5], dim=1)

        return F  # (B,5)

    q = X.shape[0]
    for s in range(0, q, batch_size):
        Xe = X[s:s+batch_size]
        plist = [space.dict_from_vector(Xe[i]) for i in range(Xe.shape[0])]
        caseB = builder.build_batch(plist)  # must return batched (B,·) tensors

        try:
            N = solver.solve(
                **caseB, conv_bc_outer='dirichlet',
                assert_conservation=False, dtype=caseB["rho"].dtype
            )  # (B,nr,nt)
            #! This and caseB = ...
            #! 1553.57s / 4090.7s

            # Row-wise finiteness diag (won’t spam if all finite)
            finite_mask = torch.isfinite(N).all(dim=2).all(dim=1)
            bad = (~finite_mask).sum().item()
            if bad:
                print(f"\n[diag] non-finite rows: {bad}/{N.shape[0]}, max|N|={float(N.abs().amax())}")

            F = metrics_from_N(N)

        except Exception as e:
            dev = caseB["rho"].device
            print(f"\n[diag] solver exception on chunk: {repr(e)}")
            F = torch.full((Xe.shape[0], 5), penalty_val, dtype=torch.float64, device=dev)

        plist_all.extend(plist)
        F_all.append(F.detach().cpu().numpy())

    return plist_all, np.vstack(F_all)


##### TPE GMM ##### TPE GMM ##### TPE GMM ##### TPE GMM ##### TPE GMM ##### TPE GMM ##### TPE GMM ##### TPE GMM ##### TPE GMM ##### TPE GMM ##### TPE GMM ##### TPE GMM ##### TPE GMM #####


# ---------- helpers ----------
def _logit(u: torch.Tensor, eps=1e-6):
    u = u.clamp(eps, 1 - eps)
    return torch.log(u) - torch.log(1 - u)

def _silverman_bandwidth(z: torch.Tensor, min_h=1e-4):
    std = z.std(dim=0, unbiased=True).clamp_min(1e-6)
    n   = z.shape[0]
    h   = (1.06 * std * (n ** (-1/5))) if n > 1 else torch.ones_like(std)
    return h.clamp_min(min_h)

def _temper_to_interior(U: torch.Tensor, tau=0.90, eps=1e-6):
    U = U.clamp(eps, 1 - eps)
    return (U**tau) / (U**tau + (1 - U)**tau)

# ---------- multivariate KDE (mixture of correlated Gaussians in logit space) ----------
@torch.no_grad()
def fit_gmm_logit(U: torch.Tensor, idx: torch.Tensor, n_components: int = 20, reg_covar: float = 1e-6):
    '''
    Fits a Gaussian Mixture Model in logit-space.
    '''
    d = U.shape[1]
    
    # Not enough points to fit a GMM
    if idx.numel() < d + 1:
        print(f"[fit_gmm_logit] Warning: Not enough points to fit GMM (need at least {d+1}, got {idx.numel()}). Returning None.")
        return None
    elif idx.numel() < n_components:
        n_components = idx.numel()
        print(f"[fit_gmm_logit] Warning: Reducing n_components to {n_components} due to insufficient points.")

    Z_torch = _logit(U[idx])
    Z_np = Z_torch.cpu().numpy() # sklearn requires numpy

    # Fit the GMM
    try:
        gmm = GaussianMixture(
            n_components=min(n_components, Z_np.shape[0]),
            covariance_type='full',  # For parameter dependencies
            reg_covar=reg_covar,     # Stability
            random_state=0         # Reproducibility
        ).fit(Z_np)
        
        # Store the fitted model
        return {"gmm_model": gmm, "d": d}
        
    except Exception as e:
        print(f"[fit_gmm_logit] Warning: GMM fit failed ({e}). Returning None.")
        return None


@torch.no_grad()
def gmm_sample(pack: dict, n: int) -> torch.Tensor:
    '''
    Sample from the GMM in logit-space, then transform back to unit-space.
    '''
    if pack is None or "gmm_model" not in pack:
        d = pack.get("d", 2) # Fallback dimension
        z = torch.randn((n, d), device=device, dtype=torch.float64)
        return _temper_to_interior(torch.sigmoid(z))

    gmm_model = pack["gmm_model"]
    
    # Sample in logit-space
    Z_samples_np, _ = gmm_model.sample(n)
    
    # Convert back to torch tensor and unit-space
    Z_samples_torch = torch.tensor(Z_samples_np, device=device, dtype=torch.float64)
    U_samples = torch.sigmoid(Z_samples_torch)
    
    return _temper_to_interior(U_samples)


@torch.no_grad()
def gmm_logpdf(U: torch.Tensor, pack: dict) -> torch.Tensor:
    '''
    Calculate the log-probability density of U under the GMM.
    (Jacobian cancels in ℓ/g ratio; we omit it.)
    '''
    d = U.shape[1]
    if pack is None or "gmm_model" not in pack:
        # Fallback log-pdf (standard normal in logit-space)
        Z = _logit(U)
        return -0.5 * (Z**2).sum(dim=1) - 0.5 * d * math.log(2 * math.pi)

    gmm_model = pack["gmm_model"]
    
    # Transform U to logit-space and evaluate
    Z_torch = _logit(U)
    Z_np = Z_torch.cpu().numpy()
    
    # Get log-probabilities
    log_probs_np = gmm_model.score_samples(Z_np)
    
    return torch.tensor(log_probs_np, device=device, dtype=torch.float64)

# ---------- multi-objective split (GOOD/BAD) ----------
def split_good_bad_allzero(
    U_all: torch.Tensor,
    F_all: torch.Tensor,
    tol: float = 1e-12,     # tolerance for "zero"
    good_frac: float = 1.0, # 1.0 → use *all* feasible; else random top-X% of feasible
):
    # keep only finite rows
    finite = torch.isfinite(F_all).all(dim=1)
    U_all, F_all = U_all[finite], F_all[finite]
    n = U_all.shape[0]
    if n == 0:
        dev = U_all.device
        return (torch.empty(0, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.long, device=dev))

    # feasible ⇔ all metrics ~ 0
    feas_mask = (F_all.abs() <= tol).all(dim=1)   # metrics nonneg, so abs is safe
    feas_idx  = torch.nonzero(feas_mask).flatten()
    infeas_idx= torch.nonzero(~feas_mask).flatten()

    if feas_idx.numel() == 0:
        # no feasible yet → GOOD empty, BAD = everything
        dev = U_all.device
        return (torch.empty(0, dtype=torch.long, device=dev),
                torch.arange(n, device=dev, dtype=torch.long))

    # choose GOOD set: either all feasible or a random X% subset (they’re all equal on F)
    if good_frac >= 1.0 or feas_idx.numel() <= 2:
        good_idx = feas_idx
    else:
        k = max(2, int(round(good_frac * feas_idx.numel())))
        perm = torch.randperm(feas_idx.numel(), device=feas_idx.device)
        good_idx = feas_idx[perm[:k]]

    bad_idx = infeas_idx
    return good_idx, bad_idx

# ---------- TPE proposal: draw from ℓ, rank by ℓ/g ----------
@torch.no_grad()
def tpe_gmm_propose(space, good_pack, bad_pack, n_draw=2048, top_q=128, explore_p=0.20):
    d = space.dim
    # fallback or forced exploration
    need_fallback = (good_pack is None) or (bad_pack is None)
        
    if need_fallback or (torch.rand(()) < explore_p):
        U = torch.rand(n_draw, d, device=space.lows.device, dtype=space.lows.dtype)
        return _temper_to_interior(U)[:min(top_q, n_draw)]

    Ucand = gmm_sample(good_pack, n_draw)
    logl  = gmm_logpdf(Ucand, good_pack)
    logg  = gmm_logpdf(Ucand, bad_pack)
    score = logl - logg
    top_idx = torch.topk(score, k=min(top_q, n_draw)).indices
    return Ucand[top_idx]


##### TPE KDE ##### TPE KDE ##### TPE KDE ##### TPE KDE ##### TPE KDE ##### TPE KDE ##### TPE KDE ##### TPE KDE ##### TPE KDE ##### TPE KDE ##### TPE KDE ##### TPE KDE ##### TPE KDE #####


# ---------- helpers ----------
def _logit(u: torch.Tensor, eps=1e-6):
    u = u.clamp(eps, 1 - eps)
    return torch.log(u) - torch.log(1 - u)

def _silverman_bandwidth(z: torch.Tensor, min_h=1e-4):
    std = z.std(dim=0, unbiased=True).clamp_min(1e-6)
    n   = z.shape[0]
    h   = (1.06 * std * (n ** (-1/5))) if n > 1 else torch.ones_like(std)
    return h.clamp_min(min_h)

def _temper_to_interior(U: torch.Tensor, tau=0.90, eps=1e-6):
    U = U.clamp(eps, 1 - eps)
    return (U**tau) / (U**tau + (1 - U)**tau)

# ---------- multivariate KDE (mixture of correlated Gaussians in logit space) ----------
@torch.no_grad()
def fit_mv_kde_logit(U: torch.Tensor, idx: torch.Tensor,
                     jitter=1e-6, min_h=1e-4, shrink=0.05):
    d = U.shape[1]
    I = torch.eye(d, device=U.device)
    # not enough points to fit a correlated KDE → tell caller to fallback
    if idx.numel() < 3:
        return None

    Z = _logit(U[idx])                  # (m,d)
    Zc = Z - Z.mean(dim=0, keepdim=True)
    n = Zc.shape[0]

    cov = (Zc.T @ Zc) / max(1, n)
    tr  = torch.trace(cov) / d
    cov = (1 - shrink) * cov + shrink * tr * I   # shrinkage
    cov = cov + jitter * I                       # jitter

    evals, evecs = torch.linalg.eigh(cov)
    evals = evals.clamp_min(1e-10)               # PSD clamp
    cov_psd = (evecs * evals) @ evecs.T
    L = torch.linalg.cholesky(cov_psd)

    h = Zc.std(dim=0, unbiased=False).clamp_min(min_h)
    return {"Z": Z, "mu": Z.mean(dim=0), "L": L, "h": h}


@torch.no_grad()
def mv_kde_sample(pack: dict, n: int) -> torch.Tensor:
    # empty / None handled by caller    
    Zc = pack["Z"]
    d  = pack["mu"].numel()
    if Zc.numel() == 0:
        z = torch.randn((n, d), device=Zc.device)
        return _temper_to_interior(torch.sigmoid(z))

    L, h = pack["L"], pack["h"]
    m = Zc.shape[0]
    idx = torch.randint(0, m, (n,), device=Zc.device)
    z0  = Zc[idx]
    xi  = torch.randn((n, d), device=Zc.device, dtype=Zc.dtype)
    HL  = torch.diag(h) @ L
    z   = z0 + (xi @ HL.t())
    return _temper_to_interior(torch.sigmoid(z))


@torch.no_grad()
def mv_kde_logpdf(U: torch.Tensor, pack: dict) -> torch.Tensor:
    """
    log p(u) = log mean_i N(z | z_i, Σ); mixture over kernel centers.
    (Jacobian cancels in ℓ/g ratio; we omit it.)
    """
    Z   = _logit(U)                     # (m,d)
    Zc  = pack["Z"]                     # (n,d)
    d   = Z.shape[1]
    if Zc.numel() == 0:
        return -0.5 * (Z**2).sum(dim=1) - 0.5 * d * math.log(2 * math.pi)

    L   = pack["L"]
    h   = pack["h"]
    Hinv = torch.diag(1.0 / h)

    const = -0.5 * d * math.log(2 * math.pi) - torch.log(h).sum()
    m, n = Z.shape[0], Zc.shape[0]
    chunk = max(1, 4096 // d)
    lse_terms = []
    for s in range(0, n, chunk):
        Zs = Zc[s:s+chunk]                         # (c,d)
        dz = Z.unsqueeze(1) - Zs.unsqueeze(0)      # (m,c,d)
        # y = L^{-1} H^{-1} dz  via triangular solve
        y  = torch.linalg.solve_triangular(L, (dz @ Hinv.t()).transpose(1,2), upper=False).transpose(1,2)  # (m,c,d)
        quad = 0.5 * (y**2).sum(dim=2)            # (m,c) #! 57.65s / 4090.7s
        lp = const - quad                          # (m,c)
        lse_terms.append(lp)
    lp_all = torch.cat(lse_terms, dim=1)           # (m,n)
    return torch.logsumexp(lp_all, dim=1) - math.log(n)

# ---------- multi-objective split (GOOD/BAD) ----------
def split_good_bad_allzero(
    U_all: torch.Tensor,
    F_all: torch.Tensor,
    tol: float = 1e-12,     # tolerance for "zero"
    good_frac: float = 1.0, # 1.0 → use *all* feasible; else random top-X% of feasible
):
    # keep only finite rows
    finite = torch.isfinite(F_all).all(dim=1)
    U_all, F_all = U_all[finite], F_all[finite]
    n = U_all.shape[0]
    if n == 0:
        dev = U_all.device
        return (torch.empty(0, dtype=torch.long, device=dev),
                torch.empty(0, dtype=torch.long, device=dev))

    # feasible ⇔ all metrics ~ 0
    feas_mask = (F_all.abs() <= tol).all(dim=1)   # metrics nonneg, so abs is safe
    feas_idx  = torch.nonzero(feas_mask).flatten()
    infeas_idx= torch.nonzero(~feas_mask).flatten()

    if feas_idx.numel() == 0:
        # no feasible yet → GOOD empty, BAD = everything
        dev = U_all.device
        return (torch.empty(0, dtype=torch.long, device=dev),
                torch.arange(n, device=dev, dtype=torch.long))

    # choose GOOD set: either all feasible or a random X% subset (they’re all equal on F)
    if good_frac >= 1.0 or feas_idx.numel() <= 2:
        good_idx = feas_idx
    else:
        k = max(2, int(round(good_frac * feas_idx.numel())))
        perm = torch.randperm(feas_idx.numel(), device=feas_idx.device)
        good_idx = feas_idx[perm[:k]]

    bad_idx = infeas_idx
    return good_idx, bad_idx

# ---------- TPE proposal: draw from ℓ, rank by ℓ/g ----------
@torch.no_grad()
def tpe_mv_propose(space, good_pack, bad_pack, n_draw=2048, top_q=128, explore_p=0.20):
    d = space.dim
    # fallback or forced exploration
    need_fallback = (good_pack is None) or (bad_pack is None) \
        or (good_pack["Z"].numel() == 0) or (bad_pack["Z"].numel() == 0)
    if need_fallback or (torch.rand(()) < explore_p):
        U = torch.rand(n_draw, d, device=space.lows.device, dtype=space.lows.dtype)
        return _temper_to_interior(U)[:min(top_q, n_draw)]

    Ucand = mv_kde_sample(good_pack, n_draw)
    logl  = mv_kde_logpdf(Ucand, good_pack)
    logg  = mv_kde_logpdf(Ucand, bad_pack)
    score = logl - logg #! 39.1s /  4090.7s
    top_idx = torch.topk(score, k=min(top_q, n_draw)).indices
    return Ucand[top_idx]
    
    
##### phase_a ##### phase_a ##### phase_a ##### phase_a ##### phase_a ##### phase_a ##### phase_a ##### phase_a ##### phase_a ##### phase_a ##### phase_a ##### phase_a #####


def sobol_batch(q, d):
    sob = torch.quasirandom.SobolEngine(dimension=d, scramble=True)
    return sob.draw(q).to(device)

def run_phase_A_mv_tpe(space,
                       builder,
                       solver,
                       eval_vec,# callable(case, solver) -> torch(k,)
                       warmup_batches=50, 
                       warmup_q=128,
                       tpe_rounds=12,
                       tpe_draws=2048,
                       tpe_top_q=128,
                       verbose=True,
                       log=True,
                       method='KDE',
                       gmm_components=50,
                       ):
    """
    Multi-objective Phase A:
      1) Sobol warmup (2-3 batches)
      2) Multivariate TPE (mixture GMM in logit space) with GOOD/BAD from Pareto
    Keeps and returns a Pareto archive.
    """
    # detect k from first eval later
    archive = None
    U_buf, F_buf = [], []   # unit samples and objective tensors
    tol_feas = 1e-12
    good_total = 0
    # --- Sobol warmup ---
    # 1. Determine warmup target based on method
    min_good_points_target = space.dim + 1  # default for KDE
        
    if log:
        print(f"[Warmup] Method: {method}")
        print(f"[Warmup] Running Sobol batches until {min_good_points_target} good points are found (batches: {warmup_batches}).")
        
    b = 0 
    
    # 2. Run loop until target is met.
    while good_total < min_good_points_target:
        b += 1
        if b > warmup_batches:
            if log:
                print(f"[Warmup] WARNING: Hit {warmup_batches} batches, but only found {good_total} good points. Continuing anyway.")
            if good_total == 0:
                 print("[Warmup] ERROR: Found 0 good points. TPE will fail. Exiting.")
                 return None # TPE cannot start with 0 good points
                

        U = sobol_batch(warmup_q, space.dim)
        plist, F_np = evaluate_batch_batched(space, builder, solver, U, eval_vec, batch_size=warmup_q, verbose=verbose)
        
        if archive is None:
            k = F_np.shape[1]
            archive = ParetoArchive(k=k)
        
        archive.add_batch(plist, F_np)
        U_buf.append(U)
        F_buf.append(torch.tensor(F_np, device=space.lows.device, dtype=torch.float64))
        
        feas_in_batch = int(((np.abs(F_np) <= tol_feas).all(axis=1)).sum())
        good_total += feas_in_batch

        
        if log:
            print(f"[warmup batch {b}]: Feasible = {good_total}  |  Pareto size = {archive.size_front()}/{min_good_points_target}")
    
    # --- TPE rounds ---
    # --- TPE rounds ---
    for r in range(1, tpe_rounds + 1):
        U_all = torch.vstack(U_buf)
        F_all = torch.vstack(F_buf)

        good_idx, bad_idx = split_good_bad_allzero(
            U_all, F_all, tol=tol_feas, good_frac=1.0   # use all feasible
        )

        if method == 'GMM':
            # subsample for tractability
            tpe_gmm_max_b = 50_000
            tpe_gmm_max_g = 5_000_000
            if bad_idx.numel() > tpe_gmm_max_b:
                #if log:
                    #print(f"[TPE {r:02d}/{tpe_rounds}] Subsampling BAD {bad_idx.numel()} → {tpe_gmm_max_b}")
                bad_idx = bad_idx[torch.randperm(bad_idx.numel(), device=bad_idx.device)[:tpe_gmm_max_b]]
            if good_idx.numel() > tpe_gmm_max_g:
                if log:
                    print(f"[TPE {r:02d}/{tpe_rounds}] Subsampling GOOD {good_idx.numel()} → {tpe_gmm_max_g}")
                good_idx = good_idx[torch.randperm(good_idx.numel(), device=good_idx.device)[:tpe_gmm_max_g]]

            good_pack = fit_gmm_logit(U_all, good_idx, n_components=gmm_components)
            bad_pack  = fit_gmm_logit(U_all, bad_idx,  n_components=gmm_components)
            Ucand = tpe_gmm_propose(space, good_pack, bad_pack,
                                    n_draw=tpe_draws, top_q=tpe_top_q)

        else:  # 'KDE'
            # subsample for tractability
            tpe_kde_max_b = 5_000
            tpe_kde_max_g = 100_000
            if bad_idx.numel() > tpe_kde_max_b:
                #if log:
                    #print(f"[TPE {r:02d}/{tpe_rounds}] Subsampling BAD {bad_idx.numel()} → {tpe_kde_max_b}")
                bad_idx = bad_idx[torch.randperm(bad_idx.numel(), device=bad_idx.device)[:tpe_kde_max_b]]
            if good_idx.numel() > tpe_kde_max_g:
                if log:
                    print(f"[TPE {r:02d}/{tpe_rounds}] Subsampling GOOD {good_idx.numel()} → {tpe_kde_max_g}")
                good_idx = good_idx[torch.randperm(good_idx.numel(), device=good_idx.device)[:tpe_kde_max_g]]

            good_pack = fit_mv_kde_logit(U_all, good_idx)
            bad_pack  = fit_mv_kde_logit(U_all, bad_idx)
            Ucand = tpe_mv_propose(space, good_pack, bad_pack,
                                   n_draw=tpe_draws, top_q=tpe_top_q)

        if (good_pack is None) or (bad_pack is None):
            if log:
                print(f"[TPE {r:02d}/{tpe_rounds}] Fallback to uniform (insufficient elites)")

        #!Nudge Towards Center
        def nudge_interior(U, eps=0.1):
            return U*(1-eps) + 0.5*eps
        Ucand = nudge_interior(Ucand)

        #!Increase Exploration for the first 25 rounds
        if r <= 25:
            Ucand = (torch.rand_like(Ucand) if torch.rand(()) < 0.2 else Ucand)

        plist, F_np = evaluate_batch_batched(
            space, builder, solver, Ucand, eval_vec,
            batch_size=Ucand.shape[0], verbose=verbose
        )
        archive.add_batch(plist, F_np)

        U_buf.append(Ucand)
        F_buf.append(torch.tensor(F_np, device=space.lows.device, dtype=torch.float64))

        good_total += int(((np.abs(F_np) <= tol_feas).all(axis=1)).sum())
        
        if log:
            f_mins = np.min(F_np, axis=0)
            f_mins_str = ", ".join(f"{v:.6g}" for v in f_mins)
            print(f"[TPE {r:02d}/{tpe_rounds}] batch best fs = {f_mins_str} | "
                  f"feasible-so-far ≈ {good_total} | Pareto size = {archive.size_front()}")

    return {
        "archive": archive,
        "U_all": torch.vstack(U_buf),
        "F_all": torch.vstack(F_buf),
    }
    
    
##### Main Function ##### Main Function ##### Main Function ##### Main Function ##### Main Function ##### Main Function ##### Main Function ##### Main Function ##### Main Function ##### Main Function #####


def main():
    space = ParamSpace([
    #Spatial Source
	ParamDef(name="A_SR",       low=0.5e23,     high=3e23,          scale="linear"), # DONE
	ParamDef(name="Rc_SR",      low=0.90,       high=1.10,          scale="linear"), # DONE
	ParamDef(name="o_SR",       low=0.05,       high=0.20,          scale="linear"), # DONE
 
	#Temporal Source
	ParamDef(name="A_ST",       low=0,          high=0.25,          scale="linear"), # DONE
	ParamDef(name="f_ST",       low=0.1,        high=10,          scale="log10"), # DONE
	ParamDef(name="p_ST",       low=0,          high=2*math.pi,          scale="linear"), # DONE
 
	#Diffusion Profile
	ParamDef(name="A_D",        low=1,          high=10,          scale="log10"), # DONE
	ParamDef(name="AL_D",       low=0.20,       high=1.00,          scale="linear"), # DONE
	ParamDef(name="A0_D",       low=0.01,       high=0.20,          scale="linear"), # DONE
	ParamDef(name="Rc_D",       low=0.90,       high=1.10,          scale="linear"), # DONE
	ParamDef(name="RwL_D",      low=0.05,       high=0.200,          scale="linear"), # DONE
	ParamDef(name="RwR_D",      low=0.05,       high=0.200,          scale="linear"), # DONE
 
 	#Convection Profile
	ParamDef(name="A_V",        low=0.5,        high=7.5,          scale="log10"), # DONE #? Log?
	ParamDef(name="A1_V",       low=0.10,       high=0.45,          scale="linear"), # DONE
	ParamDef(name="A2_V",       low=0.25,       high=3.00,          scale="linear"), # DONE
	ParamDef(name="R1_V",       low=0.35,       high=0.60,          scale="linear"), # DONE
	ParamDef(name="R2_V",       low=0.65,       high=0.85,          scale="linear"), # DONE
	ParamDef(name="R3_V",       low=0.95,       high=1.15,          scale="linear"), # DONE
	ParamDef(name="Flip_V",     low=-1.0,       high=1.0,          scale="linear"), # DONE
	ParamDef(name="Bounce_V",   low=-1.0,       high=1.0,          scale="linear"), # DONE
 
	#Initial Density Profile
	ParamDef(name="A_N0",       low=5.5e19,     high=9e19,          scale="linear"), # DONE
	ParamDef(name="Xs_N0",      low=0.90,       high=1.10,          scale="linear"), # DONE
	ParamDef(name="H_N0",       low=0.006,      high=0.018,          scale="linear"), # DONE
	ParamDef(name="a_N0",       low=0.006,      high=0.016,          scale="linear"), # DONE
	ParamDef(name="B_N0",       low=0.85e20,    high=1.20e20,          scale="linear"), # DONE
 
 	#Edge Boundary Condition
	ParamDef(name="A_mag",      low=1e18,       high=1e22,          scale="log10"), # DONE
    ParamDef(name="A_sign",     low=-1.0,       high=1.0,          scale="linear"), # DONE
    ])

    #Define Scene Builder
    builder = SceneBuilder(
        rho=torch.linspace(1e-3, 1.2, steps=151, device=device, dtype=torch.float32), #601
        time=torch.linspace(1e-3, 0.065317, steps=161, device=device, dtype=torch.float32), #641
    )

    #Define Solver
    solver  = SolverV2_opt()

    #Define Evaluation Vector Func
    Maximum_Density = 5e20
    Maximum_Edge_Density = 1e20
    eval_vec = make_eval_vector(N_cap_all=Maximum_Density, N_cap_edge=Maximum_Edge_Density, N_min_core=2e20, std_max=5e-3)

    #!Visualize Profiles / Parameter Space
    t0 = time.perf_counter()
    n = 2048
    resultA = run_phase_A_mv_tpe(
        space=space,
        builder=builder,
        solver=solver,
        eval_vec=eval_vec,   # or your custom multi-metric function
        warmup_batches=50,
        warmup_q=2*8_192, #128
        tpe_rounds=1000, #12
        tpe_draws=4*n, #2048
        tpe_top_q=n, #128
        verbose=False,
        log=True,
        method='KDE',
        gmm_components=50,
    )
    duration = time.perf_counter() - t0

    archive = resultA["archive"]
    pareto_X, pareto_F = archive.front()
    print(f"\nPhase A completed in {duration:.1f} seconds.")
    print("Final Pareto size:", len(pareto_X))
    
    F_results = np.vstack(archive.F_all) if len(archive.F_all) else np.empty((0, archive.k))
    X_results = [dict(x) for x in archive.X_phys_all]  # list of dicts, same as before

    X_raw = np.array([list(X_results_i.values()) for X_results_i in X_results])
    F_raw = np.array(F_results)
    Good_X_raw = np.array([list(par_X_i.values()) for par_X_i in pareto_X])
    Good_F_raw = pareto_F.copy()
    
    Dict_ResultA = {
    'F_results': F_raw,
    'X_results': X_raw,
    'Good_X_raw': Good_X_raw,
	'Good_F_raw': Good_F_raw,
    'warmup_batches': 50,
    'warmup_q': 2*8192,
	'tpe_rounds': 1000,
	'tpe_top_q': 2048,
	'method': 'KDE',
    }
    WritePickle(Dict_ResultA, 'Opt_Results3.pickle', safe=False)
    
    print("Results saved to Opt_Results3.pickle")
    
    return True