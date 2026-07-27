#!/usr/bin/env python3
"""
@author: akilinc

fem_pixel.py  –  Pure-Python FEM replacing Abaqus (Case 5 / Case 6)
====================================================================
Element  : CPS4, plane stress, 2x2 Gauss
Material : von Mises + Ludwik hardening  sy(ep) = sy0 + K*ep^n
BCs      : All 4 edges prescribed from DIC data
Solver   : Incremental loading, Newton-Raphson with CONSISTENT tangent

Outputs (GP-averaged to element level, same grid as input maps):
  U    (nx_nodes, ny_nodes, 2)
  S    (nx_elems, ny_elems, 3)  [s11, s22, s12]
  E    (nx_elems, ny_elems, 3)  [e11, e22, g12]
  PE   (nx_elems, ny_elems, 3)
  PEEQ (nx_elems, ny_elems)
  RF   (nx_nodes, ny_nodes, 2)

Usage:
    from fem_pixel import run_fem
    result = run_fem(disp_x, disp_y, yield_map, K_map, n_exp=0.245,
                     x_size=0.1, y_size=0.1, element_size=0.001,
                     scale_factor=1.84, E_mod=205000., nu=0.3,
                     N_inc=20, verbose=True)
"""

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve, splu
import time
import functools
print = functools.partial(print, flush=True)   # live progress under runners

# Optional fast multithreaded direct solver (MKL Pardiso).
#   pip install pypardiso
try:
    import pypardiso
    def _solve(A, b):
        return pypardiso.spsolve(A.tocsr(), b)
    _SOLVER_NAME = "pypardiso (MKL, multithreaded)"
except Exception:
    def _solve(A, b):
        return spsolve(A.tocsr(), b)
    _SOLVER_NAME = "scipy SuperLU (single-threaded; 'pip install pypardiso' for a large speed-up)"

# ── Gauss quadrature 2x2 ────────────────────────────────────────────────────
_G = 1.0/np.sqrt(3.0)
GP_XI  = np.array([[-_G,-_G],[_G,-_G],[_G,_G],[-_G,_G]])
GP_W   = np.ones(4)
N_GP   = 4

# von Mises plane-stress matrix:  svm^2 = s^T M s
_M = np.array([[1.,-0.5,0.],[-0.5,1.,0.],[0.,0.,3.]])

def _vm(s):
    return np.sqrt(np.maximum(s[...,0]**2-s[...,0]*s[...,1]+s[...,1]**2+3*s[...,2]**2, 0.))

# ── Shape function derivatives ───────────────────────────────────────────────
def _dN(xi,eta):
    return 0.25*np.array([[-(1-eta),(1-eta),(1+eta),-(1+eta)],
                           [-(1-xi),-(1+xi),(1+xi), (1-xi)]])

# ── B matrix ────────────────────────────────────────────────────────────────
def _B_detJ(coords, xi, eta):
    dNn = _dN(xi,eta); J = dNn@coords
    dJ  = J[0,0]*J[1,1]-J[0,1]*J[1,0]
    Ji  = np.array([[J[1,1],-J[0,1]],[-J[1,0],J[0,0]]])/dJ
    dNx = Ji@dNn
    B   = np.zeros((3,8))
    for k in range(4):
        B[0,2*k]=dNx[0,k]; B[1,2*k+1]=dNx[1,k]
        B[2,2*k]=dNx[1,k]; B[2,2*k+1]=dNx[0,k]
    return B, dJ

def _Cps(E,nu):
    f=E/(1-nu*nu)
    return f*np.array([[1,nu,0],[nu,1,0],[0,0,(1-nu)/2.]])

# ── Hardening laws ───────────────────────────────────────────────────────────
# sy(ep) = sy0 + K*hf(ep),  H(ep) = K*hfp(ep)
# 'ludwik'  : hf(ep)=ep^n (analytic, keeps hardening forever)
# 'tabular' : piecewise-linear interpolation of ep^n at the same 50 knots that
#             Case5/generate_mesh write into the .inp (E_p=linspace(0,0.2,50)),
#             perfectly plastic beyond the last knot — exactly what Abaqus solves.
def make_hardening(n_exp, mode='ludwik', ep_max=0.2, n_pts=50):
    if mode == 'ludwik':
        def hf(ep):  return np.where(ep>0, np.maximum(ep,0.)**n_exp, 0.)
        def hfp(ep): return n_exp*np.where(ep>1e-15, ep**(n_exp-1.), 0.)
        return hf, hfp
    elif mode == 'tabular':
        ep_k = np.linspace(0., ep_max, n_pts)
        f_k  = np.where(ep_k>0, ep_k**n_exp, 0.)
        sl   = np.diff(f_k)/np.diff(ep_k)
        def hf(ep):
            return np.interp(np.clip(ep,0.,ep_k[-1]), ep_k, f_k)
        def hfp(ep):
            idx = np.clip(np.searchsorted(ep_k, ep, side='right')-1, 0, len(sl)-1)
            return np.where(ep < ep_k[-1], sl[idx], 0.)
        return hf, hfp
    raise ValueError(f"unknown hardening mode '{mode}'")


# ── Return mapping ───────────────────────────────────────────────────────────
# A = I + r*CM  is block-diagonal:  cm13=cm23=0  → analytical solve
def _rm(s_tr, ep0, sy0, K, hf, cm11, cm12, cm33, max_it=50, tol=1e-10):
    """
    Vectorised return mapping.
    Returns: sigma (N,3), deps_p (N,3), dg (N,) equiv-plastic increment
    """
    def sy(ep): return sy0 + K*hf(ep)

    def solve_s(dg):
        sk = sy(ep0+dg)
        r  = dg/np.where(sk>1e-30,sk,1e-30)
        a=1+r*cm11; b=r*cm12; c=1+r*cm33; d=a*a-b*b
        return np.stack([(a*s_tr[:,0]-b*s_tr[:,1])/d,
                          (a*s_tr[:,1]-b*s_tr[:,0])/d,
                          s_tr[:,2]/c], axis=1)

    phi0 = _vm(s_tr)-sy(ep0)
    pl   = phi0>0
    sig  = s_tr.copy(); dp=np.zeros_like(s_tr); dg=np.zeros(len(ep0))
    if not pl.any(): return sig, dp, dg

    idx=np.where(pl)[0]
    s0=s_tr[idx]; e0=ep0[idx]; sy0p=sy0[idx]; Kp=K[idx]
    def syp(g): return sy0p+Kp*hf(e0+g)
    def solp(g):
        sk=syp(g); r=g/np.where(sk>1e-30,sk,1e-30)
        a=1+r*cm11; b=r*cm12; c=1+r*cm33; d=a*a-b*b
        return np.stack([(a*s0[:,0]-b*s0[:,1])/d,
                          (a*s0[:,1]-b*s0[:,0])/d,
                          s0[:,2]/c], axis=1)
    def res(g): return _vm(solp(g))-syp(g)

    # Guarded Newton: res(g) is monotone decreasing, res(0)>0.
    # Bracket [g_lo,g_hi] first, then Newton with bisection fallback —
    # unconditionally stable even in the perfectly-plastic (H=0) clamp
    # of tabular hardening.
    n_p=len(idx)
    g_lo=np.zeros(n_p); g_hi=np.full(n_p,1e-4)
    for _ in range(80):
        open_=res(g_hi)>0
        if not open_.any(): break
        g_hi=np.where(open_,g_hi*2.,g_hi)
    g=0.5*(g_lo+g_hi)
    for _ in range(max(max_it,100)):
        r=res(g); conv=(np.abs(r)/(syp(g)+1e-30)).max()
        if conv<tol: break
        g_lo=np.where(r>0,g,g_lo); g_hi=np.where(r<=0,g,g_hi)
        h=np.maximum(np.abs(g),1e-8)*1e-7
        dr=(res(g+h)-r)/h; dr=np.where(np.abs(dr)<1e-30,-1.,dr)
        g_new=g-r/dr
        bad=~np.isfinite(g_new)|(g_new<=g_lo)|(g_new>=g_hi)
        g=np.where(bad,0.5*(g_lo+g_hi),g_new)

    sf=solp(g); sv=_vm(sf)
    nd=(sf@_M.T)/np.maximum(sv,1e-30)[:,None]
    sig[idx]=sf; dp[idx]=g[:,None]*nd; dg[idx]=g
    return sig, dp, dg


# ── Consistent tangent ───────────────────────────────────────────────────────
def _cep(sigma, dg, ep0, sy0p, Kp, hf, hfp, C_ps, cm11, cm12, cm33):
    """
    Analytical consistent tangent for plastic GPs.
    Returns C_ep (N_p, 3, 3)  where C_ep = T @ C_ps
    T = A^-1 - outer(w,p)/(1+beta)
    with w = A^-1 * alpha*CM*s,  p = A^-1 * n,
         alpha = (1-r*H)/(sy*H)
    """
    sk = sy0p+Kp*hf(ep0+dg)
    Hk = Kp*hfp(ep0+dg)
    r  = dg/np.where(sk>1e-30,sk,1e-30)
    a=1+r*cm11; b=r*cm12; c=1+r*cm33; d=a*a-b*b
    Hs = np.where(np.abs(Hk)>1e-30,Hk,1e-30)
    alpha = (1.-r*Hk)/(sk*Hs)

    s0,s1,s2 = sigma[:,0],sigma[:,1],sigma[:,2]
    # CM*s
    q0=cm11*s0+cm12*s1; q1=cm12*s0+cm11*s1; q2=cm33*s2
    # w = A^-1 * alpha*q
    aq0=alpha*q0; aq1=alpha*q1; aq2=alpha*q2
    w0=(a*aq0-b*aq1)/d; w1=(-b*aq0+a*aq1)/d; w2=aq2/c
    # n, p = A^-1*n
    sv=_vm(sigma); sv_=np.where(sv>1e-30,sv,1e-30)
    n0=(s0-.5*s1)/sv_; n1=(s1-.5*s0)/sv_; n2=3*s2/sv_
    p0=(a*n0-b*n1)/d; p1=(-b*n0+a*n1)/d; p2=n2/c
    beta=n0*w0+n1*w1+n2*w2
    denom=1.+beta

    # Build A^-1 (N_p,3,3)
    z=np.zeros_like(a)
    Ai=np.stack([np.stack([a/d,-b/d,z],1),
                 np.stack([-b/d,a/d,z],1),
                 np.stack([z,z,1/c],1)],axis=1)     # (N,3,3)
    wv=np.stack([w0,w1,w2],1); pv=np.stack([p0,p1,p2],1)
    corr=np.einsum('ni,nj->nij',wv,pv)/denom[:,None,None]
    T=Ai-corr
    return np.einsum('nij,jk->nik',T,C_ps)          # (N,3,3)


# ── Mesh ─────────────────────────────────────────────────────────────────────
class _Mesh:
    def __init__(self,xs,ys,el,sf):
        nx=int(round(xs/el)); ny=int(round(ys/el))
        self.nx,self.ny=nx,ny; nxn,nyn=nx+1,ny+1
        self.n_nodes=nxn*nyn; self.n_elems=nx*ny; self.n_dof=2*nxn*nyn
        self.node_ids=np.arange(nxn*nyn).reshape((nxn,nyn),order='F')
        xp=np.linspace(0,nx*el,nxn)*sf; yp=np.linspace(0,ny*el,nyn)*sf
        self.coords=np.zeros((nxn*nyn,2))
        ii,jj=np.meshgrid(np.arange(nxn),np.arange(nyn),indexing='ij')
        nd=self.node_ids[ii,jj]
        self.coords[nd.ravel(),0]=np.repeat(xp,nyn)
        self.coords[nd.ravel(),1]=np.tile(yp,nxn)
        self.elem_ids=np.arange(nx*ny).reshape((nx,ny),order='F')
        ie,je=np.meshgrid(np.arange(nx),np.arange(ny),indexing='ij')
        ef=self.elem_ids[ie,je].ravel()
        conn=np.zeros((nx*ny,4),dtype=int)
        conn[ef,0]=self.node_ids[ie.ravel(),  je.ravel()  ]
        conn[ef,1]=self.node_ids[ie.ravel()+1,je.ravel()  ]
        conn[ef,2]=self.node_ids[ie.ravel()+1,je.ravel()+1]
        conn[ef,3]=self.node_ids[ie.ravel(),  je.ravel()+1]
        self.conn=conn
        bc=np.concatenate([self.node_ids[:,0],self.node_ids[:,-1],
                           self.node_ids[0,1:-1],self.node_ids[-1,1:-1]])
        bcd=np.union1d(2*bc,2*bc+1)
        self.dofs_bc=bcd.astype(int)
        self.dofs_free=np.setdiff1d(np.arange(self.n_dof),bcd).astype(int)


# ── Element precompute ───────────────────────────────────────────────────────
def _precomp(mesh,C_ps):
    c=mesh.coords[mesh.conn[0]]
    Ke=np.zeros((8,8)); Bs=np.zeros((N_GP,3,8)); dJs=np.zeros(N_GP)
    for g,(xi_eta,w) in enumerate(zip(GP_XI,GP_W)):
        B,dJ=_B_detJ(c,xi_eta[0],xi_eta[1])
        Ke+=w*(B.T@C_ps@B)*dJ; Bs[g]=B; dJs[g]=dJ
    return Ke,Bs,dJs


# ── Assembly (COO→CSR) ───────────────────────────────────────────────────────
def _assemble(mesh, Ke_arr, ld, rc=None):
    """Ke_arr : (n_e,8,8)  or scalar broadcast (8,8).
    rc : optional precomputed (rows, cols) raveled index arrays (constant per
    mesh — pass them to avoid rebuilding every Newton iteration)."""
    n_e=mesh.n_elems; n_dof=mesh.n_dof
    if rc is None:
        rows=ld[:,:,None].repeat(8,axis=2).ravel()
        cols=ld[:,None,:].repeat(8,axis=1).ravel()
    else:
        rows,cols=rc
    if Ke_arr.ndim==2:
        data=np.broadcast_to(Ke_arr,(n_e,8,8))
    else:
        data=Ke_arr
    return coo_matrix((data.ravel(),(rows,cols)),
                      shape=(n_dof,n_dof)).tocsr()


# ── Internal forces ──────────────────────────────────────────────────────────
def _Fint(mesh,sig,Bs,dJs,ld):
    fe=np.einsum('g,g,gak,ega->ek',GP_W,dJs,Bs,sig)
    F=np.zeros(mesh.n_dof); np.add.at(F,ld,fe); return F


# ── Main solver ──────────────────────────────────────────────────────────────
def run_fem(disp_x,disp_y,yield_map,K_map,n_exp,
            x_size,y_size,element_size,scale_factor,
            E_mod=205000.,nu=0.3,N_inc=20,max_nr=15,nr_tol=1e-6,
            hardening='ludwik',ep_table_max=0.2,n_table=50,
            snapshot_fractions=None,
            verbose=True):
    """
    hardening : 'ludwik'  – analytic sy = sy0 + K*ep^n
                'tabular' – piecewise-linear 50-pt table clamped at ep_table_max
                            (matches the *Plastic table Abaqus interpolates)
    snapshot_fractions : optional list of load fractions in (0,1]; the S/E/PEEQ
                fields at those pseudo-times are recorded during ONE incremental
                solve and returned in result['frames'] = {fraction: {...}}.
                (Replaces re-solving the whole problem per load level.)
    """
    t0=time.time()
    hf,hfp=make_hardening(n_exp,hardening,ep_table_max,n_table)
    mesh=_Mesh(x_size,y_size,element_size,scale_factor)
    nx,ny=mesh.nx,mesh.ny; nxn,nyn=nx+1,ny+1; n_e=mesh.n_elems

    if verbose:
        print(f"[FEM] {nx}x{ny} elems | {len(mesh.dofs_free)} free DOFs | solver: {_SOLVER_NAME}")

    # element->grid output helpers (also used for snapshots)
    def gm(a): return a.mean(axis=1)
    def tg(a):
        if a.ndim==1: return a.reshape(nx,ny,order='F')
        return a.reshape(nx,ny,a.shape[1],order='F')

    # Material per GP
    sy0_gp=np.repeat(yield_map.ravel(order='F'),N_GP)
    K_gp  =np.repeat(K_map.ravel(order='F'),N_GP)

    C_ps=_Cps(E_mod,nu); CM=C_ps@_M
    cm11,cm12,cm33=CM[0,0],CM[0,1],CM[2,2]
    Ke,Bs,dJs=_precomp(mesh,C_ps)

    # DOF maps
    nd_all=mesh.node_ids
    nodes=mesh.conn
    ld=np.zeros((n_e,8),dtype=int)
    ld[:,0::2]=2*nodes; ld[:,1::2]=2*nodes+1
    dof_I=mesh.dofs_free; dof_B=mesh.dofs_bc

    # constant assembly index arrays (reused for every tangent assembly)
    _rc=(ld[:,:,None].repeat(8,axis=2).ravel(),
         ld[:,None,:].repeat(8,axis=1).ravel())

    # Elastic K – factorized ONCE and reused for every increment's predictor
    K_el=_assemble(mesh,Ke,ld,_rc)
    KII_el=K_el[dof_I][:,dof_I].tocsr()
    KIB_el=K_el[dof_I][:,dof_B].tocsr()
    try:
        _lu_el=splu(KII_el.tocsc())
        solve_el=_lu_el.solve
    except Exception:
        solve_el=lambda b: _solve(KII_el,b)

    # Prescribed displacements
    u_bc=np.zeros(mesh.n_dof)
    u_bc[2*nd_all.ravel(order='F')]  =disp_x.ravel(order='F')
    u_bc[2*nd_all.ravel(order='F')+1]=disp_y.ravel(order='F')

    # State
    u=np.zeros(mesh.n_dof)
    eps_p =np.zeros((n_e,N_GP,3)); ep_bar=np.zeros((n_e,N_GP))
    sig=np.zeros((n_e,N_GP,3)); eps_tot=np.zeros((n_e,N_GP,3))

    # Incremental loading with automatic cutback (Abaqus-style):
    # pseudo-time t: 0 -> 1, initial/maximum step 1/N_inc, halved on failure.
    t=0.; dt=1./N_inc; dt_max=1./N_inc; dt_min=dt_max/1024.
    inc=0
    snaps={}
    pending=sorted(snapshot_fractions) if snapshot_fractions else []
    while t<1.-1e-12:
        dt=min(dt,1.-t)
        if pending:                      # land exactly on snapshot fractions
            dt=min(dt,max(pending[0]-t,1e-12))
        inc+=1
        du_B=u_bc[dof_B]*dt
        u_save=u.copy()

        u[dof_B]+=du_B
        # Elastic predictor (reused factorization)
        u[dof_I]+=solve_el(-KIB_el@du_B)

        KII=KII_el   # start with elastic tangent, replaced after 1st iter

        # Saved converged state (updated each NR iter before possible break)
        sf_acc=sig.copy(); dp_acc=np.zeros_like(eps_p)
        ep_new=ep_bar.copy()
        converged=False

        for nrit in range(max_nr):
            u_e   =u[ld]
            eps_tot=np.einsum('gak,ek->ega',Bs,u_e)
            sig_tr =np.einsum('ij,egj->egi',C_ps,eps_tot-eps_p)

            # Return mapping
            sf,dp,dg=_rm(sig_tr.reshape(-1,3),ep_bar.ravel(),
                         sy0_gp,K_gp,hf,cm11,cm12,cm33)
            sf =sf.reshape(n_e,N_GP,3)
            dp =dp.reshape(n_e,N_GP,3)
            dg_=dg.reshape(n_e,N_GP)
            ep_new=ep_bar+dg_

            # Save state from this iteration (used if we break here)
            sf_acc=sf; dp_acc=dp

            # Internal forces and residual
            R=_Fint(mesh,sf,Bs,dJs,ld); R_I=R[dof_I]
            res=np.linalg.norm(R_I)
            if not np.isfinite(res): break          # diverged -> cutback
            if nrit==0: res0=max(res,1e-30)
            rel=res/res0
            if verbose:
                print(f"  inc {inc:3d} t={t+dt:.4f} NR {nrit+1:2d} |R|={res:.2e} rel={rel:.2e}")

            # Converge on absolute OR relative residual
            if res<1e-10 or rel<nr_tol:
                converged=True; break

            # Build consistent tangent stiffness
            N_flat=n_e*N_GP
            C_ep_flat=np.broadcast_to(C_ps,(N_flat,3,3)).copy()
            pl_idx=np.where(dg>0)[0]
            if len(pl_idx):
                C_ep_flat[pl_idx]=_cep(
                    sf.reshape(-1,3)[pl_idx],
                    dg[pl_idx],
                    ep_bar.ravel()[pl_idx],
                    sy0_gp[pl_idx],K_gp[pl_idx],hf,hfp,
                    C_ps,cm11,cm12,cm33)
            C_ep_gp=C_ep_flat.reshape(n_e,N_GP,3,3)

            # Vectorised element tangent stiffness
            CB=np.einsum('egij,gjk->egik',C_ep_gp,Bs)
            Ke_ep=np.einsum('g,g,gik,egil->ekl',GP_W,dJs,Bs,CB)
            K_tang=_assemble(mesh,Ke_ep,ld,_rc)
            KII=K_tang[dof_I][:,dof_I].tocsr()

            du=_solve(KII,-R_I)
            if not np.isfinite(du).all(): break     # singular tangent -> cutback
            u[dof_I]+=du

        if not converged:
            u=u_save; dt*=0.5
            if verbose:
                print(f"  inc {inc:3d}: no convergence, cutback dt -> {dt:.2e}")
            if dt<dt_min:
                raise RuntimeError("run_fem: increment cutback below minimum "
                                   f"({dt:.2e}) - solution not converging")
            continue

        eps_p+=dp_acc; ep_bar=ep_new; sig=sf_acc
        t+=dt
        dt=min(dt*1.5,dt_max)   # grow back after success

        # record snapshot fields at requested load fractions (element fields
        # + nodal U, so a stress/strain/u2 - "time" curve can be built later
        # without re-solving; same nodal-reshape as the final-state U below)
        while pending and t>=pending[0]-1e-9:
            fsnap=pending.pop(0)
            Usnap=np.zeros((nxn,nyn,2))
            Usnap[...,0]=u[2*nd_all].reshape(nxn,nyn)
            Usnap[...,1]=u[2*nd_all+1].reshape(nxn,nyn)
            snaps[fsnap]=dict(S=tg(gm(sig)), E=tg(gm(eps_tot)), PEEQ=tg(gm(ep_bar)),
                              U=Usnap)
            if verbose:
                print(f"[FEM] snapshot recorded at load fraction {fsnap:.3f}")

    # Output
    F_all=_Fint(mesh,sig,Bs,dJs,ld)
    bc_m=np.zeros(mesh.n_dof,dtype=bool); bc_m[dof_B]=True

    U=np.zeros((nxn,nyn,2))
    U[...,0]=u[2*nd_all].reshape(nxn,nyn)
    U[...,1]=u[2*nd_all+1].reshape(nxn,nyn)
    RF=np.zeros((nxn,nyn,2))
    RF[...,0]=np.where(bc_m[2*nd_all],F_all[2*nd_all],0.).reshape(nxn,nyn)
    RF[...,1]=np.where(bc_m[2*nd_all+1],F_all[2*nd_all+1],0.).reshape(nxn,nyn)

    if verbose:
        print(f"\n[FEM] done {time.time()-t0:.1f}s  "
              f"PEEQ_max={ep_bar.max():.4f}  "
              f"svm_max={_vm(sig.reshape(-1,3)).max():.1f}MPa")
    return dict(U=U, S=tg(gm(sig)), E=tg(gm(eps_tot)),
                PE=tg(gm(eps_p)), PEEQ=tg(gm(ep_bar)), RF=RF, mesh=mesh,
                frames=snaps)


# ── Verification: equal biaxial tension (self-consistent BCs) ───────────────
def _verify():
    """
    Equal biaxial tension on 20x20 homogeneous mesh.
    BCs: u_x = eps*x, u_y = eps*y  where eps is computed analytically
    for a target sigma_vm = 400 MPa (above yield=250).
    Expected PEEQ = 0.00737, sigma_vm = 400 MPa.
    """
    print("="*55)
    print("VERIFICATION: equal biaxial tension, 20x20 mesh")
    print("="*55)
    nx,ny=20,20; el=0.001; sf=1.; E,nu=205000.,0.3
    sy0,K,nexp=250.,500.,0.245

    # Analytical solution for equal biaxial sigma_vm = sig_t
    sig_t=400.
    # ep from Ludwik: sig_t = sy0 + K*ep^n
    ep_ref=((sig_t-sy0)/K)**(1./nexp)
    # For equal biaxial: ep_11=ep_22=ep/2, eeq=ep
    eps_e=sig_t*(1.-nu)/E
    eps_tot_val=eps_e+ep_ref/2.

    nxn,nyn=nx+1,ny+1
    xs=np.linspace(0,nx*el*sf,nxn); ys=np.linspace(0,ny*el*sf,nyn)
    xx,yy=np.meshgrid(xs,ys,indexing='ij')
    disp_x=eps_tot_val*xx; disp_y=eps_tot_val*yy

    result=run_fem(disp_x,disp_y,
                   np.full((nx,ny),sy0),np.full((nx,ny),K),nexp,
                   nx*el,ny*el,el,sf,E_mod=E,nu=nu,
                   N_inc=30,max_nr=10,nr_tol=1e-8,verbose=True)

    sv=_vm(result['S'].reshape(-1,3)).mean()
    ep=result['PEEQ'].mean()
    print(f"\n  sigma_vm  = {sv:.3f} MPa  (expected {sig_t:.1f})")
    print(f"  PEEQ      = {ep:.6f}     (expected {ep_ref:.6f})")
    err=abs(sv-sig_t)/sig_t*100
    print(f"  Error     = {err:.3f} %")
    assert err<0.5, f"FAILED: {err:.2f}%"
    print("  PASSED")

if __name__=="__main__":
    _verify()
