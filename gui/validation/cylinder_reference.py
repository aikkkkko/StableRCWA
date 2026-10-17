"""Independent cylindrical-wave (2D Mie) reference, no RCWA/PML code imports.

For the scalar field Ey (TE) or Hy (TM), continuity of u and
(1/chi) du/dr gives each outgoing Hankel coefficient. chi=mu for TE,
chi=eps for TM. The cylinder and exterior are nonmagnetic here.
"""
import numpy as np
from scipy.special import jv,jvp,hankel1,h1vp


def cylinder_field(x,z,*,radius=.25,n=1.5,lam0=1.,theta_deg=30.,
                   center=(0.,0.),pol="TE",terms=24):
    x,z=np.broadcast_arrays(np.asarray(x,float),np.asarray(z,float))
    shape=x.shape;dx=(x-center[0]).ravel();dz=(z-center[1]).ravel()
    r=np.hypot(dx,dz);phi=np.arctan2(dz,dx)
    theta=np.radians(theta_deg);k=2*np.pi/lam0;kc=k*n
    phi0=np.pi/2+theta
    m=np.arange(-terms,terms+1)
    chi=1. if pol=="TE" else n*n
    J=jv(m,k*radius);Jp=jvp(m,k*radius)
    H=hankel1(m,k*radius);Hp=h1vp(m,k*radius)
    Jc=jv(m,kc*radius);Jcp=jvp(m,kc*radius)
    a=(kc/chi*Jcp*J-k*Jp*Jc)/(k*Hp*Jc-kc/chi*Jcp*H)
    b=(J+a*H)/Jc
    # The incident amplitude is 1 at the global coordinate origin.
    phase=np.exp(1j*k*(-np.sin(theta)*center[0]+np.cos(theta)*center[1]))
    weights=phase*(1j**m)*np.exp(-1j*m*phi0)
    u=np.empty(len(r),complex);ur=np.empty_like(u);up=np.empty_like(u)
    outside=r>=radius
    for sel,inside in ((outside,False),(~outside,True)):
        rr=r[sel];pp=phi[sel]
        if not len(rr):continue
        angle=np.exp(1j*np.outer(m,pp))*weights[:,None]
        if inside:
            radial=b[:,None]*jv(m[:,None],kc*rr)
            dr=b[:,None]*kc*jvp(m[:,None],kc*rr)
        else:
            radial=jv(m[:,None],k*rr)+a[:,None]*hankel1(m[:,None],k*rr)
            dr=k*(jvp(m[:,None],k*rr)+a[:,None]*h1vp(m[:,None],k*rr))
        u[sel]=np.sum(angle*radial,axis=0)
        ur[sel]=np.sum(angle*dr,axis=0)
        up[sel]=np.sum(angle*(1j*m[:,None])*radial,axis=0)
    # Derivatives at r=0 are not needed in this benchmark. Main u remains exact.
    angular=np.divide(up,r,out=np.zeros_like(up),where=r>1e-12)
    ux=np.cos(phi)*ur-np.sin(phi)*angular
    uz=np.sin(phi)*ur+np.cos(phi)*angular
    chi_point=np.where(outside,1.,chi)
    inc=np.exp(1j*k*(-np.sin(theta)*x+np.cos(theta)*z))
    if pol=="TE":
        result=dict(Ey=u.reshape(shape),Hx=(1j*uz/k).reshape(shape),Hz=(-1j*ux/k).reshape(shape))
    else:
        result=dict(Hy=u.reshape(shape),Ex=(-1j*uz/(k*chi_point)).reshape(shape),Ez=(1j*ux/(k*chi_point)).reshape(shape))
    result["main"]=u.reshape(shape);result["incident"]=inc
    result["scattered"]=u.reshape(shape)-inc
    result["scattering_width"]=float(4/k*np.sum(abs(a)**2))
    return result
