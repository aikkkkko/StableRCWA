"""Two-dimensional worker, slice coordinates, power channels and saved views."""
import json,os,tempfile,unittest
from pathlib import Path
import numpy as np
from gui_models import build_stack,from_stack,preset
from gui_models_2d import load_rcwa2d
from gui_worker import compute
from gui_fields import view_data,COMPONENTS,render_view

class TestGUI2D(unittest.TestCase):
    # These tests check the Python 2-D path against rcwa2d directly (atol 3e-12), so they pin the worker to it. Periodic 2-D
    # models now go to C++ when a program is available; that route is tested in test_cpp_2d_backend.py (same checks plus fallback).
    @classmethod
    def setUpClass(cls):
        cls._backend=os.environ.get("RCWA_BACKEND");os.environ["RCWA_BACKEND"]="python"

    @classmethod
    def tearDownClass(cls):
        if cls._backend is None:os.environ.pop("RCWA_BACKEND",None)
        else:os.environ["RCWA_BACKEND"]=cls._backend

    def test_absorbing_uniform_layer_matches_fresnel_airy(self):
        rc=load_rcwa2d();n=1.65+.12j;d=.27
        for theta in (0.,27.):
            q0=np.cos(np.radians(theta));q1=np.sqrt(n*n-np.sin(np.radians(theta))**2)
            phase=np.exp(2j*np.pi*q1*d)
            st=rc.Stack2D(1.,[rc.Layer2D(d,n=n)],1.,rc.Lattice(1.1,.8))
            for psi in (0.,90.):
                adm=q1 if psi==90 else q1/(n*n)
                r01=(q0-adm)/(q0+adm)
                r=r01*(1-phase**2)/(1-r01*r01*phase**2)
                t=(1-r01*r01)*phase/(1-r01*r01*phase**2)
                out=rc.solve2d(st,1.,theta,33.,0,0,psi_deg=psi)
                self.assertAlmostEqual(float(out.DEr.sum()),float(abs(r)**2),places=12)
                self.assertAlmostEqual(float(out.DEt.sum()),float(abs(t)**2),places=12)
                self.assertLess(float(out.DEr.sum()+out.DEt.sum()),1.)

    def test_real_two_dimensional_fields_on_both_slice_directions(self):
        rc=load_rcwa2d()
        for plane,at in [('xz',.4),('yz',.5)]:
            cfg=preset('pillar_2d');cfg['sampling'].update(slice_plane=plane,slice_at=at,nx=45,nz=25)
            cfg,st,_,_=build_stack(cfg);i=cfg['incident'];sam=cfg['sampling']
            r=rc.solve2d(st,i['wavelength'],i['angle_deg'],i['phi_deg'],i['M'],i['My'],psi_deg=i['psi_deg'])
            with tempfile.TemporaryDirectory() as td:
                m=compute(cfg,td)
                with np.load(Path(td)/'fields.npz') as archive:f=dict(archive)
                ref=rc.field_slice(r,np.linspace(sam['z_min'],sam['z_max'],sam['nz']),axis=plane[0],at=at,npts=sam['nx'])
                for k in COMPONENTS:np.testing.assert_allclose(f[k],ref[k],atol=3e-12)
                self.assertAlmostEqual(m['R_total']+m['T_total'],1.,places=10)
                for o in m['orders']:
                    self.assertAlmostEqual(o['R'],o['R_s']+o['R_p'],places=10)
                    self.assertAlmostEqual(o['T'],o['T_s']+o['T_p'],places=10)
                self.assertEqual(m['field_plane'],plane)
                if plane=='yz':
                    np.testing.assert_allclose(f['y'],np.arange(45)*.8/45)
                    np.testing.assert_allclose(f['x'],at)
                im=render_view(td,'Hz','phase_relative')
                self.assertTrue((Path(td)/im['filename']).stat().st_size>1000)
                self.assertIn('固定'+('y' if plane=='xz' else 'x'),im['note'])

    def test_uniform_medium_has_correct_phase_and_incident_normalization(self):
        rc=load_rcwa2d();n=1.4;th=np.radians(20.);ph=np.radians(30.)
        st=rc.Stack2D(n,[rc.Layer2D(.3,n=n)],n,rc.Lattice(1.,.8))
        cfg=from_stack(st,lam0=1.,theta_deg=20,phi_deg=30,pol='jones',psi_deg=45,delta_deg=90,M=1,My=1)
        cfg['sampling'].update(nx=40,nz=20,z_min=-.1,z_max=.4,slice_plane='yz',slice_at=.37)
        with tempfile.TemporaryDirectory() as td:
            compute(cfg,td)
            with np.load(Path(td)/'fields.npz') as archive:f=dict(archive)
            for key in ['E2','H2']:
                v,_=view_data(f,cfg,key);np.testing.assert_allclose(v,1.,atol=4e-12)
            E=(1j*np.array([-np.sin(ph),np.cos(ph),0.])+np.array([np.cos(th)*np.cos(ph),np.cos(th)*np.sin(ph),np.sin(th)]))/np.sqrt(2)
            for k,e in zip(COMPONENTS[:3],E):
                v,_=view_data(f,cfg,k,'phase_relative')
                np.testing.assert_allclose(np.exp(1j*v),np.exp(1j*np.angle(e)),atol=4e-12)

    def test_two_d_my_zero_matches_conical_efficiencies(self):
        from conical.rcwa_conical import solve_conical
        import rcwa1d
        cfg=preset('pillar_2d');cfg['incident'].update(M=5,My=0,polarization='jones',psi_deg=33,delta_deg=50)
        cfg['structure']['layers']=[dict(name='extruded',kind='patterned',thickness=.3,edges=[0,.35,1.],n_values=[1.7,1.])]
        cfg,st,_,_=build_stack(cfg);i=cfg['incident']
        a=load_rcwa2d().solve2d(st,1.,20.,30.,5,0,psi_deg=33,delta_deg=50)
        one=rcwa1d.Stack(1.,[rcwa1d.Layer(.3,edges=[0,.35,1.],n_values=[1.7,1.])],1.45,1.)
        b=solve_conical(one,1.,20.,30.,M=5,psi_deg=33,delta_deg=50)
        np.testing.assert_allclose(a.DEr,b.DEr,atol=3e-12)
        np.testing.assert_allclose(a.DEt,b.DEt,atol=3e-12)

if __name__=='__main__':unittest.main(verbosity=2)
