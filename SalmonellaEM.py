from __future__ import division
from __future__ import print_function
from pylab import *
import pandas as pd
from numba import jit, jitclass
from numba import int64, int32, int16, float64, float32, int16
from EMestimation import *

Salmonella_dtype = [('dim', int64),
                    ('Nh', int64),
                    ('vmag', float64[:]),
                    ('v', float64[:, :])]
@jitclass(Salmonella_dtype)
class ThreeStateModel(EMestimation):
    def __init__(self, dim, Nphi):
        """Initialize using the number of velocity states."""
        self.dim = dim
        Ntheta = Nphi*4
        self.Nh = Ntheta*Nphi + 2
        Nrun = Ntheta*Nphi
        theta, phi = linspace(0, 2*pi, Ntheta+1)[:-1], linspace(0, 0.5*pi, Nphi+1)[:-1]
        # THETA, PHI = meshgrid(theta, phi)
        THETA = zeros((Nphi, Ntheta))
        PHI = zeros((Nphi, Ntheta))
        for i in arange(Nphi):
            for j in arange(Ntheta):
                THETA[i, j] = theta[j]
                PHI[i, j] = phi[i]
        self.vmag = zeros(self.Nh)
        if dim == 2:
            v = zeros((self.Nh, 2))
            v[2:, 0] = (cos(PHI)*cos(THETA)).flatten()
            v[2:, 1] = (cos(PHI)*sin(THETA)).flatten()
            self.vmag[2:] = (cos(PHI)**2).flatten()
        elif dim == 3:
            v = zeros((self.Nh, 3))
            v[2:, 0] = (cos(PHI)*cos(THETA)).flatten()
            v[2:, 1] = (cos(PHI)*sin(THETA)).flatten()
            v[2:, 2] = sin(PHI).flatten()
            self.vmag[2:] = ones_like(PHI).flatten()
        else:
            assert dim == 2 or dim == 3
        self.v = v
    def pss(self, pars):
        k1, k2, k3, k4 = pars[:4]
        O = k2*k4 + k1*k4 + k1*k3
        p1 = k2*k4/O
        p2 = k1*k4/O
        p3 = k1*k3/O
        return array((p1, p2, p3))
    def Pinit(self, Np, pars):
        """Initialize hidden state distribution for each path."""
        pinit = self.pss(pars)
        p0 = ones((Np, self.Nh))
        p0[:, 0] = pinit[0]
        p0[:, 1] = pinit[1]
        p0[:, 2:] = pinit[2]/(self.Nh - 2)
        return p0
    def getQ(self, pars):
        """Generate the propagator matrix for the hidden state Markov process."""
        k1, k2, k3, k4 = pars[:4]
        A = array(((-k1, k2, 0.), (k1, -k2-k3, k4), (0., k3, -k4)))
        Q = eye(3) + A
        Ap = dot(A, eye(3))
        for n in arange(2, 10):
            Ap = dot(Ap, A/n)
            Q += Ap
        return Q
    def UandT(self, dx, pars, Q):
        """Evaluate the observation and hidden state probability matrices for a given path."""
        k1, k2, k3, k4 = pars[:4]
        mV, Dtrap, Dtum, Dswim = pars[4:]
        Nt, _ = dx.shape
        U = zeros((Nt, self.Nh))
        Darr = array((Dtrap, Dtum, Dswim))
        for n in arange(self.Nh):
            r = (dx - mV*self.v[n])**2
            D = Darr[min(2, n)]
            factor = (4.*pi*D)**(self.dim/2.)
            U[:, n] = exp(-psum(r)/(4.*D))/factor
        T0 = zeros((self.Nh, self.Nh))
        T0[:2, :2] = Q[:2, :2]
        T0[1, 2:] = Q[1, 2]
        T0[2:, 1] = Q[2, 1]/(self.Nh - 2)
        T0[2:, 2:] = eye(self.Nh - 2)*Q[2, 2]
        # T = eye(self.Nh) + T0
        # Ap = dot(T0, eye(self.Nh))
        # for n in arange(2, 10):
        #     Ap = dot(Ap, T0/n)
        #     T += Ap
        return U, T0
    def Mstep(self, DXarray, pInds, S, S2):
        """Extract maximimum likelihood parameters given hidden state distribution for the E step."""
        Np = pInds.size - 1
        Npoints, Nh = S.shape
        ## transition rates
        P = psum(S2.reshape(Np, -1).T).reshape(Nh, Nh)
        P /= sum(P)
        P33 = zeros((3, 3))
        P33[:2, :2] = P[:2, :2]
        for i in (0, 1):
            P33[i, 2] = sum(P[i, 2:])
            P33[2, i] = sum(P[2:, i])
        P33[2, 2] = sum(P[2:, 2:])
        pssFull = psum(S.T)
        pssFull /= sum(pssFull)
        pss = array((pssFull[0], pssFull[1], sum(pssFull[2:])))
        Popt = (P33<=P33.T)*P33 + (P33>P33.T)*P33.T
        # Popt = P33
        T = Popt/pss
        k1, k2, k3, k4 = T[1, 0], T[0, 1], T[2, 1], T[1, 2]
        ## swim speed and diffusivities
        mV, Dtrap, Dtum, Dswim = 0., 0., 0., 0.
        NmV, NDtrap, NDtum, NDswim = 0., 0., 0., 0.
        for p in arange(Np):
            a, b = pInds[p:p+2]
            Sp = S[a:b]
            dxp = DXarray[a:b]
            NmV += sum(Sp*self.vmag)
            for n in arange(2, Nh):
                mV += sum(Sp[:, n]*psum(self.v[n]*dxp))
        mV /= NmV
        for p in arange(Np):
            a, b = pInds[p:p+2]
            Sp = S[a:b]
            dxp = DXarray[a:b]
            NDtrap += 2.*self.dim*sum(Sp[:, 0])
            NDtum += 2.*self.dim*sum(Sp[:, 1])
            NDswim += 2.*self.dim*sum(Sp[:, 2:])
            Dtrap += sum(Sp[:, 0]*psum(dxp**2))
            Dtum += sum(Sp[:, 1]*psum(dxp**2))
            for n in arange(2, Nh):
                Dswim += sum(Sp[:, n]*psum((dxp - mV*self.v[n])**2))
        Dtrap /= NDtrap
        Dtum /= NDtum
        Dswim /= NDswim
        pars = array((k1, k2, k3, k4, mV, Dtrap, Dtum, Dswim))
        return pars
################################################################################
################################################################################

fourState_dtype = [('dim', int64),
                    ('Nh', int64),
                    ('vmag', float64[:]),
                    ('v', float64[:, :])]
@jitclass(fourState_dtype)
class FourStateModel(EMestimation):
    def __init__(self, dim, Nphi):
        """Initialize using the number of velocity states."""
        self.dim = dim
        Ntheta = Nphi*4
        self.Nh = Ntheta*Nphi + 3
        Nrun = Ntheta*Nphi
        theta, phi = linspace(0, 2*pi, Ntheta+1)[:-1], linspace(0, 0.5*pi, Nphi+1)[:-1]
        # THETA, PHI = meshgrid(theta, phi)
        THETA = zeros((Nphi, Ntheta))
        PHI = zeros((Nphi, Ntheta))
        for i in arange(Nphi):
            for j in arange(Ntheta):
                THETA[i, j] = theta[j]
                PHI[i, j] = phi[i]
        self.vmag = zeros(self.Nh)
        if dim == 2:
            v = zeros((self.Nh, 2))
            v[3:, 0] = (cos(PHI)*cos(THETA)).flatten()
            v[3:, 1] = (cos(PHI)*sin(THETA)).flatten()
            self.vmag[3:] = (cos(PHI)**2).flatten()
        elif dim == 3:
            v = zeros((self.Nh, 3))
            v[3:, 0] = (cos(PHI)*cos(THETA)).flatten()
            v[3:, 1] = (cos(PHI)*sin(THETA)).flatten()
            v[3:, 2] = sin(PHI).flatten()
            self.vmag[3:] = ones_like(PHI).flatten()
        else:
            assert dim == 2 or dim == 3
        self.v = v
    def pss(self, pars):
        k1, k2, k3, k4, k5, k6 = pars[:6]
        A = array(((-k1, k2, 0., 0.), (k1, -k2-k3, k4, 0.),
                   (0., k3, -k4-k5, k6), (0., 0., k5, -k6)))
        u, s, v = svd(A)
        pss = v[-1, :]
        pss /= sum(pss)
        return pss
    def Pinit(self, Np, pars):
        """Initialize hidden state distribution for each path."""
        # pinit = self.pss(pars)
        p0 = ones((Np, self.Nh))/self.Nh
        # p0[:, 0] = pinit[0]
        # p0[:, 1] = pinit[1]
        # p0[:, 2:] = pinit[2]/(self.Nh - 2)
        return p0
    def getQ(self, pars):
        """Generate the propagator matrix for the hidden state Markov process."""
        k1, k2, k3, k4, k5, k6 = pars[:6]
        A = array(((-k1, k2, 0., 0.), (k1, -k2-k3, k4, 0.),
                   (0., k3, -k4-k5, k6), (0., 0., k5, -k6)))
        Q = eye(4) + A
        Ap = dot(A, eye(4))
        for n in arange(2, 10):
            Ap = dot(Ap, A/n)
            Q += Ap
        return Q
    def UandT(self, dx, pars, Q):
        """Evaluate the observation and hidden state probability matrices for a given path."""
        k1, k2, k3, k4, k5, k6 = pars[:6]
        mV, Dtrap, Dtum1, Dtum2, Dswim = pars[6:]
        Nt, _ = dx.shape
        U = zeros((Nt, self.Nh))
        Darr = array((Dtrap, Dtum1, Dtum2, Dswim))
        for n in arange(self.Nh):
            r = (dx - mV*self.v[n])**2
            D = Darr[min(3, n)]
            factor = (4.*pi*D)**(self.dim/2.)
            U[:, n] = exp(-psum(r)/(4.*D))/factor
        T0 = zeros((self.Nh, self.Nh))
        T0[:3, :3] = Q[:3, :3]
        T0[2, 3:] = Q[2, 3]
        T0[3:, 2] = Q[3, 2]/(self.Nh - 3)
        T0[3:, 3:] = eye(self.Nh - 3)*Q[3, 3]
        return U, T0
    def Mstep(self, DXarray, pInds, S, S2):
        """Extract maximimum likelihood parameters given hidden state distribution for the E step."""
        Np = pInds.size - 1
        Npoints, Nh = S.shape
        ## transition rates
        P = psum(S2.reshape(Np, -1).T).reshape(Nh, Nh)
        P /= sum(P)
        P44 = zeros((4, 4))
        P44[:3, :3] = P[:3, :3]
        for i in (0, 1, 2):
            P44[i, 3] = sum(P[i, 3:])
            P44[3, i] = sum(P[3:, i])
        P44[3, 3] = sum(P[3:, 3:])
        pssFull = psum(S.T)
        pssFull /= sum(pssFull)
        pss = array((pssFull[0], pssFull[1], pssFull[2], sum(pssFull[3:])))
        Popt = (P44<=P44.T)*P44 + (P44>P44.T)*P44.T
        # Popt = P33
        T = Popt/pss
        k1, k2, k3, k4 = T[1, 0], T[0, 1], T[2, 1], T[1, 2]
        k5, k6 = T[3, 2], T[2, 3]
        ## swim speed and diffusivities
        mV, Dtrap, Dtum1, Dtum2, Dswim = 0., 0., 0., 0., 0.
        NmV, NDtrap, NDtum1, NDtum2, NDswim = 0., 0., 0., 0., 0.
        for p in arange(Np):
            a, b = pInds[p:p+2]
            Sp = S[a:b]
            dxp = DXarray[a:b]
            NmV += sum(Sp*self.vmag)
            for n in arange(3, Nh):
                mV += sum(Sp[:, n]*psum(self.v[n]*dxp))
        mV /= NmV
        for p in arange(Np):
            a, b = pInds[p:p+2]
            Sp = S[a:b]
            dxp = DXarray[a:b]
            NDtrap += 2.*self.dim*sum(Sp[:, 0])
            NDtum1 += 2.*self.dim*sum(Sp[:, 1])
            NDtum2 += 2.*self.dim*sum(Sp[:, 2])
            NDswim += 2.*self.dim*sum(Sp[:, 3:])
            Dtrap += sum(Sp[:, 0]*psum(dxp**2))
            Dtum1 += sum(Sp[:, 1]*psum(dxp**2))
            Dtum2 += sum(Sp[:, 2]*psum(dxp**2))
            for n in arange(3, Nh):
                Dswim += sum(Sp[:, n]*psum((dxp - mV*self.v[n])**2))
        Dtrap /= NDtrap
        Dtum1 /= NDtum1
        Dtum2 /= NDtum2
        Dswim /= NDswim
        pars = array((k1, k2, k3, k4, k5, k6, mV, Dtrap, Dtum1, Dtum2, Dswim))
        return pars
