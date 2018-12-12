from __future__ import division
from __future__ import print_function
from pylab import *
import pandas as pd
from numba import jit, jitclass
from numba import int64, int32, int16, float64, float32, int16
from scipy.linalg import logm, expm


@jit(nopython=True)
def psum(a):
    """Returns partial sum over all dimensions except the 0th."""
    Ndim = len(a.shape)
    N = a.shape[0]
    out = zeros(N)
    for n in arange(N):
        out[n] = sum(a[n])
    return out
@jit(nopython=True)
def pmean(a):
    """Compute the mean over all dimensions except the 0th."""
    Ndim = len(a.shape)
    N = a.shape[0]
    return psum(a)/N
@jit(nopython=True)
def Estep(p0, U, T):
    """Compute the hidden state distribution for a given path."""
    Nt, Nh = U.shape
    alpha = zeros((Nt, Nh))
    alpha[0] = U[0]*p0  #  P(S1)*P(Y1|S1)
    for t in arange(1, Nt):
        ## P[St, Y_{1:t}]
        alpha[t] = dot(T, alpha[t-1])*U[t, :]
        ## P[St | Y_{1:t}] = P[St, Y_{1:t}]/sum_{St}P[St, Y_{1:t}]
        alpha[t] = alpha[t]/sum(alpha[t])
    beta = zeros((Nt, Nh))
    beta[Nt-1, :] = 1.
    for t in arange(Nt-1)[::-1]:
        ## P[Y_{t+1:T} | St]
        beta[t] = dot(beta[t+1], T)*U[t+1]
        ## P[St | Y_{t+1:T}] = P[Y_{t+1:T} | St]/sum_{St}P[Y_{t+1:T} | St]
        beta[t] = beta[t]/sum(beta[t])
    ## P[St | Y_{1:T}] = P[Y_{t+1:T} | St]P[St, Y_{1:t}]
    ##                      / sum_{St}P[Y_{t+1:T}, St | Y_{1:t}]
    S0 = alpha*beta
    S = (S0.T/psum(S0)).T
    S2 = zeros((Nh, Nh))
    for t in arange(1, Nt):
        S2t = (U[t]*beta[t]*(T*alpha[t-1]).T).T
        S2 += S2t/sum(S2t)
    return S, S2
@jit(nopython=True)
def EstepLite(p0, U, T):
    """Perform Estep() without returning the transition distribution S2. Used
    for justE()."""
    Nt, Nh = U.shape
    alpha = zeros((Nt, Nh))
    alpha[0] = U[0]*p0
    for t in arange(1, Nt):
        alpha[t] = dot(T, alpha[t-1])*U[t, :]
        alpha[t] = alpha[t]/sum(alpha[t])
    beta = zeros((Nt, Nh))
    beta[Nt-1, :] = 1.
    for t in arange(Nt-1)[::-1]:
        beta[t] = dot(beta[t+1], T)*U[t+1]
        beta[t] = beta[t]/sum(beta[t])
    S0 = alpha*beta
    S = (S0.T/ psum(S0)).T
    return S
@jit(nopython=True)
def Viterbi(p0, U, T):
    """Compute maximum likelihood path after EM finishes."""
    Nt, Nh = U.shape
    alpha = zeros((Nt, Nh))
    T2 = zeros((Nt, Nh), int64)
    alpha[0] = U[0]*p0  #  P(S1)*P(Y1|S1)
    for t in arange(1, Nt):
        q = T*alpha[t-1]
        for n in arange(Nh):
            T2[t, n] = q[n].argmax()
        w = zeros(Nh)
        for n in arange(Nh):
            w[n] = q[n, T2[t, n]]
        alpha[t] = w*U[t]
    S = zeros(Nt, int64)
    S[Nt-1] = alpha[Nt-1].argmax()
    for t in arange(Nt-1)[::-1]:
        S[t] = T2[t+1, S[t+1]]
    return S
class EMestimation(object):
    """Estimate hidden states and model parameters from a set of paths."""

    def allViterbi(self, DXarray, pInds, S, pars):
        """Compute maximum likelihood paths after EM finishes."""
        Npoints, _ = DXarray.shape
        Np = pInds.size - 1
        HS_ML = empty(Npoints, dtype=int16)
        Q = self.getQ(pars)
        for p in arange(Np):
            a, b = pInds[p:p+2]
            p0 = S[a]
            dx = DXarray[a:b]
            U, T = self.UandT(dx, pars, Q)
            HS_ML[a:b] = Viterbi(p0, U, T)
        return HS_ML
    def allMAP(self, DXarray, pInds, S, pars):
        """Extract MAP tracks from EM output, i.e., from max_S P(S_t| {X_1:T},
        {I_1:T})."""
        Npoints, _ = DXarray.shape
        Np = pInds.size - 1
        HS_ML = empty(Npoints, dtype=int16)
        for p in arange(Np):
            a, b = pInds[p:p+2]
            for n in arange(a, b):
                HS_ML[n] = S[n].argmax()
        return HS_ML
    def EM(self, DXarray, pInds, N, pars):
        """Evaluate the expectation-maximization algorithm with N iterations."""
        Np = pInds.size - 1
        Npoints, _ = DXarray.shape
        p0 = self.Pinit(Np, pars)
        S = empty((Npoints, self.Nh))
        S2 = empty((Np, self.Nh, self.Nh))
        for n in arange(N):
            Q = self.getQ(pars)
            for p in arange(Np):
                a, b = pInds[p:p+2]
                dx = DXarray[a:b]
                Nt, _ = dx.shape
                if dx.size == self.dim: # if only one point in path
                    Sp = zeros((1, self.Nh))
                    Sp[0, 0] = 1.
                    S[a:b] = Sp
                    p0[p] = Sp
                    continue
                U, T = self.UandT(dx, pars, Q)
                Sp, S2p = Estep(p0[p], U, T)
                S[a:b] = Sp
                p0[p] = S[a]
                S2[p] = S2p
            pars = self.Mstep(DXarray, pInds, S, S2)
        return pars, S
    def justE(self, DXarray, pInds, N, pars):
        """Compute the hidden states with fixed parameters."""
        Npoints, _ = DXarray.shape
        Np = pInds.size - 1
        p0 = self.Pinit(Np, pars)
        Q = self.getQ(pars)
        S = empty((Npoints, self.Nh))
        for n in arange(N):
            for p in arange(Np):
                a, b = pInds[p:p+2]
                dx = DXarray[a:b]
                if dx.size == self.dim: # if only one point in path
                    Sp = zeros((1, self.Nh))
                    Sp[0, 1] = 1.
                    S[a:b] = Sp
                    p0[p] = Sp
                    continue
                U, T = self.UandT(dx, pars, Q)
                S[a:b] = EstepLite(p0[p], U, T)
                p0[p] = S[a]
        return S
    def Pinit(self, Np, pars):
        """Initialize hidden state distribution for each path.
        Returns an array of shape (Np, Nh), normalized to a probability
        distribution along axis=1."""
        raise NotImplementedError
    def Mstep(self, DXarray, pInds, S, S2):
        """Extract maximimum likelihood parameters given hidden state
        distribution for the E step. Returns a tuple of model parameter
        values."""
        raise NotImplementedError
    def UandT(self, dx, pars, Q):
        """Evaluate the observation and hidden state probability matrices for
        a given path. Returns (Nt x Nh) array U and (Nh x Nh) array T."""
        raise NotImplementedError
    def getQ(self, pars):
        """Generate the propagator matrix for the hidden state Markov
        process."""
        raise NotImplementedError
################################################################################
TwoState_dtype = [('NVstates', int64),
                 ('Nh', int64),
                 ('v', float64[:, :]),
                 ('vmag2', float64[:])]
@jitclass(TwoState_dtype)
class TwoStateMixture(EMestimation):
    def __init__(self, NVstates):
        """Initialize using the number of velocity states."""
        self.NVstates = NVstates
        self.Nh = 2*self.NVstates
        theta = linspace(0, 2*pi, NVstates+1)[:-1]
        self.v = zeros((self.Nh, 2))
        self.v[NVstates:, 0] = cos(theta)
        self.v[NVstates:, 1] = sin(theta)
        self.vmag2 = zeros(self.Nh)
        self.vmag2[NVstates:] = 1.0
    def pss(self, pars):
        k1, k2 = pars[0], pars[1]
        return array([k2/(k1 + k2), k1/(k1 + k2)])
    def Pinit(self, Np, pars):
        """Initialize hidden state distribution for each path."""
        k1, k2 = pars[0], pars[1]
        p0 = ones((Np, self.Nh))
        p0[:, :self.NVstates] = k2/(k1 + k2)/self.NVstates
        p0[:, self.NVstates:] = k1/(k1 + k2)/self.NVstates
        return p0
    def getQ(self, pars):
        """Generate the propagator matrix for the hidden state Markov
        process."""
        k1, k2 = pars[0], pars[1]
        A = array(((-k1, k2), (k1, -k2)))
        Q = eye(2) + A
        Ap = dot(A, eye(2))
        for n in arange(2, 10):
            Ap = dot(Ap, A/n)
            Q += Ap
        return Q
    def UandT(self, dx, pars, Q):
        """Evaluate the observation and hidden state probability matrices for a
        given path."""
        k1, k2, mV, D = pars
        Nt, _ = dx.shape
        U = zeros((Nt, self.Nh))
        for n in arange(self.Nh):
            r = (dx - mV*self.v[n])**2
            U[:, n] = exp(-psum(r)/(4.*D))/(4.*pi*D)
        I = eye(self.NVstates)
        T = zeros((self.Nh, self.Nh))
        T[:self.NVstates, :self.NVstates] = I*Q[0, 0]
        T[:self.NVstates, self.NVstates:] = I*Q[0, 1]
        T[self.NVstates:, :self.NVstates] = I*Q[1, 0]
        T[self.NVstates:, self.NVstates:] = I*Q[1, 1]
        return U, T
    def Mstep(self, DXarray, pInds, S, S2):
        """Extract maximimum likelihood parameters given hidden state
        distribution for the E step."""
        Np = pInds.size - 1
        Npoints, Nh = S.shape
        P = pmean(S2.reshape(Np, -1).T).reshape(Nh, Nh)
        P2 = array([[mean(diag(P[:self.NVstates, :self.NVstates])),
                        mean(diag(P[:self.NVstates, self.NVstates:]))],
                    [mean(diag(P[self.NVstates:, :self.NVstates])),
                        mean(diag(P[self.NVstates:, self.NVstates:]))]])
        pssFull = pmean(S.T)
        pss = array([sum(pssFull[:self.NVstates]), sum(pssFull[self.NVstates:])])
        P2opt1 = (P2<=P2.T)*P2 + (P2>P2.T)*P2.T
        T = P2opt1/pss
        k1, k2 = T[1, 0], T[0, 1]
        mV, D = 0., 0.
        NmV, ND = 0., 0.
        for p in arange(Np):
            a, b = pInds[p:p+2]
            NmV += sum(S[p]*self.vmag2)
            ND += 4.*sum(S[p].shape[0])
            for n in arange(Nh):
                mV += sum(S[p][:, n]*psum(self.v[n]*dx[p]))
                D += sum(S[p][:, n]*psum((dx[p] - mV*self.v[n])**2))
        mV /= NmV
        for p in arange(Np):
            a, b = pInds[p:p+2]
            ND += 4.*sum(S[p].shape[0])
            for n in arange(Nh):
                D += sum(S[p][:, n]*psum((dx[p] - mV*self.v[n])**2))
        D /= ND
        return k1, k2, mV, D
