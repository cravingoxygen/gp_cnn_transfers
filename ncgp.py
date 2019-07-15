#import magma
from ctypes import (CDLL, c_uint, POINTER, c_int64, c_float, byref)
import ctypes
from memoized_property import memoized_property
import numpy as np
from von_mises import standard_gaussian_atoms
import torch

from scipy.special import expit as sigmoid


def diag_add(K, diag):
    if isinstance(K, torch.Tensor):
        K.view(K.numel())[::K.shape[-1]+1] += diag
    elif isinstance(K, np.ndarray):
        K.flat[::K.shape[-1]+1] += diag
    else:
        raise TypeError("What do I do with a `{}`, K={}?".format(type(K), K))


np_float_type = np.dtype(np.float64)
torch_float_type = torch.float32
to_np_float_type = torch.float64
# change `cuda` to 'cpu' if no GPUs available. That will break `magma.posv`
# though, because it wants `torch.Tensor`s to be in GPU
cuda = 'cuda'

class uplo:
    upper         = 121
    lower         = 122
    full          = 123  # lascl, laset
    hessenberg    = 124  # lascl

    safe_mode = False

    @classmethod
    def check(klass, M, t):
        if klass.safe_mode:
            M = np.reshape(M, (-1,) + M.shape[-2:])
            if t == klass.upper:
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        assert not np.any(np.isnan(M[i, j, j:])), (
                            "M is not upper triangular")
            elif t == klass.lower:
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        assert not np.any(np.isnan(M[i, j, :j+1])), (
                            "M is not lower triangular")
            elif t == klass.full:
                assert not np.any(np.isnan(M)), (
                            "M is not full")


    @classmethod
    def enforce(klass, M, t):
        if klass.safe_mode:
            M = np.reshape(M, (-1,) + M.shape[-2:])
            if t == klass.upper:
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        M[i, j, :j] = np.nan
            elif t == klass.lower:
                for i in range(M.shape[0]):
                    for j in range(M.shape[1]):
                        M[i, j, j+1:] = np.nan


def posv(A, b, lower=False):
    """
    Solve linear system `Ax = b`, using the Cholesky factorisation of `A`.
    `A` must be positive definite.
    `A` and `b` will be overwritten during the routine, with the Cholesky
    factorisation and the solution respectively.
    Only the strictly (lower/upper) triangle of `A` is accessed.
    """
    #import pdb; pdb.set_trace()
    #info = magma_int()
    if not (A.shape[0] == A.shape[1]
            and A.shape[1] == b.shape[0]):
        raise ValueError("Bad shape of A, b ({}, {})".format(A.shape, b.shape))
    uplo_A = uplo.lower if lower else uplo.upper
    uplo.check(A, uplo_A)

    if isinstance(A, np.ndarray):
        assert isinstance(b, np.ndarray), "A and B have to be of the same type"
        A_ptr, b_ptr = A, b
    elif isinstance(A, torch.Tensor):
        assert A.t().is_contiguous(), "`A` is not in Fortran format"
        #assert 'cuda' in A.device.type, "`A` is not in GPU, use Numpy version."
        assert b.t().is_contiguous(), "`b` is not in Fortran format"
        #assert 'cuda' in b.device.type, "`b` is not in GPU, use Numpy version."
        assert A.is_cuda == b.is_cuda
        #A_ptr = ctypes.cast(A.data_ptr(), POINTER(c_float))
        #b_ptr = ctypes.cast(b.data_ptr(), POINTER(c_float))

    #args = [uplo_A,
    #        A.shape[0], b.shape[1],
    #        A_ptr, A.shape[0],
    #        b_ptr, b.shape[0]]
    #f = by_dtype_posv[dtype_iscuda(A)]
    #f(*args, byref(info))

    #info = info.value
    #if info < 0:
    #    raise ValueError("Illegal {}th argument: {} = {}"
    #                     .format(-info, f.argnames[-info], args[-info]))
    #if info > 0:
    #    raise ValueError("Error code: {}".format(info))
    uplo.enforce(A, uplo_A)
    return b


class AP:
    """
    Represents an approximate posterior
    """
    def __init__(self, K, L=None, l=None):
        if isinstance(K, AP):
            self.K = K.K
            self.N = K.N
            self._x_stored = K._x_stored
            self._lK_stored = K._lK_stored
        else:
            self.K = torch.from_numpy(K).to(
                device=cuda, dtype=torch_float_type)
            self.N = K.shape[0]
            self._x_stored = torch.empty_like(self.K)
            self._lK_stored = torch.empty_like(self.K)

        if L is None:
            L = np.ones([self.N], dtype=np_float_type) * 0.5
            self._invK_Sigma = self._x_stored
            self._invK_Sigma[...] = 0.
            self._diag_Sigma = self.to_numpy_flat(self.K.diag())
        if l is None:
            l = np.ones([self.N], dtype=np_float_type) * 0.25
            self._mu = np.zeros([self.N], dtype=np_float_type)
        self.L = L
        self.l = l

    @memoized_property
    def sqrtL(self):
        L = torch.from_numpy(self.L).to(device=cuda, dtype=torch_float_type)
        torch.nn.functional.relu_(L)
        L.sqrt_()
        return L[:, None] #L.view([L.numel(), 1])

    @memoized_property
    def invK_Sigma(self):
        #                                  K
        x = self._x_stored
        lK = self._lK_stored
        #                     lK = L^{1/2} K
        torch.mul(self.K, self.sqrtL, out=lK)
        #                      x = L^{1/2} K L^{1/2}
        torch.mul(lK, self.sqrtL.t(), out=x)
        #                      I + L^{1/2} K L^{1/2}
        diag_add(x, 1.) 
        #        out = lK^T = (I + L^{1/2} K L^{1/2})^{-1} L^{1/2} K
        # need x.t() and lK.t() for Fortran format.
        out = posv(x.t(), lK.t()) #unicorns
        out = lK.t()
        #                     (I + L^{1/2} K L^{1/2})^{-1} L^{1/2} K
        out *= -self.sqrtL
        #           - L^{1/2} (I + L^{1/2} K L^{1/2})^{-1} L^{1/2} K
        diag_add(out.t(), 1.)  # need to be contiguous
        #         I - L^{1/2} (I + L^{1/2} K L^{1/2})^{-1} L^{1/2} K
        return out

    @memoized_property
    def diag_Sigma(self):
        # diag(K - K L^{1/2} (I + L^{1/2} K L^{1/2})^{-1} L^{1/2} K)
        dS = self.K[:, None, :] @ self.invK_Sigma.t()[:, :, None]
        # ensure none is less than 0
        torch.nn.functional.threshold_(dS, 1e-12, 1e-12)
        return self.to_numpy_flat(dS)

    @memoized_property
    def invK_mu(self):
        # l = sigma^{-1} mu
        #sigmainv_mu = torch.tensor(
        #    self.l[:, None], dtype=torch_float_type, device=cuda)
        #torch.mm(self.invK_Sigma, sigmainv_mu, out=sigmainv_mu)
        #return sigmainv_mu
        return self.invK_Sigma @ torch.tensor(self.l[:, None], dtype=torch_float_type, device=cuda)

    @memoized_property
    def mu(self):
        return self.to_numpy_flat(self.K @ self.invK_mu)

    @staticmethod
    def to_numpy_flat(array):
        return array.to(device='cpu', dtype=to_np_float_type).numpy().ravel()


a, prob = standard_gaussian_atoms(5)
a = a[:, None]
prob = prob[:, None]


class ClassifierLikelihood():
    def __init__(self, y):
        """
        `y`: labels
        """
        # y \in {0, 1}
        self.y = y
        # yh \in {-1, 1}
        self.yh = 2*y - 1

    def g(self, mu):
        """
        Returns the gradient of the likelihood function at `mu`.
        """
        raise NotImplementedError()

    def H(self, mu):
        """
        Returns the Hessian of the likelihood function at `mu`.
        """
        raise NotImplementedError()

    def log_prob(self, z):
        """
        Returns the log probability of the likelihood function for some data,
        given the values `z` of the latent function.
        """
        raise NotImplementedError()

    def update_posterior(self, ap, alpha, L, l):
        if np.equal(alpha, 1):
            return AP(ap, L, l)
        return AP(ap, (1-alpha)*ap.L + alpha*L, (1-alpha)*ap.l + alpha*l)

    def laplace(self, ap, alpha=1):
        """
        Requires: g, H
        """
        L = -self.H(ap.mu)
        l = self.g(ap.mu) + L*ap.mu
        return self.update_posterior(ap, alpha, L, l)

    def vi(self, ap, alpha=1):
        """
        Requires: log_prob
        """
        #Moments of the current approximate posterior
        mu = ap.mu
        s2 = ap.diag_Sigma
        s = np.sqrt(s2)

        #Locations to evaluate the integrand
        z = mu + a * s
        #nz = (z - mu)/s2
        nz = a / s

        lp = self.log_prob(z)

        l = (prob * nz * lp).sum(0)
        L = -0.5*(prob * (nz**2 - 1/s2) * lp).sum(0)
        return self.update_posterior(ap, alpha, L, l)

    def ep(self, ap, alpha=1):
        """
        Requires: log_prob
        """

        #Moments of the current approximate posterior
        mu = ap.mu
        mu2 = mu**2
        s2 = ap.diag_Sigma
        s = np.sqrt(s2)

        #Locations to evaluate the integrand
        z = mu + a * s
        z2 = z**2

        prob_ratio = np.exp(self.log_prob(z)-ap.l*z+ap.L*z2/2)
        z0 = (prob*prob_ratio   ).sum(0)

        mu_next = (prob*prob_ratio*(z)).sum(0) / z0
        s2_next = (prob*prob_ratio*(z-mu_next)**2).sum(0) / z0

        #Natural parameters of new approximate posterior
        L = ap.L +       1/s2_next -  1/s2
        l = ap.l + mu_next/s2_next - mu/s2
        return self.update_posterior(ap, alpha, L, l)


class SigmoidClassifierLikelihood(ClassifierLikelihood):
    def g(self, mu):
        return self.yh * sigmoid(-self.yh*mu)

    def H(self, mu):
        return -sigmoid(mu) * sigmoid(-mu)

    def log_prob(self, z):
        """
        log_sigmoid(x) = -log(1+e^(-x))
                    = -log_sum_exp(0, -x)

        returns self.y*logsigmoid(z) + (1-self.y)*logsigmoid(-z)
        """
        return (self.y-1)*np.logaddexp(0, z) - self.y*np.logaddexp(0, -z)


if __name__ == "__main__":
    scl = SigmoidClassifierLikelihood(np.array([0., 0, 1]))
    ap_l = AP(np.eye(3)+np.ones([3, 3])/3)
    ap_v = AP(np.eye(3)+np.ones([3, 3])/3)
    ap_e = AP(np.eye(3)+np.ones([3, 3])/3)

    for i in range(100):
        ap_l = scl.laplace(ap_l, alpha=1)
        ap_v = scl.vi(ap_v, alpha=1)
        ap_e = scl.ep(ap_e, alpha=1)

    print(ap_l.mu)
    print(ap_v.mu)
    print(ap_e.mu)

    print(ap_l.diag_Sigma)
    print(ap_v.diag_Sigma)
    print(ap_e.diag_Sigma)
