from numpy import pi, sqrt, exp, cos, real, asarray, fromiter, zeros, double
from numpy.fft import ifft
from scipy.special import kv, gamma, binom, bernoulli
from itertools import chain
import numpy as np
cimport numpy as np

#====== QUADRATURE ============================================================#

class fejer2:
    """Constructs and stores abscissae and weights for a Fejer type quadrature.
    """

    def __init__(self):
        self.abscissae = [asarray([0], double),
            asarray([1. / sqrt(2), 0, -1. / sqrt(2)], double)]
        self.weights = [asarray([2], double),
            asarray([2. / 3., 2. / 3., 2. / 3.], double)]

    def extend(self):
        n = 2 * (len(self.abscissae[-1]) + 1)
        self.abscissae.append(fromiter((cos(k * pi / n) for k in range(1, n)),
            double))
        v = [2. / (1 - 4 * k**2) for k in range(n//2)]
        v = v + [(n - 3.)/(n - 1) - 1] + v[-1:0:-1]
        self.weights.append(real(ifft(v))[1:])

    def error(self, n, eval):
        high = sum(self.weights[n] * eval)
        low  = sum(self.weights[n - 1] * eval[1::2])
        return abs(high - low) / abs(high)

fejer = fejer2()

#====== BUCHHOLZ POLYNOMIALS ==================================================#

class buchholz_poly():
    """Constructs and stores Buchholz polynomials of arbitrary parameters."""

    def __init__(self):
        self.farray = np.ones((1, 1), double)
        self.garray = np.ones((2, 1), double)
        self.garray[1, 0] = 1. / 6

    def next_f(self):
        n = len(self.farray)
        tmp = np.zeros((n + 1, n + 1), double)
        tmp[:n, :n] = self.farray
        self.farray = tmp
#       self.farray.resize((n + 1, n + 1))
        for m in range(n):
            tmp = (binom(2 * n - 1, 2 * m) *
                   abs(bernoulli(2 * (n - m))[-1]) *
                   4**(n - m) / (n - m))
            for k in range(m + 1):
                self.farray[-1, k] += tmp * self.farray[m, k]
                self.farray[-1, k + 1] += - tmp * self.farray[m, k] / 2

    def next_g(self):
        n = len(self.garray)
        tmp = np.zeros((n + 1, (n + 1) // 2), double)
        tmp[:self.garray.shape[0], :self.garray.shape[1]] = self.garray
        self.garray = tmp
#       self.garray.resize((n + 1, (n + 1) // 2))
        for m in range(n - 1, -1, -2):
            tmp = (binom(n - 1, n - m - 1) *
                    abs(bernoulli(n - m + 1)[-1]) *
                    2**(n - m) / (n - m + 1))
            for k in range(max(1, (m + 1) // 2)):
                self.garray[-1, (n + 2**m)%2 + k] += tmp * self.garray[m, k]

    def p(self, int n, int b, double z):
        cdef int binom_tmp = 1
        cdef double total, term, tmp
        cdef int i, s

        while len(self.farray) < n//2 + 1:
            self.next_f()
        while len(self.garray) < n + 1:
            self.next_g()

        cdef np.double_t[:,:] f_view = self.farray
        cdef np.double_t[:,:] g_view = self.garray

        if n > 0:
            total = 0
            for i in range((n + 1)//2):
                total -= (-1)**i * z**(2 * i + 2 - n%2) * g_view[n, i]
        else:
            total = 1

        for s in range(1, n//2 + 1):
            tmp = 0
            for i in range(s + 1):
                tmp += b**i * f_view[s, i]
            term = tmp

            tmp = 0
            for i in range((n - 2 * s + 1) // 2):
                tmp -= ((-1)**i * z**(2 * i + 2 - (n - 2 * s)%2) *
                        g_view[n - 2 * s, i])
            term *= tmp

            binom_tmp *= ((n - 2 * s - 2) * (n - 2 * s - 3) /
                                    (2 * s * (2 * s - 1)))
            term *= binom_tmp

            total += term

        for i in range(n):
            total *= z / (i + 1)

        return (-1)**(n + n//2) * total

buchholz = buchholz_poly()

#====== HYPERGEOMETRIC U FUNCTION =============================================#

def hyperu_lowb(int a, int b, double z):
    """Return an array of [U(0,b,z), ..., U(a-1,b,z), U(a,b,z)].

    List contains all functions with first parameter <= a,
    second parameter = b. a should be much larger than b.
    Uses backward recursion. Should be robust for a wide 
    range of real z > 0.
    """
    thresh = 1e-13
    cdef int n, i, tmpa
    cdef np.ndarray[np.double_t, ndim=1] output

    if z < 8:

        # Force the recurrence to start from a high a
        # parameter. Reduces stress on the asymptotic
        # series.
        tmpa = a if a > 50 else 50
        output = zeros(tmpa + 1, double)

        # Compute the starting values from an asymptotic
        # expansion in a
        arg = sqrt((4 * tmpa - 2 * b) * z)
        n = 0
        summand = ((-1)**n * buchholz.p(n, b, z) * kv(b + n - 1, arg) *
                   arg**(1 - b - n))
        output[tmpa] = summand
        while abs(summand / output[tmpa]) > thresh:
            n = n + 1
            last_summand = summand
            summand = ((-1)**n * buchholz.p(n, b, z) * kv(b + n - 1, arg) *
                       arg**(1 - b - n))
            if abs(summand) > abs(last_summand): break
            output[tmpa] += summand
        output[tmpa] = 2**b * exp(z / 2) * output[tmpa] / gamma(tmpa - b + 1)

        arg = sqrt((4 * (tmpa - 1) - 2 * b) * z)
        n = 0
        summand = ((-1)**n * buchholz.p(n, b, z) * kv(b + n - 1, arg) *
                   arg**(1 - b - n))
        output[tmpa - 1] = summand
        while abs(summand / output[tmpa - 1]) > thresh:
            n = n + 1
            summand = (-1)**n
            summand = summand * buchholz.p(n, b, z)
            summand = summand * kv(b + n - 1, arg)
            summand = summand * arg**(1 - b - n)
            output[tmpa - 1] = output[tmpa - 1] + summand
        output[tmpa - 1] = (2**b * exp(z / 2) * output[tmpa - 1] /
                            gamma(tmpa - b))

        # Generate the remaining values by backward recurrence
        i = tmpa - 2
        for i in range(tmpa - 2, -1, -1):
            output[i] = (- (b - 2 * (i + 1) - z) * output[i + 1] -
                         (i + 1) * (i - b + 2) * output[i + 2])

        output = (z**(1-b) / output[b-1]) * output

        return output[0:a + 1]

    else:
        # Use Miller's algorithm
        tmpa = a + 50 if a < 50 else 2 * a
        output = zeros(tmpa + 1, double)

        output[-1] = 0
        output[-2] = 1
        i = tmpa - 2
        for i in range(tmpa - 2, -1, -1):
            if output[i + 1] > 1e100:
                output[i + 1:] = output[i + 1:] / output[i + 1]
            output[i] = (- (b - 2 * (i + 1) - z) * output[i + 1] -
                        (i + 1) * (i - b + 2) * output[i + 2])

        output = (z**(1-b) / output[b-1]) * output

        return output[0:a + 1]

def hyperu_summands(int m, double z):
    cdef np.ndarray[np.double_t, ndim=2] output = zeros((2 * m - 1, m), double)
    cdef int a, i

    # Generate starting values for forward recurrence in b
    output[:,0] = hyperu_lowb(2 * m + 1, 3, z)[3:]
    output[:,1] = hyperu_lowb(2 * m + 1, 5, z)[3:]

    # Forward recurrence in b
    for a in range(3, 2 * m + 2):
        for i in range(3, (a - 1)//2 + 1):
            output[a - 3][i - 1] = ((2 * i - 1 + z) *
                (2 * i - 2 + z - z * (2 * i - 1 - a) / (2 * i - 1 + z) - 
                    z * (2 * i - 2 - a) / (2 * i - 3 + z)) * 
                output[a - 3][i - 2] / z**2 - 
                (2 * i - 1 + z) * (2 * i - 2 - a) * (2 * i - 3 - a) * 
                output[a - 3][i - 3] / (z**2 * (2 * i - 3 + z))
                )

    return output

def hyperu_llrr_summands(m, z):
    cdef np.ndarray[np.double_t, ndim=2] output = zeros((m - 3, m // 2 - 1), double)
    cdef int a, i

    # Generate starting values for forward recurrence in b
    output[:,0] = hyperu_lowb(m + 2, 6, z)[6:]
    output[:,1] = hyperu_lowb(m + 2, 8, z)[6:]

    # Forward recurrence in b
    for a in range(4, m + 1):
        for i in range(4, a//2 + 1):
            output[a - 4][i - 2] = (
                ((2 * i + z) / z**2) * ((2 * i + z - 1) - 
                    z * (2 * i - a - 3) / (2 * i + z - 2) - 
                    z * (2 * i - a - 2) / (2 * i + z)) * 
                output[a - 4][i - 3] -
                ((2 * i + z) / z**2) * ((2 * i - a - 3) * 
                    (2 * i - a - 4) / (2 * i + z - 2)) * 
                output[a - 4][i - 4]
                )

    return output

#====== LEGENDRE FUNCTION =====================================================#

def legendre_p(max_n, x):
    output = zeros(max_n, double)
    output[0] = np.float64(3 - 3 * np.float128(x)**2)
    output[1] = 5 * x * output[0]
    for n in range(3, max_n):
        output[n - 1] = (output[n - 2] * (2 * n + 1) * x -
                         output[n - 3] * (n + 2)) / (n - 1)
    return output




