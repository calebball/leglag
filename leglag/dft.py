from itertools import chain, zip_longest, islice
from scipy.special import hyp2f1, gamma
from numpy.linalg import norm
from numpy.polynomial.laguerre import laggauss
import numpy as np
from leglag.utilities import fejer

thresh = 1e-6

#====== HANDLER FUNCTIONS =====================================================#

def dft_domain_energy(domain, functional="glda", fit=None, parameters=None,
        quad_check=True, quad_start=1):
    """Computes a Density Functional correlation estimate for an electron domain.

    Arguments:
        domain      (OneDDomain)    electron domain to compute energy in
    Optional:
        functional  (str)   density functional type (default: glda)
        fit         (str)   density functional fit (default: type's default)
        parameters  (str)   extra parameters for density functional evaluation
        quad_check  (bool)  flag for `adaptive' quadrature
        quad_start  (int)   initial quadrature degree
    """

    try:
        if domain.side:
            def weights(quadn):
                while len(fejer.weights) <= quadn: fejer.extend()
                for x, w in zip(fejer.abscissae[quadn], fejer.weights[quadn]):
                    t = (x + 1) / 2
                    yield 0.5 * w / (1 - t)**2

            def abscissae(quadn):
                while len(fejer.abscissae) <= quadn: fejer.extend()
                for x in fejer.abscissae[quadn]:
                    t = (x + 1) / 2
                    yield domain.position + t / (1 - t)

        else:
            def weights(quadn):
                while len(fejer.weights) <= quadn: fejer.extend()
                for x, w in zip(fejer.abscissae[quadn], fejer.weights[quadn]):
                    t = (x + 1) / 2
                    yield 0.5 * w / t**2

            def abscissae(quadn):
                while len(fejer.abscissae) <= quadn: fejer.extend()
                for x in fejer.abscissae[quadn]:
                    t = (x + 1) / 2
                    yield domain.position - (1 - t) / t

    except AttributeError:
        centroid = sum(domain.position) / 2
        width = abs(domain.position[0] - domain.position[1]) / 2
        def weights(quadn):
            while len(fejer.weights) <= quadn: fejer.extend()
            for w in fejer.weights[quadn]:
                yield width * w
        def abscissae(quadn):
            while len(fejer.abscissae) <= quadn: fejer.extend()
            for x in fejer.abscissae[quadn]:
                yield centroid + width * x

    quadn = quad_start

    if functional == "glda":
        if domain.electrons < 2: return 0
        if fit:
            def f(rho, eta):
                return glda(rho, eta, fit=fit)
        else:
            f = glda
        par_func = domain.eta
        def par_func(x):
            return domain.eta(x, return_rho=True)

    elif functional == "gsblda":
        if domain.electrons < 2: return 0
        if fit:
            def f(rho, eta):
                return gsblda(rho, eta, fit=fit)
        else:
            f = gsblda
        par_func = domain.eta
        def par_func(x):
            return domain.eta(x, return_rho=True)

    elif functional == "0lda":
        if domain.electrons < 2: return 0
        if fit:
            def f(rho, eta):
                return zerolda(rho, eta, fit=fit)
        else:
            f = zerolda
        def par_func(x):
            return domain.eta(x, return_rho=True)

    if functional == "alphalda":
        if domain.electrons < 2: return 0
        if fit:
            def f(rho, eta, alpha):
                return alphalda(rho, eta, alpha, fit=fit)
        else:
            f = alphalda
        def par_func(x):
            tmp = domain.eta(x, return_rho=True)
            return (tmp[0], tmp[1], parameters[0])

    elif functional == "lda":
        if domain.electrons < 1: return 0
        if fit:
            def f(rho):
                return lda(rho, fit=fit)
        else:
            f = lda
        def par_func(x): return (domain.rho(x),)

    elif functional == "sblda":
        if domain.electrons < 1: return 0
        if fit:
            def f(rho):
                return sblda(rho, fit=fit)
        else:
            f = sblda
        def par_func(x): return (domain.rho(x),)

    par_vec = [par_func(x) for x in abscissae(quadn)]
    current_result = sum(weight * p[0] * f(*p)
            for weight, p in zip(weights(quadn), par_vec))

    if quad_check:

        error_dropping = []

        last_result = sum(weight * p[0] * f(*p)
                for weight, p in zip(fejer.weights[quadn - 1], par_vec[1::2]))
        error = abs(current_result - last_result)
        if current_result > 0: error /= abs(current_result)

        while quadn < 10:
            quadn += 1

            last_result = current_result
            last_error = error

            it = islice(abscissae(quadn), 0, 2**(quadn + 1) - 1, 2)
            par_vec = [p for p in chain.from_iterable(zip_longest(
                [par_func(x) for x in it], par_vec)) if p is not None]

            it = islice(abscissae(quadn), 0, 2**(quadn + 1) - 1, 2)
            current_result = sum(weight * p[0] * f(*p)
                    for weight, p in zip(weights(quadn), par_vec))

            error = abs(current_result - last_result)

            error_dropping.append(last_error > error)

    return current_result

#====== FUNCTIONAL EVALUATION =================================================#

def lda(rho, fit="loos2016"):
    """Evaluates an LDA type density functional.

    Arguments:
        rho (float) Electron density
    Optional:
        fit (str)   LDA functional to use (default: loos2016)
    """

    if rho < 10**(-100):
        return 0
    rs = 1 / (2 * rho)

    if fit == "gill2013":
        # 1D LDA fit from:
        # Loos P-F., Ball C. J., Gill P. M. W., "Uniform Electron Gases II: the
        # generalised local density approximation", J. Chem. Phys., (2013)
        a = - np.pi**2 / 360
        b = 0.75 - np.log(2 * np.pi) / 2
        g = 19 / 8
        return a * hyp2f1(1, 1.5, g, 2 * a * (1 - g) * rs / b)

    elif fit == "loos2016":
        # 1D LDA fit from:
        # Loos P-F., Ball C. J., Gill P. M. W., "", J. Chem. Phys., (2013)
#       k = 0.4182681382935852
        k = 0.414254
#       with np.errstate(over='raise'):
#           try:
#               t = (np.sqrt(k * rs + 0.25) - 0.5) / (k * rs)
#           except FloatingPointError:
#               return 0

        if rs < 10**(-20):
            return -0.0274156
        else:
            t = (np.sqrt(k * rs + 0.25) - 0.5) / (k * rs)

#       eps0 = - np.pi**2 / 360
#       eps1 = 0.00844
#       eta0 = 0.75 - np.log(np.sqrt(2 * np.pi))
#       eta1 = 0.35993316711936446

#       eps0 = -0.027415567780803773941
#       eps1 = 0.00845
#       eta0 = -0.16893853320467274178
#       eta1 = 0.359933

#       c = [k * eta0, 4 * k * eta0 + k**1.5 * eta1,
#               5 * eps0 + eps1 / k, eps0]

#       return sum(c[j] * t**(j + 2) * (1 - t)**(3-j) for j in range(4))

#       return (k * eta0 * t**2 * (1 - t)**3 +
#           (4 * k * eta0 + eta1 * k**1.5) * t**3 * (1 - t)**2 +
#           (5 * eps0 + eps1 / k) * t**4 * (1 - t) +
#           eps0 * t**5)

        return - (0.069983463134168500 * t**2 * (1 - t)**3 +
            0.183966893747766600 * t**3 * (1 - t)**2 +
            0.116679725669143600 * t**4 * (1 - t) +
            0.027415567780803773941 * t**5)

def glda(rho, eta, fit="gill2013"):
    """Evaluates a gLDA type density functional.

    Arguments:
        rho (float) Electron density
        eta (float) Curvature parameter (see Loos et. al. 2013)
    """

    if eta == 0:
        return 0
    elif eta < 1:
        rs = 1 / (2 * rho)
        a = ((1 - eta) * (np.log(1 - eta)**2 - 6 * np.log(1 - eta)) / 348 -
                eta * np.pi**2 / 360)
        b = ((0.75 - np.log(2 * np.pi) / 2) * eta -
                (1 - eta) * np.log(1 - eta) / 16)
        g = 19 / 16 * (4 - 3 * np.sqrt(1 - eta)) / (2 - np.sqrt(1 - eta))
        result = a * hyp2f1(1, 1.5, g, 2 * a * (1 - g) * rs / b)
        if np.isinf(result):
            print("in glda")
            print("{:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}".format(
                rho, eta, a, b, g, result))
        return result
    else:
        return lda(rho)

def zerolda(rho, eta, fit="gill2013"):
    """Evaluates a 0LDA type density functional.

    Arguments:
        rho (float) Electron density
        eta (float) Curvature parameter (see Loos et. al. 2013)
    """

    if eta == 0:
        return 0
    elif eta < 1:
        rs = 1 / (2 * rho)
        a = ((1 - eta) * (np.log(1 - eta)**2 - 6 * np.log(1 - eta)) / 348 -
                eta * np.pi**2 / 360)
        b = ((0.75 - np.log(2 * np.pi) / 2) * eta -
                (1 - eta) * np.log(1 - eta) / 16)
        g = 19 / 16 * (4 - 3 * np.sqrt(1 - eta)) / (2 - np.sqrt(1 - eta))
        result = a * hyp2f1(1, 1.5, g, 2 * a * (1 - g) * rs / b)
        if np.isinf(result):
            print("in glda")
            print("{:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}".format(
                rho, eta, a, b, g, result))
        return result
    else:
        return 0

def alphalda(rho, eta, alpha, fit="gill2013"):
    """Evaluates an aLDA type density functional.

    Arguments:
        rho     (float) Electron density
        eta     (float) Curvature parameter (see Loos et. al. 2013)
        alpha   (float) Coefficient of LDA fallback
    """

    if eta == 0:
        return 0
    elif eta < 1:
        rs = 1 / (2 * rho)
        a = ((1 - eta) * (np.log(1 - eta)**2 - 6 * np.log(1 - eta)) / 348 -
                eta * np.pi**2 / 360)
        b = ((0.75 - np.log(2 * np.pi) / 2) * eta -
                (1 - eta) * np.log(1 - eta) / 16)
        g = 19 / 16 * (4 - 3 * np.sqrt(1 - eta)) / (2 - np.sqrt(1 - eta))
        result = a * hyp2f1(1, 1.5, g, 2 * a * (1 - g) * rs / b)
        if np.isinf(result):
            print("in glda")
            print("{:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}".format(
                rho, eta, a, b, g, result))
        return result
    else:
        return alpha * lda(rho)

def sblda(rho, fit="loos2016"):
    """Evaluates an SBLDA type density functional.

    Arguments:
        rho (float) Electron density
    """

#   with np.errstate(divide='raise', over='raise'):
#       try:
#           rs = 1 / (2 * rho)
#       except FloatingPointError:
#           return 0

    if rho < 10**(-100):
        return 0
    rs = 1 / (2 * rho)

#   eta0 = 0.75 - np.log(np.sqrt(2 * np.pi))
    eta0 = -0.16893853320467274178

    a0 = - 0.06462278462550937
    a1 =   0.5350618556634981
    a2 = - 0.49071909740923203
    b0 =  53.117106369892795
    b1 =   1.5311441556615546
    b2 =   2.196057832619186

#   with np.errstate(over='raise'):
#       try:
#           den = b0 + b1 * rs**5 + b2 * rs**5.5 + rs**6
#       except FloatingPointError:
#           return lda(rho, fit='loos2016')
#       else:
#           num = a0 + a1 * rs + a2 * rs**2 - eta0 * rs**3
#           return lda(rho, fit="loos2016") + rs**2 * num / den

    if rs > np.double(1.0e+50): return lda(rho, fit=fit)
    den = b0 + b1 * rs**5 + b2 * rs**5.5 + rs**6
    num = a0 + a1 * rs + a2 * rs**2 - eta0 * rs**3
    return lda(rho, fit=fit) + rs**2 * num / den

def gsblda(rho, eta, fit="gill2013"):
    """Evaluates a generalised SBLDA type density functional.

    Arguments:
        rho (float) Electron density
        eta (float) Curvature parameter (see Loos et. al. 2013)
    """

    if eta == 0:
        return 0
    elif eta < 1:
        rs = 1 / (2 * rho)
        a = ((1 - eta) * (np.log(1 - eta)**2 - 6 * np.log(1 - eta)) / 348 -
                eta * np.pi**2 / 360)
        b = ((0.75 - np.log(2 * np.pi) / 2) * eta -
                (1 - eta) * np.log(1 - eta) / 16)
        g = 19 / 16 * (4 - 3 * np.sqrt(1 - eta)) / (2 - np.sqrt(1 - eta))
        result = a * hyp2f1(1, 1.5, g, 2 * a * (1 - g) * rs / b)
        if np.isinf(result):
            print("in glda")
            print("{:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}  {:10.6f}".format(
                rho, eta, a, b, g, result))
        return result
    else:
        return sblda(rho)

def gsblda(rho, eta, fit="gill2013"):
    """Evaluates a generalised SBLDA type density functional.

    Arguments:
        rho (float) Electron density
        eta (float) Curvature parameter (see Loos et. al. 2013)
    """

    if eta == 0:
        return 0
    elif eta < 1:
        if rho < 10**(-100):
            return 0
        rs = 1 / (2 * rho)
        sqrt1meta = np.sqrt(1 - eta)
        a = (- 34.88113088349792 + 34.88113088349792 * sqrt1meta +
                43.869851384127514 * eta) / (1 +
                1.348145530801427 * sqrt1meta -
                0.6721307918005824 * eta)
        b = (185.08839303941474 + 7.948419564703617 * sqrt1meta -
                174.90107312179194 * eta) / (1 -
                0.5711994254518804 * eta)
        g = np.exp(- rs / 10) * (1.5342440399687922 -
                0.026315672307260052 * sqrt1meta +
                0.43654715611170275 * (1 - eta)
                ) + (1 - np.exp(- rs / 10)) * (1.611074202872881 +
                0.3618818163840141 * sqrt1meta -
                0.7188721832676245 * (1 - eta) +
                0.24591625072518983 * sqrt1meta**3)
        result = a * hyp2f1(1.5, 2, g, - np.power(
            gamma(g) * np.sqrt(np.pi) * a / (gamma(g - 3 / 2) * b), 2 / 3) *
            rs)

    else:
        if rho < 10**(-100):
            return 0
        rs = 1 / (2 * rho)
        sqrt1meta = 0
        a = (- 34.88113088349792 + 43.869851384127514) / (1 -
                0.6721307918005824)
        b = (185.08839303941474 - 174.90107312179194) / (1 -
                0.5711994254518804)
        g = np.exp(- rs / 10) * (1.5342440399687922
                ) + (1 - np.exp(- rs / 10)) * (1.611074202872881)
        result = a * hyp2f1(1.5, 2, g, - np.power(
            gamma(g) * np.sqrt(np.pi) * a / (gamma(g - 3 / 2) * b), 2 / 3) *
            rs)

    return max(- result / 1000, glda(rho, eta))




