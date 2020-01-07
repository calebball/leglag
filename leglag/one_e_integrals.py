from math import sqrt
from scipy.special import hyperu, gamma, lpmn, lqmn
from numpy import zeros, ones, double
import numpy as np
from leglag.utilities import hyperu_summands

#====== KINETIC MATRIX ========================================================#

def inf_kinetic(domain):
    """Compute the kinetic matrix for a semi-infinite domain.

    Arguments:
        domain  (InfDomain) target domain
    Returns:
        matrix  (np.array(fns, fns))    calculated kinetic matrix
    """

    matrix = np.zeros((domain.functions, domain.functions), dtype=np.double)
    iterator = np.nditer(matrix, flags=['multi_index'], 
        op_flags=['writeonly'])

    while not iterator.finished:
        i = iterator.multi_index[0] + 1
        j = iterator.multi_index[1] + 1
        m = min(i, j)

        iterator[0] = domain.alpha**2 * (
            np.float64(m * (m + 1) * (2 * m + 1)) / 
            np.float64(3.0 * sqrt(i * (i + 1) * j * (j + 1)))
            )
        if i == j: iterator[0] = iterator[0] - (domain.alpha**2 / np.float64(2))

        iterator.iternext()
    return matrix


def fin_kinetic(domain):
    """Compute the kinetic matrix for a finite domain.

    Arguments:
        domain  (FinDomain) target domain
    Returns:
        matrix  (np.array(fns, fns))    calculated kinetic matrix
    """

    hw = domain.halfwidth
    matrix = np.zeros((domain.functions, domain.functions), dtype=np.double)
    iterator = np.nditer(matrix, flags=['multi_index'], 
        op_flags=['writeonly'])

    while not iterator.finished:
        i = min(iterator.multi_index[:2]) + 1
        j = max(iterator.multi_index[:2]) + 1

        if (i + j) % 2 == 0:
            iterator[0] = sqrt(
                np.float64(i * (i+1) * (i+2) * (i+3) * (i+1.5) * (j+1.5)) /
                np.float64(j * (j+1) * (j+2) * (j+3))
                ) * np.float64(i**2 + 3 * i - 1) / np.float64(6 * hw**2)
        else:
            iterator[0] = np.float64(0)

        iterator.iternext()
    return matrix


#====== POTENTIAL MATRIX ======================================================#

def inf_potential(domain, x):
    """Compute the potential matrix for a semi-infinite domain.

    Arguments:
        domain  (InfDomain) target domain
    Returns:
        matrix  (np.array(fns, fns))    calculated potential matrix
    """

    # Rename some useful quantities
    max_m = domain.functions
    alpha = domain.alpha
    A = domain.position
    zeta = 2 * alpha * abs(A - x)

    # Setup storage for the output
    matrix = zeros((max_m, max_m), dtype=double)
    iterator = np.nditer(matrix, flags=['multi_index'], op_flags=['readwrite'])

    if zeta > 1e-14:
        summands = hyperu_summands(max_m, zeta)
        while not iterator.finished:
            m, n = iterator.multi_index
            tmp = ll_coeffs[(m + 1, n + 1)]
            iterator[0] = (2 * alpha * gamma(m + 1) * gamma(n + 1) *
                sqrt((m + 1) * (m + 2) * (n + 1) * (n + 2)) *
                sum(tmp[i - 1] * zeta**(2 * i) * summands[m + n, i - 1]
                        for i in range(1, min(m, n) + 2)))
            iterator.iternext()

    else:
        while not iterator.finished:
            m = max(iterator.multi_index)
            n = min(iterator.multi_index)
            iterator[0] = alpha * sqrt(((n + 1) * (n + 2)) /
                                            ((m + 1) * (m + 2)))
            iterator.iternext()

    return matrix

def fin_potential(domain, x):
    """Compute the potential matrix for a finite domain.

    Arguments:
        domain  (FinDomain) target domain
    Returns:
        matrix  (np.array(fns, fns))    calculated potential matrix
    """

    # Rename some useful quantities
    max_m = domain.functions
    d = x - (min(domain.position) + domain.halfwidth)

    # Setup storage for the output
    matrix = zeros((max_m, max_m), dtype=double)
    iterator = np.nditer(matrix, flags=['multi_index'], op_flags=['readwrite'])

    if abs(1 - abs(d) / domain.halfwidth) < 1e-14:
        while not iterator.finished:
            m, n = iterator.multi_index
            mu = max(m, n) + 1
            nu = min(m, n) + 1

            integral = ((sqrt((mu + 1.5) * (nu + 1.5)) /
                         (2 * domain.halfwidth)) *
                        sqrt((nu * (nu + 1) * (nu + 2) * (nu + 3)) /
                             (mu * (mu + 1) * (mu + 2) * (mu + 3))))

            if d < 0:
                iterator[0] = (-1)**(m + n) * integral
            else:
                iterator[0] = integral

            iterator.iternext()

    else:
        while not iterator.finished:
            m, n = iterator.multi_index
            m += 1
            n += 1

            normalisation = 2.0 * sqrt(
                np.float64((m + 1.5) * (n + 1.5)) /
                np.float64(m * (m + 1) * (m + 2) * (m + 3) * 
                    n * (n + 1) * (n + 2) * (n + 3))
                ) / domain.halfwidth

            if d > 1:
                legendre = (
                    lpmn(2, min(m, n) + 1, d / domain.halfwidth)[0][2][-1] *
                    lqmn(2, max(m, n) + 1, d / domain.halfwidth)[0][2][-1]
                    )
            else:
                legendre = (
                    lpmn(2, min(m, n) + 1, d / domain.halfwidth)[0][2][-1] *
                    (-1)**max(m, n) * lqmn(2, max(m, n) + 1,
                        abs(d / domain.halfwidth))[0][2][-1]
                    )

            if d < 0:
                iterator[0] = - normalisation * legendre
            else:
                iterator[0] = normalisation * legendre

            iterator.iternext()
    return matrix


def inf_nuclear_attraction(domain):
    """Compute the nuclear attraction matrix for a semi-infinite domain.

    Arguments:
        domain  (InfDomain) target domain
    Returns:
        matrix  (np.array(fns, fns))    calculated nuclear attraction matrix
    """

    return sum(- nucleus[1] * inf_potential(domain, nucleus[0])
               for nucleus in domain.molecule.nuclei)

def fin_nuclear_attraction(domain):
    """Compute the nuclear attraction matrix for a finite domain.

    Arguments:
        domain  (FinDomain) target domain
    Returns:
        matrix  (np.array(fns, fns))    calculated nuclear attraction matrix
    """

    return sum(- nucleus[1] * fin_potential(domain, nucleus[0])
               for nucleus in domain.molecule.nuclei)


class LLCoeffs(dict):
    """Self-constructing dictionary of Laguerre product expansion
    coefficients."""
    def __missing__(self, key):
        (m, n) = key
        tmp = ones(min(m, n), double)
        for i in range(min(m, n) - 1, 0, -1):
            tmp[i - 1] = tmp[i] * (m + n - 2 * i) * (m + n - 2 * i - 1) / (
                (m - i) * (n - i))
        tmp = tmp / 2
        for i in range(2, min(m, n) + 1):
            tmp[i - 1:] = tmp[i - 1:] / ((i - 1) * (i + 1))

        self[(m, n)] = tmp
        self[(n, m)] = tmp
        return tmp

ll_coeffs = LLCoeffs()


#====== FIELD MATRIX ==========================================================#

def inf_field(domain):
    """Compute the field response matrix for a semi-infinite domain.

    Arguments:
        domain  (InfDomain) target domain
    Returns:
        matrix  (np.array(fns, fns))    calculated field matrix
    """

    a = domain.alpha
    matrix = np.zeros((domain.functions, domain.functions), dtype=np.double)
    iterator = np.nditer(matrix, flags=['multi_index'], op_flags=['writeonly'])

    while not iterator.finished:
        i = iterator.multi_index[0] + 1
        j = iterator.multi_index[1] + 1
        m = min(i, j)

        if i == j:
            iterator[0] = np.double(2 * m + 1) / np.double(2 * a)
        elif i == j - 1 or i == j + 1:
            iterator[0] = - np.double(np.sqrt(m * (m + 2))) / np.double(2 * a)

        iterator.iternext()
    return matrix


def fin_field(domain):
    """Compute the field response matrix for a finite domain.

    Arguments:
        domain  (FinDomain) target domain
    Returns:
        matrix  (np.array(fns, fns))    calculated field matrix
    """

    hw = domain.halfwidth
    matrix = np.zeros((domain.functions, domain.functions), dtype=np.double)
    iterator = np.nditer(matrix, flags=['multi_index'], op_flags=['writeonly'])

    while not iterator.finished:
        i = min(iterator.multi_index[:2]) + 1
        j = max(iterator.multi_index[:2]) + 1

        if i == j - 1:
            iterator[0] = 0.5 * hw * sqrt(
                np.double(i * (i + 4)) / np.double((i + 1.5) * (i + 2.5)))

        iterator.iternext()
    return matrix




