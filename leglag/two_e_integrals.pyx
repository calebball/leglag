from itertools import chain, islice, combinations_with_replacement, product
from functools import reduce
from pkg_resources import resource_filename
import os.path
import sys
import operator
from pkgutil import get_loader
from numpy import sqrt, zeros, ones, fromiter, double
from scipy.special import lpmn, lqmn, gamma, beta
from leglag.one_e_integrals import inf_potential, fin_potential
from leglag.utilities import fejer, hyperu_summands, hyperu_llrr_summands, legendre_p
import numpy as np
cimport numpy as np

thresh = 1e-10

#====== INF-INF INTEGRALS =====================================================#

def true_inf_inf(domain_one, domain_two, start=0):
    """Compute the two electron integral tensor between two semi-infinite
    domains.

    Uses an analytic expression for the integrals.

    Arguments:
        domain_one  (InfDomain)
        domain_two  (InfDomain)
    Returns:
        integrals   (np.array(fns_one, fns_one, fns_two, fns_two))
    """

    cdef int m, n, l, s
    cdef int i, j, k

    cdef np.double_t[:] zeta_powers
    cdef np.double_t[:, :] coeff_arr, summands

    # Rename/precompute useful quantities
    a1 = domain_one.alpha
    a2 = domain_two.alpha
    a = a1  # WARNING! We need to assume alpha is equal in both domains
            #          If we do not the integral is likely a two-variable
            #          hypergeometric
    d = abs(domain_one.position - domain_two.position)

    m_one = domain_one.functions
    m_two = domain_two.functions

    # Setup bulk storage
    cdef np.ndarray[np.uint_t, ndim=4] mask = ones(
            (m_one, m_one, m_two, m_two), dtype=np.uint)
    cdef np.ndarray[np.double_t, ndim=4] integrals = zeros(
            (m_one, m_one, m_two, m_two), dtype=np.double)

    if d > 0:

        # Use an integral bound to exclude integrals
        d1_v_large = abs(inf_potential(domain_one, domain_two.position))
        d2_v_large = abs(inf_potential(domain_two, domain_one.position))

        for m, n in combinations_with_replacement(
                range(domain_one.functions), 2):
            for l, s in combinations_with_replacement(
                    range(domain_two.functions), 2):
                bound = min(d1_v_large[m, n] * domain_two.s_star_p[l, s],
                            d2_v_large[l, s] * domain_one.s_star_p[m, n])
                if bound < thresh:
                    mask[m, n, l, s] = 0
                    mask[m, n, s, l] = 0
                    mask[n, m, l, s] = 0
                    mask[n, m, s, l] = 0

        # Precompute the summands
        summands = hyperu_llrr_summands(2 * (m_one + m_two), 2 * a * d)

        zeta_powers = fromiter(((2 * a * d)**(2 * k + 1)
                for k in range(2, m_one + m_two + 1)), double)
        for m, n in product(range(m_one), repeat=2):
            for l, s in product(range(m_two), repeat=2):
                if mask[m, n, l, s] == 0: continue
                coeff_arr = llrr_coeffs[(m + 1, n + 1, l + 1, s + 1)]

                integrals[m, n, l, s] = sum(zeta_powers[i + j - 2] *
                                            summands[m + n + l + s, i + j - 2] *
                                            coeff_arr[i - 1, j - 1]
                                            for i in range(1, min(m, n) + 2)
                                            for j in range(1, min(l, s) + 2))

                integrals[m, n, l, s] *= (2 * a * gamma(m + 3) * gamma(n + 3) *
                        gamma(l + 3) * gamma(s + 3) / (sqrt(m + 1) *
                        sqrt(m + 2) * sqrt(n + 1) * sqrt(n + 2) * sqrt(l + 1) *
                        sqrt(l + 2) * sqrt(s + 1) * sqrt(s + 2)))

    else:
        for m, n in product(range(start, m_one), repeat=2):
            for l, s in product(range(start, m_two), repeat=2):
                coeff_arr = llrr_coeffs[(m + 1, n + 1, l + 1, s + 1)]
                integrals[m, n, l, s] = sum(gamma(2 * i + 2 * j + 1) *
                            coeff_arr[i - 1, j - 1]
                            for i in range(1, min(m, n) + 2)
                            for j in range(1, min(l, s) + 2))

                integrals[m, n, l, s] *= (2 * a * gamma(m + 3) *
                        gamma(n + 3) * gamma(l + 3) *
                        gamma(s + 3) / (sqrt(m + 1) * sqrt(m + 2) *
                        sqrt(n + 1) * sqrt(n + 2) * sqrt(l + 1) *
                        sqrt(l + 2) * sqrt(s + 1) * sqrt(s + 2) *
                        gamma(m + n + l + s + 6)))

    return integrals

def quasi_inf_inf(domain_one):
    """Compute the two electron integral tensor within one semi-infinite
    domain.

    Retrieves precomputed integrals and scales them to the given domain

    Arguments:
        domain_one  (InfDomain)
    Returns:
        integrals   (np.array(fns_one, fns_one, fns_one, fns_one))
    """

    # Setup bulk storage
    integrals = zeros((domain_one.functions, domain_one.functions, 
                      domain_one.functions, domain_one.functions), 
                     dtype=np.double)

    with open(resource_filename('leglag',
            'integral_data/LLLLquasi_integrals.dat')) as int_file:
        for line in int_file:
            (m, n, l, s, value) = line.split()
            m = int(m) - 1
            n = int(n) - 1
            l = int(l) - 1
            s = int(s) - 1
            if max(m, n, l, s) + 1 > domain_one.functions: break
            value = domain_one.alpha * np.float64(value)

            integrals[m, n, l, s] = value
            integrals[m, n, s, l] = value
            integrals[n, m, l, s] = value
            integrals[n, m, s, l] = value

            integrals[l, s, m, n] = value
            integrals[l, s, n, m] = value
            integrals[s, l, m, n] = value
            integrals[s, l, n, m] = value

    return integrals


#====== INF-FIN INTEGRALS =====================================================#

def true_inf_fin(domain_one, domain_two, quad_check=True, quad_start=1):
    """Compute the two electron integral tensor between one semi-infinite
    domain and one finite domain.

    Utilises a customised Clenshaw-Curtis quadrature routine to evaluate the
    integrals.

    Arguments:
        domain_one  (InfDomain)
        domain_two  (FinDomain)
    Returns:
        integrals   (np.array(fns_one, fns_one, fns_two, fns_two))
    """

    # Declare some variables
    cdef int m, n, l, s
    cdef int i
    cdef double largest_integral = thresh
    cdef double curr_approx

    # Rename/precompute useful quantities
    alpha = domain_one.alpha
    hw = domain_two.halfwidth
    d = abs(domain_one.position - min(domain_two.position) - hw)

    potential_m = domain_one.functions + 1
    density_m   = domain_two.functions + 1

    sign = -1 if domain_one.side else 1
    fin_norm = np.fromiter((sqrt((m + 2.5) /
        (hw * (m + 1) * (m + 2) * (m + 3) * (m + 4)))
        for m in range(density_m)), np.double)

    # Setup some temporary storage
    potential2 = [[None] * potential_m for x in range(potential_m)]

    # Setup bulk storage
    cdef np.ndarray[np.double_t, ndim=4] integrals = zeros(
                     (domain_one.functions, domain_one.functions,
                      domain_two.functions, domain_two.functions),
                      dtype=np.double)
    cdef np.ndarray[np.uint_t, ndim=4] mask = ones(
                (domain_one.functions, domain_one.functions,
                domain_two.functions, domain_two.functions),
                dtype=np.uint)
    cdef np.ndarray[np.double_t, ndim=4] last_approx = zeros(
                        (domain_one.functions, domain_one.functions,
                         domain_two.functions, domain_two.functions),
                        dtype=np.double)

    # Use an integral bound to exclude integrals
    if domain_one.side:
        d1_v_large = abs(inf_potential(domain_one, max(domain_two.position)))
        d1_v_small = abs(inf_potential(domain_one, min(domain_two.position)))
    else:
        d1_v_large = abs(inf_potential(domain_one, min(domain_two.position)))
        d1_v_small = abs(inf_potential(domain_one, max(domain_two.position)))
    d2_v_large = abs(fin_potential(domain_two, domain_one.position))

    for m, n in combinations_with_replacement(
            range(domain_one.functions), 2):
        for l, s in combinations_with_replacement(
                range(domain_two.functions), 2):
            bound = min(d1_v_large[m, n] * domain_two.s_star_p[l, s] +
                        d1_v_small[m, n] * domain_two.s_star_n[l, s],
                        d2_v_large[l, s] * domain_one.s_star_p[m, n])
            if bound < thresh:
                mask[m, n, l, s] = 0
                mask[m, n, s, l] = 0
                mask[n, m, l, s] = 0
                mask[n, m, s, l] = 0

    # Compute the necessary functions at the first set of
    # quadrature points
    quadn = quad_start
    while len(fejer.abscissae) <= quadn: fejer.extend()
    summands  = [hyperu_summands(potential_m, 2 * alpha * (hw * x + d))
                 for x in fejer.abscissae[quadn]]
    density   = [legendre_p(density_m, x)
                 for x in fejer.abscissae[quadn]]

    # Create transpose arrays to feed into fejer.error
    density_t = list(map(np.asarray, zip(*density)))

    # Compute the potential
    tmp1 = (2 * alpha * (hw * fejer.abscissae[quadn] + d))**2
    for m in range(potential_m):
        for n in range(m + 1):
            tmp2 = llpp_coeffs[(m + 1, n + 1)]
            potential2[m][n] = fromiter((sum(tmp1[x]**i *
                summands[x][m + n, i - 1] * tmp2[i - 1]
                for i in range(1, min(m, n) + 2))
                for x in range(len(fejer.abscissae[quadn]))), double)
            potential2[n][m] = potential2[m][n]

    # Check if we're allowing the quadrature to be refined
    if quad_check:

        # Estimate the current error of each integral
        for m, n in product(range(domain_one.functions), repeat=2):
            for l, s in product(range(domain_two.functions), repeat=2):
                if not mask[m, n, l, s]: continue
                last_approx[m, n, l, s] = np.einsum('i,i,i,i',
                        fejer.weights[quadn - 1], potential2[m][n][1::2],
                        density_t[l][1::2], density_t[s][1::2])
                curr_approx = np.einsum('i,i,i,i', fejer.weights[quadn],
                        potential2[m][n], density_t[l], density_t[s])
                if abs(curr_approx) > largest_integral:
                    largest_integral = abs(curr_approx)
                integrals[m, n, l, s] = abs(curr_approx -
                        last_approx[m, n, l, s]) / largest_integral
                last_approx[m, n, l, s] = curr_approx

        # Add points to the quadrature until we acheive the
        # desired accuracy
        while integrals.max() > thresh:
            quadn += 1
            while len(fejer.abscissae) <= quadn: fejer.extend()

            # Evaluate the necessary functions at only the new
            # quadrature points and interleave them with the
            # previously calculated values
            it = islice(fejer.abscissae[quadn], 0, 2**(quadn + 1) - 2, 2)
            summands = list(chain.from_iterable(zip(
                [hyperu_summands(potential_m, 2 * alpha * (hw * x + d)) 
                for x in it], 
                summands)))
            summands.append(hyperu_summands(potential_m, 
                2 * alpha * (hw * fejer.abscissae[quadn][-1] + d)))

            it = islice(fejer.abscissae[quadn], 0, 2**(quadn + 1) - 2, 2)
            density = list(chain.from_iterable(zip(
                [legendre_p(density_m, x) for x in it], density)))
            density.append(legendre_p(density_m, fejer.abscissae[quadn][-1]))

            # Rebuild the transposed versions
            density_t = list(map(np.asarray, zip(*density)))

            # Compute the potential
            tmp1 = [fromiter(((2 * alpha * (hw * x + d))**(2 * i)
                    for i in range(1, potential_m + 1)), double)
                    for x in fejer.abscissae[quadn]]
            for m in range(potential_m):
                for n in range(m + 1):
                    tmp2 = [x[:min(m, n) + 1] *
                            llpp_coeffs[(m + 1, n + 1)][:min(m, n) + 1]
                            for x in tmp1]
                    potential2[m][n] = fromiter(
                        (np.dot(summands[x][m + n,:min(m, n) + 1], tmp2[x])
                        for x in range(len(fejer.abscissae[quadn]))), double)
                    potential2[n][m] = potential2[m][n]

            # Estimate the current error of each integral
            for m, n in product(range(domain_one.functions), repeat=2):
                for l, s in product(range(domain_two.functions), repeat=2):
                    if not mask[m, n, l, s]: continue
                    # We need this to fly, so I'm using np.einsum
                    # for some compiled code magic
                    curr_approx = np.einsum('i,i,i,i', fejer.weights[quadn],
                                    potential2[m][n], density_t[l], density_t[s])
                    if abs(curr_approx) > largest_integral:
                        largest_integral = abs(curr_approx)
                    integrals[m, n, l, s] = abs(curr_approx -
                        last_approx[m, n, l, s]) / largest_integral
                    last_approx[m, n, l, s] = curr_approx

        # Finally add the factored out normalisation terms and
        # construct the integral array
        for m, n in product(range(domain_one.functions), repeat=2):
            for l, s in product(range(domain_two.functions), repeat=2):
                if not mask[m, n, l, s]: continue
                integrals[m, n, l, s] = (sign**(l + s) * hw *
                        fin_norm[l] * fin_norm[s] *
                        2 * alpha * gamma(m + 1) * gamma(n + 1) *
                        sqrt((m + 1) * (m + 2) * (n + 1) * (n + 2)) *
                        last_approx[m, n, l, s])

    else:
        # We're not refining, so we just use the starting
        # quadrature
        for m, n in product(range(domain_one.functions), repeat=2):
            for l, s in product(range(domain_two.functions), repeat=2):
                if not mask[m, n, l, s]: continue
                integrals[m, n, l, s] = (sign**(l + s) * hw *
                        fin_norm[l] * fin_norm[s] *
                        2 * alpha * gamma(m + 1) * gamma(n + 1) *
                        sqrt((m + 1) * (m + 2) * (n + 1) * (n + 2)) *
                        np.einsum('i,i,i,i', fejer.weights[quadn],
                            potential2[m][n], density_t[l], density_t[s]))

    return integrals, quadn


#====== FIN-FIN INTEGRALS =====================================================#

def true_fin_fin(domain_one, domain_two, quad_check=True, quad_start=1):
    """Compute the two electron integral tensor between two finite
    domains.

    Uses a Clenshaw-Curtis quadrature routine to evaluate the integrals.

    Arguments:
        domain_one  (FinDomain)
        domain_two  (FinDomain)
    Returns:
        integrals   (np.array(fns_one, fns_one, fns_two, fns_two))
    """

    # Declare some variables
    cdef int m, n, l, s
    cdef int i
    cdef double largest_integral = thresh
    cdef double curr_approx

    # Rename/precompute some useful quantities
    hw_one = domain_one.halfwidth
    hw_two = domain_two.halfwidth
    c_one = sum(domain_one.position) / 2
    c_two = sum(domain_two.position) / 2

    potential_m = domain_one.functions + 1
    density_m   = domain_two.functions + 1

    sign = 1 if c_two > c_one else -1
    norm_one = np.fromiter((sqrt((m + 2.5) /
        (hw_one * (m + 1) * (m + 2) * (m + 3) * (m + 4)))
        for m in range(potential_m)), np.double)
    norm_two = np.fromiter((sqrt((m + 2.5) /
        (hw_two * (m + 1) * (m + 2) * (m + 3) * (m + 4)))
        for m in range(density_m)), np.double)

    # Setup some bulk storage
    cdef np.ndarray[np.uint_t, ndim=4] mask = ones(
                (domain_one.functions, domain_one.functions,
                 domain_two.functions, domain_two.functions),
                dtype=np.uint)
    cdef np.ndarray[np.double_t, ndim=4] integrals = zeros(
                (domain_one.functions, domain_one.functions,
                 domain_two.functions, domain_two.functions),
                dtype=np.double)
    cdef np.ndarray[np.double_t, ndim=4] last_approx = zeros(
                (domain_one.functions, domain_one.functions,
                 domain_two.functions, domain_two.functions),
                dtype=np.double)

    # Use an integral bound to exclude integrals
    if c_two > c_one:
        d1_v_large = abs(fin_potential(domain_one, min(domain_two.position)))
        d1_v_small = abs(fin_potential(domain_one, max(domain_two.position)))
        d2_v_large = abs(fin_potential(domain_two, max(domain_one.position)))
        d2_v_small = abs(fin_potential(domain_two, min(domain_one.position)))
    else:
        d1_v_large = abs(fin_potential(domain_one, max(domain_two.position)))
        d1_v_small = abs(fin_potential(domain_one, min(domain_two.position)))
        d2_v_large = abs(fin_potential(domain_two, min(domain_one.position)))
        d2_v_small = abs(fin_potential(domain_two, max(domain_one.position)))

    for m, n in combinations_with_replacement(
            range(domain_one.functions), 2):
        for l, s in combinations_with_replacement(
                range(domain_two.functions), 2):
            bound = min(d1_v_large[m, n] * domain_two.s_star_p[l, s] +
                        d1_v_small[m, n] * domain_two.s_star_n[l, s],
                        d2_v_large[l, s] * domain_one.s_star_p[m, n] +
                        d2_v_small[l, s] * domain_one.s_star_n[m, n])
            if bound < thresh:
                mask[m, n, l, s] = 0
                mask[m, n, s, l] = 0
                mask[n, m, l, s] = 0
                mask[n, m, s, l] = 0

    # Compute the necessary functions at the first set of 
    # quadrature points
    quadn = quad_start
    while len(fejer.abscissae) <= quadn: fejer.extend()
    potential_q = [lqmn(2, potential_m,
                        (hw_two * x + abs(c_two - c_one)) / hw_one)[0][2][2:]
                   for x in fejer.abscissae[quadn]]
    potential_p = [lpmn(2, potential_m,
                        (hw_two * x + abs(c_two - c_one)) / hw_one)[0][2][2:]
                   for x in fejer.abscissae[quadn]]
    density_p   = [lpmn(2, density_m, x)[0][2][2:] 
                   for x in fejer.abscissae[quadn]]

    # Create transpose arrays to feed into fejer.error
    potential_q_t = list(map(np.asarray, zip(*potential_q)))
    potential_p_t = list(map(np.asarray, zip(*potential_p)))
    density_p_t = list(map(np.asarray, zip(*density_p)))

    # Check if we're allowing the quadrature to be refined
    if quad_check:

        # Estimate the current error of each integral
        for m, n in product(range(domain_one.functions), repeat=2):
            for l, s in product(range(domain_two.functions), repeat=2):
                if not mask[m, n, l, s]: continue
                last_approx[m, n, l, s] = np.einsum('i,i,i,i,i',
                        fejer.weights[quadn - 1],
                        potential_q_t[max(m, n)][1::2],
                        potential_p_t[min(m, n)][1::2],
                        density_p_t[l][1::2], density_p_t[s][1::2])
                curr_approx = np.einsum('i,i,i,i,i', fejer.weights[quadn],
                        potential_q_t[max(m, n)], potential_p_t[min(m, n)],
                        density_p_t[l], density_p_t[s])
                if abs(curr_approx) > largest_integral:
                    largest_integral = abs(curr_approx)
                integrals[m, n, l, s] = abs(curr_approx -
                    last_approx[m, n, l, s]) / largest_integral
                last_approx[m, n, l, s] = curr_approx

        # Add points to the quadrature until we acheive the
        # desired accuracy
        while integrals.max() > thresh:
            quadn += 1
            while len(fejer.abscissae) <= quadn: fejer.extend()

            # Evaluate the necessary functions at only the new
            # quadrature points and interleave them with the 
            # previously calculated values
            it = islice(fejer.abscissae[quadn], 0, 2**(quadn + 1) - 2, 2)
            potential_q = list(chain.from_iterable(zip([lqmn(2, potential_m,
                (hw_two * x + abs(c_two - c_one)) / hw_one)[0][2][2:]
                for x in it],
                potential_q)))
            potential_q.append(lqmn(2, potential_m,
                (hw_two * fejer.abscissae[quadn][-1] + abs(c_two - c_one)) /
                hw_one)[0][2][2:])

            it = islice(fejer.abscissae[quadn], 0, 2**(quadn + 1) - 2, 2)
            potential_p = list(chain.from_iterable(zip([lpmn(2, potential_m,
                (hw_two * x + abs(c_two - c_one)) / hw_one)[0][2][2:]
                for x in it],
                potential_p)))
            potential_p.append(lpmn(2, potential_m,
                (hw_two * fejer.abscissae[quadn][-1] + abs(c_two - c_one)) /
                hw_one)[0][2][2:])

            it = islice(fejer.abscissae[quadn], 0, 2**(quadn + 1) - 2, 2)
            density_p = list(chain.from_iterable(zip(
                [lpmn(2, density_m, x)[0][2][2:] for x in it], density_p)))
            density_p.append(lpmn(2, density_m, fejer.abscissae[quadn][-1])[0][2][2:])

            # Rebuild the transposed versions
            potential_q_t = list(map(np.asarray, zip(*potential_q)))
            potential_p_t = list(map(np.asarray, zip(*potential_p)))
            density_p_t = list(map(np.asarray, zip(*density_p)))

            # Estimate the current error of each integral
            for m, n in product(range(domain_one.functions), repeat=2):
                for l, s in product(range(domain_two.functions), repeat=2):
                    if not mask[m, n, l, s]: continue
                    curr_approx = np.einsum('i,i,i,i,i', fejer.weights[quadn],
                            potential_q_t[max(m, n)], potential_p_t[min(m, n)],
                            density_p_t[l], density_p_t[s])
                    if abs(curr_approx) > largest_integral:
                        largest_integral = abs(curr_approx)
                    integrals[m, n, l, s] = abs(curr_approx -
                        last_approx[m, n, l, s]) / largest_integral
                    last_approx[m, n, l, s] = curr_approx

        # Finally add the factored out normalisation terms and
        # construct the integral array
        for m, n in product(range(domain_one.functions), repeat=2):
            for l, s in product(range(domain_two.functions), repeat=2):
                if not mask[m, n, l, s]: continue
                integrals[m, n, l, s] = (sign**(m + n + l + s) * 2 * hw_two *
                    norm_one[m] * norm_one[n] * norm_two[l] * norm_two[s] *
                    last_approx[m, n, l, s])

    else:
        # We're not refining, so we just use the starting
        # quadrature
        for m, n in product(range(domain_one.functions), repeat=2):
            for l, s in product(range(domain_two.functions), repeat=2):
                if not mask[m, n, l, s]: continue
                integrals[m, n, l, s] = (sign**(m + n + l + s) * 2 * hw_two *
                    norm_one[m] * norm_one[n] * norm_two[l] * norm_two[s] *
                    np.einsum('i,i,i,i,i', fejer.weights[quadn],
                        potential_q_t[max(m, n)], potential_p_t[min(m, n)],
                        density_p_t[l], density_p_t[s]))

    return integrals, quadn


def quasi_fin_fin(domain):
    """Compute the two electron integral tensor within one finite domain.

    Uses a recurrence of the Legendre product to evaluate the integrals
    analytically.

    Arguments:
        domain_one  (FinDomain)
    Returns:
        integrals   (np.array(fns_one, fns_one, fns_two, fns_two))
    """

    integrals = np.zeros((2 * domain.functions - 1, domain.functions, 
                          2 * domain.functions - 1, domain.functions), 
                         dtype=np.double)

    log2 = np.log(2)

    # Begin by filling the starting values
    for i in range(2 * domain.functions - 1):
        for j in range(2 * domain.functions - 1):
            if (i + j) % 2 == 0:
                m = i + 1
                l = j + 1

                normalisation = (-1)**((m - l + 2) / 2) * (15.0 / 2.0) * double(sqrt(
                    m * (m + 1) * (m + 2) * (m + 3) * (m + 1.5) *
                    l * (l + 1) * (l + 2) * (l + 3) * (l + 1.5)
                    )) / double((m + l - 1) * (m + l + 1) * (m + l + 3) * (m + l + 5) *
                         (m + l + 7))

                term1 = (12.0 / (m + l - 1.0) + 12.0 / (m + l + 1.0) +
                         12.0 / (m + l + 3.0) + 12.0 / (m + l + 5.0) +
                         12.0 / (m + l + 7.0) - 25 - 12 * np.euler_gamma -
                         12 * log2 + 24 * sum(1.0 / (2 * n + 1.0)
                                              for n in range((m + l) // 2 - 1))
                        ) / (gamma((m - l + 6.0) / 2.0) *
                             gamma((l - m + 6.0) / 2.0))

                if m - l == 0:
                    term2 = 4.5
                elif abs(m - l) == 2:
                    term2 = 17.0 / 6.0
                elif abs(m - l) == 4:
                    term2 = 25.0 / 48.0
                else:
                    tmp = abs((m - l) // 2)
                    term2 = ((-1)**tmp * 6.0 / double((tmp - 2) * (tmp - 1) * tmp *
                             (tmp + 1) * (tmp + 2)))

                if m - l == 0:
                    term3 = 3 * (np.euler_gamma + log2)
                elif abs(m - l) == 2:
                    term3 = 2 * (np.euler_gamma + log2)
                elif abs(m - l) == 4:
                    term3 = 0.5 * (np.euler_gamma + log2)
                else:
                    term3 = 0

                integrals[i,0,j,0] = normalisation * (term1 + term2 + term3)

    # Now use a recurrence relation to fill the rest of the array
    for i in range(2 * domain.functions - 1):
        pp_recurrence(integrals[i,0,:,:], domain.functions)
    for i in range(domain.functions):
        for j in range(domain.functions):
            pp_recurrence(integrals[:,:,i,j], domain.functions)

    integrals = integrals.compress([True if x < domain.functions else False 
                                    for x in range(2 * (domain.functions) - 1)],
                                   axis=0)
    integrals = integrals.compress([True if x < domain.functions else False 
                                    for x in range(2 * (domain.functions) - 1)],
                                   axis=2)

    integrals = integrals / domain.halfwidth

    return integrals


#====== RECURRENCE FUNCTIONS ==================================================#

def pp_recurrence(matrix, M):
    """Perform a recurrence of a Legendre product from starting points stored
    in a given matrix."""

    for j in range(M - 1):
        for i in range(2 * M - 2 - j):
            mu = i + 1
            nu = j + 1

            term1 = (sqrt(mu * (mu + 4) / ((2 * mu + 3) * (2 * mu + 5))) * 
                     matrix[i + 1, j])

            if i > 0:
                term2 = sqrt((mu - 1) * (mu + 3) / ((2 * mu + 1) * 
                             (2 * mu + 3))) * matrix[i - 1, j]
            else:
                term2 = 0

            if j > 0:
                term3 = sqrt((nu - 1) * (nu + 3) / ((2 * nu + 1) * 
                             (2 * nu + 3))) * matrix[i, j - 1]
            else:
                term3 = 0

            matrix[i, j + 1] = sqrt((2 * nu + 3) * (2 * nu + 5) / 
                                    (nu**2 + 4 * nu)) * (term1 + term2 - term3)

    return matrix

#====== SUMMATION COEFFICIENTS ================================================#

class LLPPCoeffs(dict):
    """Self-constructing dictionary of coefficients for an expansion of a
    quad-product of two Laguerre and two Legendre polynomials."""
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

class LLRRCoeffs(dict):
    """Self-constructing dictionary of coefficients for an expansion of a
    quad-product of Laguerre polynomials."""
    def __missing__(self, key):
        (m, n, l, s) = key
        tmp = zeros((min(m, n), min(l, s)), double)
        tmp[-1, -1] = reduce(operator.mul,
                ((a + abs(l - s)) / a for a in range(1, abs(m - n) + 1)), 1)

        for i in range(min(m, n) - 1, 0, -1):
            tmp[i - 1, -1] = (tmp[i, -1] *
                    (m + n + l + s - 2 * (i + min(l, s))) *
                    (m + n + l + s - 2 * (i + min(l, s)) - 1) /
                    ((m - i) * (n - i)))

        for i in range(1, min(m, n) + 1):
            for j in range(min(l, s) - 1, 0, -1):
                tmp[i - 1, j - 1] = (tmp[i - 1, j] *
                        (m + n + l + s - 2 * (i + j)) *
                        (m + n + l + s - 2 * (i + j) - 1) /
                        ((l - j) * (s - j)))

        tmp = tmp / 4
        for i in range(2, min(m, n) + 1):
            tmp[i - 1:, :] = tmp[i - 1:, :] / ((i - 1) * (i + 1))
        for j in range(2, min(l, s) + 1):
            tmp[:, j - 1:] = tmp[:, j - 1:] / ((j - 1) * (j + 1))

        self[(m, n, l, s)] = tmp
        self[(m, n, s, l)] = tmp
        self[(n, m, l, s)] = tmp
        self[(n, m, s, l)] = tmp
        self[(l, s, m, n)] = tmp.T
        self[(l, s, n, m)] = tmp.T
        self[(s, l, m, n)] = tmp.T
        self[(s, l, n, m)] = tmp.T
        return tmp

llpp_coeffs = LLPPCoeffs()
llrr_coeffs = LLRRCoeffs()






