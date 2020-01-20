from itertools import product, combinations, product, permutations
import numpy as np
from leglag.utilities import hyperu_lowb


def mp2_dispersion(molecule, fns):
    #   cdef int a, b, r, s
    #   cdef np.double_t correction
    #   cdef np.double_t[:,:,:,:] db_eris

    if not molecule._hf_complete:
        molecule.run_hartree_fock()

    correction = 0

    #   for dom in molecule.domains:
    #       i = dom.num - 1
    #       if dom.functions == 0: continue
    #       db_eris = dom.mo_double_bar_array[i]
    #       for a in range(dom.electrons):
    #           for b in range(a, dom.electrons):
    #               for r in range(dom.electrons, fns[i]):
    #                   for s in range(r, fns[i]):
    #                       correction += (
    #                           db_eris[a, r, b, s]**2 /
    #                           (dom.orbital_energies[a] +
    #                            dom.orbital_energies[b] -
    #                            dom.orbital_energies[r] -
    #                            dom.orbital_energies[s]))

    l = []
    for dom_one, dom_two in combinations(molecule.domains, 2):
        i = dom_one.num - 1
        j = dom_two.num - 1
        if dom_one.functions == 0 or dom_two.functions == 0:
            continue
        db_eris = dom_one.mo_double_bar_array[j]
        for a in range(dom_one.electrons):
            for b in range(dom_two.electrons):
                for r in range(dom_one.electrons + 1, dom_one.functions):
                    for s in range(dom_two.electrons + 1, dom_two.functions):
                        l.append(
                            (
                                dom_one.num,
                                a,
                                r,
                                dom_two.num,
                                b,
                                s,
                                np.double(
                                    db_eris[a, r, b, s] ** 2
                                    / (
                                        dom_one.orbital_energies[a]
                                        + dom_two.orbital_energies[b]
                                        - dom_one.orbital_energies[r]
                                        - dom_two.orbital_energies[s]
                                    )
                                ),
                            )
                        )

    l = []
    for dom_one, dom_two in combinations(molecule.domains, 2):
        i = dom_one.num - 1
        j = dom_two.num - 1
        if dom_one.functions == 0 or dom_two.functions == 0:
            continue
        db_eris = dom_one.mo_double_bar_array[j]
        for a in range(dom_one.electrons):
            for b in range(dom_two.electrons):
                for r in range(dom_one.electrons, dom_one.electrons + 1):
                    for s in range(dom_two.electrons, dom_two.electrons + 1):
                        l.append(
                            (
                                dom_one.num,
                                a,
                                r,
                                dom_two.num,
                                b,
                                s,
                                np.double(
                                    db_eris[a, r, b, s] ** 2
                                    / (
                                        dom_one.orbital_energies[a]
                                        + dom_two.orbital_energies[b]
                                        - dom_one.orbital_energies[r]
                                        - dom_two.orbital_energies[s]
                                    )
                                ),
                            )
                        )

    for dom_one, dom_two in combinations(molecule.domains, 2):
        i = dom_one.num - 1
        j = dom_two.num - 1
        if dom_one.functions == 0 or dom_two.functions == 0:
            continue
        db_eris = dom_one.mo_double_bar_array[j]
        for a in range(dom_one.electrons):
            for b in range(dom_two.electrons):
                r = dom_one.electrons
                s = dom_two.electrons
                correction += db_eris[a, r, b, s] ** 2 / (
                    dom_one.orbital_energies[a]
                    + dom_two.orbital_energies[b]
                    - dom_one.orbital_energies[r]
                    - dom_two.orbital_energies[s]
                )

    return correction


def inf_inf_dispersion(domain_one, domain_two):

    correction = 0
    for e1, e2 in product(range(domain_one.electrons), range(domain_two.electrons)):

        a = domain_one.mo_field_matrix[e1, e1]
        b = domain_two.mo_field_matrix[e2, e2]
        r = abs(domain_one.position - domain_two.position) + a + b

        ee1 = domain_one.orbital_energies[e1] - domain_one.orbital_energies[e1 + 1]
        ee2 = domain_two.orbital_energies[e2] - domain_two.orbital_energies[e2 + 1]

        #       print([domain_one.num, e1, domain_two.num, e2], end='')
        #       print(' llrr', inf_inf_disp_contribution(a, b, r, ee1, ee2))
        #       print(np.sqrt(inf_inf_disp_contribution(a, b, r, ee1, ee2) * (ee1 + ee2)))
        #       print(domain_one.mo_eri_array[domain_two.num - 1][e1, e1 + 1, e2, e2 + 1])
        #       print(
        #               (2 * (a + b)**3 * ((a - b) * (a + b) *
        #                   (a**2 + 28 * a * b + b**2) - 12 * a * b *
        #                   (a**2 + 3 * a * b + b**2) * np.log(a / b)) / (a - b)**7
        #                   if abs(a - b) > 0.1 else 16 / 35) * a * b / r**3
        #               )
        correction += inf_inf_disp_contribution(a, b, r, ee1, ee2)

    return correction


def inf_inf_disp_contribution(a, b, r, ee1, ee2):

    if abs(a - b) < 1e-1:
        c = -0.22222222222222222222 * a ** 3

    else:
        c = (
            2
            * (a - b) ** 7
            / (
                9
                * (
                    (a - b) * (a + b) * (a ** 2 + 28 * a * b + b ** 2)
                    - 12 * a * b * (a ** 2 + 3 * a * b + b ** 2) * np.log(a / b)
                )
            )
            - (a + b) ** 3
        )

    return (0.66666666666666666667 * a * b / (c + r ** 3)) ** 2 / (ee1 + ee2)


# Orbital maxima fit
#   if abs(a - b) < 1e-1:
#       c = 16 / 35

#   else:
#       c = (
#               (2 * (a + b)**3 * (
#                   (a - b) * (a + b) * (a**2 + 28 * a * b + b**2) -
#                   12 * a * b * (a**2 + 3 * a * b + b**2) *
#                   np.log(a / b))
#               ) / (a - b)**7
#           )

#   return (c * a * b / r**3)**2 / (ee1 + ee2)


def inf_fin_dispersion(domain_one, domain_two):

    correction = 0
    for e1, e2 in product(range(domain_one.electrons), range(domain_two.electrons)):

        a = domain_one.mo_field_matrix[e1, e1]
        b = domain_two.halfwidth - abs(domain_two.mo_field_matrix[e2, e2])
        r = (
            abs(
                sum(domain_two.position) / 2
                + domain_two.mo_field_matrix[e2, e2]
                - domain_one.position
            )
            + a
        )

        ee1 = domain_one.orbital_energies[e1] - domain_one.orbital_energies[e1 + 1]
        ee2 = domain_two.orbital_energies[e2] - domain_two.orbital_energies[e2 + 1]

        #       print([domain_one.num, e1, domain_two.num, e2, ee1, ee2], end='')
        if (
            b < 0.5 * domain_two.halfwidth
            and np.sign(domain_two.mo_field_matrix[e2, e2])
            - np.sign(domain_one.position - 0.5 * sum(domain_two.position))
            == 0
        ):
            #           print(' llrr', inf_inf_disp_contribution(a, b, r, ee1, ee2))
            correction += inf_inf_disp_contribution(a, b, r, ee1, ee2)
        else:
            #           print(' llmm', inf_fin_disp_contribution(a, b, r, ee1, ee2))
            correction += inf_fin_disp_contribution(a, b, r, ee1, ee2)

    return correction


def inf_fin_disp_contribution(a, b, r, ee1, ee2):

    c = -((a + b) ** 3) + 352 * a ** 10 / (
        9
        * a
        * (
            176 * a ** 6
            + 1412 * a ** 5 * b
            + 1912 * a ** 4 * b ** 2
            + 1632 * a ** 3 * b ** 3
            + 1062 * a ** 2 * b ** 4
            + 459 * a * b ** 5
            + 81 * b ** 6
        )
        - 27
        * b
        * (
            528 * a ** 6
            + 1936 * a ** 5 * b
            + 2376 * a ** 4 * b ** 2
            + 1944 * a ** 3 * b ** 3
            + 1206 * a ** 2 * b ** 4
            + 486 * a * b ** 5
            + 81 * b ** 6
        )
        * hyperu_lowb(1, 1, 3 * b / a)[1]
    )

    return (0.4364357804719848 * a * b / (c + r ** 3)) ** 2 / (ee1 + ee2)


def fin_fin_dispersion(domain_one, domain_two):

    correction = 0
    for e1, e2 in product(range(domain_one.electrons), range(domain_two.electrons)):

        a = domain_one.halfwidth - abs(domain_one.mo_field_matrix[e1, e1])
        b = domain_two.halfwidth - abs(domain_two.mo_field_matrix[e2, e2])
        r = abs(
            sum(domain_one.position) / 2
            + domain_one.mo_field_matrix[e1, e1]
            - sum(domain_two.position) / 2
            - domain_two.mo_field_matrix[e2, e2]
        )

        ee1 = domain_one.orbital_energies[e1] - domain_one.orbital_energies[e1 + 1]
        ee2 = domain_two.orbital_energies[e2] - domain_two.orbital_energies[e2 + 1]

        #       print([domain_one.num, e1, domain_two.num, e2, ee1, ee2], end='')
        if (
            a < 0.5 * domain_one.halfwidth
            and b < 0.5 * domain_two.halfwidth
            and np.sign(domain_one.mo_field_matrix[e1, e1])
            == np.sign(domain_two.position[0] - domain_one.position[0])
            and np.sign(domain_two.mo_field_matrix[e2, e2])
            == np.sign(domain_one.position[0] - domain_two.position[0])
        ):
            #           print(' llrr', inf_inf_disp_contribution(a, b, r, ee1, ee2))
            correction += inf_inf_disp_contribution(a, b, r, ee1, ee2)

        elif (
            a < 0.5 * domain_one.halfwidth
            and np.sign(domain_one.mo_field_matrix[e1, e1])
            - np.sign(domain_two.position[0] - domain_one.position[0])
            == 0
        ):
            #           print(' llmm', inf_fin_disp_contribution(a, b, r, ee1, ee2))
            correction += inf_fin_disp_contribution(a, b, r, ee1, ee2)

        elif (
            b < 0.5 * domain_two.halfwidth
            and np.sign(domain_two.mo_field_matrix[e2, e2])
            - np.sign(domain_one.position[0] - domain_two.position[0])
            == 0
        ):
            #           print(' mmll', inf_fin_disp_contribution(b, a, r, ee2, ee1))
            correction += inf_fin_disp_contribution(b, a, r, ee2, ee1)

        else:
            #           print(' mmmm', fin_fin_disp_contribution(a, b, r, ee1, ee2))
            correction += fin_fin_disp_contribution(a, b, r, ee1, ee2)

    return correction


def fin_fin_disp_contribution(a, b, r, ee1, ee2):
    c = (
        -6.113092743999619
        + 3.632932055567455 * a
        - 1.927045718585833 * a ** 2
        - 2.328692115070195 * a ** 3
        + 3.632932055567316 * b
        + 2.723227151626167 * a * b
        - 2.640309426117565 * a ** 2 * b
        - 1.927045718585744 * b ** 2
        - 2.640309426117569 * a * b ** 2
        - 2.328692115070200 * b ** 3
    )

    return (1 / (c / (a * b) + 3.8526056512471665 / (a * b) * r ** 3)) ** 2 / (
        ee1 + ee2
    )
