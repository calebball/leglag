from __future__ import print_function
from itertools import combinations, product, permutations
import numpy as np
cimport numpy as np

def mp2_correction(molecule, fns):
    """Computes the MP2 correlation estimate for a 1D molecule.

    Arguments:
        molecule    (OneDMolecule)  target molecule
        fns         (int)           number of basis functions to use in estimate
    """

    cdef int a, b, r, s
    cdef np.double_t correction
    cdef np.double_t[:,:,:,:] db_eris

    if not molecule._hf_complete: molecule.run_hartree_fock()

#   fns = {d:n for d, n in zip(molecule.domains, molecule._hf_fns)}

#   print()
#   print([(d, d.functions) for d in molecule.domains])
#   if molecule._hf_fns:
#       try:
#           fns = {d:lim for d, lim in zip(molecule.domains, molecule._hf_fns)}
#       except TypeError:
#           fns = {d:molecule._hf_fns for d in molecule.domains}
#   else:
#       fns = {d:d.functions for d in molecule.domains}

    correction = 0
    for dom in molecule.domains:
        i = dom.num - 1
        if dom.functions == 0: continue
        db_eris = dom.mo_double_bar_array[i]
        for a in range(dom.electrons):
            for b in range(a, dom.electrons):
                for r in range(dom.electrons, fns[i]):
                    for s in range(r, fns[i]):
                        correction += (
                            db_eris[a, r, b, s]**2 /
                            (dom.orbital_energies[a] +
                             dom.orbital_energies[b] -
                             dom.orbital_energies[r] -
                             dom.orbital_energies[s]))

    for dom_one, dom_two in combinations(molecule.domains, 2):
        i = dom_one.num - 1
        j = dom_two.num - 1
        if dom_one.functions == 0 or dom_two.functions == 0: continue
        db_eris = dom_one.mo_double_bar_array[j]
        for a in range(dom_one.electrons):
            for b in range(dom_two.electrons):
                for r in range(dom_one.electrons, fns[i]):
                    for s in range(dom_two.electrons, fns[j]):
                        correction += (
                            db_eris[a, r, b, s]**2 /
                            (dom_one.orbital_energies[a] +
                             dom_two.orbital_energies[b] -
                             dom_one.orbital_energies[r] -
                             dom_two.orbital_energies[s]))

    return correction

def mp2_domain_correction(domain):
    """Computes the MP2 correlation estimate within a 1D electron domain.

    Arguments:
        domain  (OneDDomain)    target domain
    """

    cdef int a, b, r, s
    cdef np.double_t correction
    cdef np.double_t[:,:,:,:] db_eris

    if not domain.molecule._hf_complete:
        domain.molecule.run_hartree_fock()
    if domain.electrons == 0 or domain.functions == 0:
        return 0

    correction = 0

    i = domain.molecule.domains.index(domain)
    db_eris = domain.mo_double_bar_array[i]
    for a in range(domain.electrons):
        for b in range(a, domain.electrons):
            for r in range(domain.electrons, domain.functions):
                for s in range(r, domain.functions):
                    correction += (
                        db_eris[a, r, b, s]**2 /
                        (domain.orbital_energies[a] +
                         domain.orbital_energies[b] -
                         domain.orbital_energies[r] -
                         domain.orbital_energies[s]))

    return correction


def mp3_correction(molecule, fns):
    """Computes the MP3 correlation estimate for a 1D molecule.

    Arguments:
        molecule    (OneDMolecule)  target molecule
        fns         (int)           number of basis functions to use in estimate
    """

    cdef int a, b, c, d, r, s, t, u
    cdef np.double_t suma = 0
    cdef np.double_t sumb = 0
    cdef np.double_t sumc = 0
    cdef np.double_t[:,:,:,:] db_eris, db_eris_b, db_eris_c

    if not molecule._hf_complete: molecule.run_hartree_fock()

#   fns = {d:n for d, n in zip(molecule.domains, molecule._hf_fns)}

#   print()
#   print([(d, d.functions) for d in molecule.domains])
#   if molecule._hf_fns:
#       try:
#           fns = {d:lim for d, lim in zip(molecule.domains, molecule._hf_fns)}
#       except TypeError:
#           fns = {d:molecule._hf_fns for d in molecule.domains}
#   else:
#       fns = {d:d.functions for d in molecule.domains}

    for dom in molecule.domains:
        i = dom.num - 1
        if dom.functions == 0: continue
        db_eris = dom.mo_double_bar_array[i]

        for a, b in combinations(range(dom.electrons), 2):
            for c, d in combinations(range(dom.electrons), 2):
                for r, s in combinations(
                        range(dom.electrons, fns[i]), 2):
                    suma += (db_eris[a, r, b, s] * db_eris[a, c, b, d] *
                            db_eris[c, r, d, s] / (
                            (dom.orbital_energies[a] + dom.orbital_energies[b] -
                             dom.orbital_energies[r] - dom.orbital_energies[s]) *
                            (dom.orbital_energies[c] + dom.orbital_energies[d] -
                             dom.orbital_energies[r] - dom.orbital_energies[s])))

        for a, b in combinations(range(dom.electrons), 2):
            for r, s in combinations(range(dom.electrons, fns[i]), 2):
                for t, u in combinations(
                        range(dom.electrons, fns[i]), 2):
                    sumb += (db_eris[a, r, b, s] * db_eris[r, t, s, u] *
                            db_eris[a, t, b, u] / (
                            (dom.orbital_energies[a] + dom.orbital_energies[b] -
                             dom.orbital_energies[r] - dom.orbital_energies[s]) *
                            (dom.orbital_energies[a] + dom.orbital_energies[b] -
                             dom.orbital_energies[t] - dom.orbital_energies[u])))

        for a, b, c in product(range(dom.electrons), repeat=3):
            for r, s, t in product(range(dom.electrons, fns[i]),
                    repeat=3):
                sumc += (db_eris[a, r, b, s] * db_eris[c, t, s, b] *
                        db_eris[r, a, t, c] / (
                        (dom.orbital_energies[a] + dom.orbital_energies[b] -
                         dom.orbital_energies[r] - dom.orbital_energies[s]) *
                        (dom.orbital_energies[a] + dom.orbital_energies[c] -
                         dom.orbital_energies[r] - dom.orbital_energies[t])))

    for dom_one, dom_two in combinations(molecule.domains, 2):
        i = dom_one.num - 1
        j = dom_two.num - 1
        if dom_one.functions == 0 or dom_two.functions == 0: continue
        db_eris = dom_one.mo_double_bar_array[j]

        for a, c in product(range(dom_one.electrons), repeat=2):
            for b, d in product(range(dom_two.electrons), repeat=2):
                for r in range(dom_one.electrons, fns[i]):
                    for s in range(dom_two.electrons, fns[j]):
                        suma += (db_eris[a, r, b, s] * db_eris[a, c, b, d] *
                                db_eris[c, r, d, s] / (
                                (dom_one.orbital_energies[a] + dom_two.orbital_energies[b] -
                                 dom_one.orbital_energies[r] - dom_two.orbital_energies[s]) *
                                (dom_one.orbital_energies[c] + dom_two.orbital_energies[d] -
                                 dom_one.orbital_energies[r] - dom_two.orbital_energies[s])))

        for a in range(dom_one.electrons):
            for b in range(dom_two.electrons):
                for r, t in product(range(dom_one.electrons, fns[i]), repeat=2):
                    for s, u in product(range(dom_two.electrons, fns[j]), repeat=2):
                        sumb += (db_eris[a, r, b, s] * db_eris[r, t, s, u] *
                                db_eris[a, t, b, u] / (
                                (dom_one.orbital_energies[a] + dom_two.orbital_energies[b] -
                                 dom_one.orbital_energies[r] - dom_two.orbital_energies[s]) *
                                (dom_one.orbital_energies[a] + dom_two.orbital_energies[b] -
                                 dom_one.orbital_energies[t] - dom_two.orbital_energies[u])))

    for dom_one, dom_two in permutations(molecule.domains, 2):
        if dom_one.functions == 0 or dom_two.functions == 0: continue
        i = dom_one.num - 1
        j = dom_two.num - 1
        db_eris = dom_one.mo_double_bar_array[j]
        db_eris_b = dom_one.mo_double_bar_array[i]

        for a, b in permutations(range(dom_one.electrons), 2):
            for c in range(dom_two.electrons):
                for r, s in permutations(range(dom_one.electrons, fns[i]), 2):
                    for t in range(dom_two.electrons, fns[j]):
                        sumc += (2 * db_eris_b[a, r, b, s] *
                                db_eris[s, b, c, t] * db_eris[r, a, t, c] / (
                                (dom_one.orbital_energies[a] + dom_one.orbital_energies[b] -
                                dom_one.orbital_energies[r] - dom_one.orbital_energies[s]) *
                                (dom_one.orbital_energies[a] + dom_two.orbital_energies[c] -
                                dom_one.orbital_energies[r] - dom_two.orbital_energies[t])))

        db_eris_b = dom_two.mo_double_bar_array[j]

        for a in range(dom_one.electrons):
            for b, c in product(range(dom_two.electrons), repeat=2):
                for r in range(dom_one.electrons, fns[i]):
                    for s, t in product(range(dom_two.electrons, fns[j]), repeat=2):
                        sumc += (db_eris[a, r, b, s] * db_eris_b[c, t, s, b] *
                                db_eris[r, a, t, c] / (
                                (dom_one.orbital_energies[a] + dom_two.orbital_energies[b] -
                                 dom_one.orbital_energies[r] - dom_two.orbital_energies[s]) *
                                (dom_one.orbital_energies[a] + dom_two.orbital_energies[c] -
                                 dom_one.orbital_energies[r] - dom_two.orbital_energies[t])))

                for r in range(dom_two.electrons, fns[j]):
                    for s, t in product(range(dom_one.electrons, fns[i]), repeat=2):
                        sumc -= (db_eris[a, s, b, r] * db_eris[s, t, c, b] *
                                 db_eris[t, a, r, c] / (
                                (dom_one.orbital_energies[a] + dom_two.orbital_energies[b] -
                                 dom_two.orbital_energies[r] - dom_one.orbital_energies[s]) *
                                (dom_one.orbital_energies[a] + dom_two.orbital_energies[c] -
                                 dom_two.orbital_energies[r] - dom_one.orbital_energies[t])))

    for dom_one, dom_two, dom_three in permutations(molecule.domains, 3):
        if (dom_one.functions == 0 or dom_two.functions == 0 or
                dom_three.functions == 0): continue
        i = dom_one.num - 1
        j = dom_two.num - 1
        k = dom_three.num - 1
        db_eris = dom_one.mo_double_bar_array[j]
        db_eris_b = dom_three.mo_double_bar_array[j]
        db_eris_c = dom_one.mo_double_bar_array[k]

        for a, r in product(range(dom_one.electrons),
                range(dom_one.electrons, fns[i])):
            for b, s in product(range(dom_two.electrons),
                    range(dom_two.electrons, fns[j])):
                for c, t in product(range(dom_three.electrons),
                        range(dom_three.electrons, fns[k])):
                    sumc += (db_eris[a, r, b, s] * db_eris_b[c, t, s, b] *
                             db_eris_c[r, a, t, c] / (
                            (dom_one.orbital_energies[a] + dom_two.orbital_energies[b] -
                             dom_one.orbital_energies[r] - dom_two.orbital_energies[s]) *
                            (dom_one.orbital_energies[a] + dom_three.orbital_energies[c] -
                             dom_one.orbital_energies[r] - dom_three.orbital_energies[t])))

    return suma + sumb + sumc

def mp3_domain_correction(domain):
    """Computes the MP3 correlation estimate within a 1D electron domain.

    Arguments:
        domain  (OneDDomain)    target domain
    """

    cdef int a, b, c, d, r, s, t, u
    cdef np.double_t suma = 0
    cdef np.double_t sumb = 0
    cdef np.double_t sumc = 0
    cdef np.double_t[:,:,:,:] db_eris

    if not domain.molecule._hf_complete:
        domain.molecule.run_hartree_fock()
    if domain.electrons == 0 or domain.functions == 0:
        return 0

    i = domain.molecule.domains.index(domain)
    db_eris = domain.mo_double_bar_array[i]

    for a, b in combinations(range(domain.electrons), 2):
        for c, d in combinations(range(domain.electrons), 2):
            for r, s in combinations(
                    range(domain.electrons, domain.functions), 2):
                suma += (db_eris[a, r, b, s] * db_eris[a, c, b, d] *
                        db_eris[c, r, d, s] / (
                        (domain.orbital_energies[a] + domain.orbital_energies[b] -
                         domain.orbital_energies[r] - domain.orbital_energies[s]) *
                        (domain.orbital_energies[c] + domain.orbital_energies[d] -
                         domain.orbital_energies[r] - domain.orbital_energies[s])))

    for a, b in combinations(range(domain.electrons), 2):
        for r, s in combinations(range(domain.electrons, domain.functions), 2):
            for t, u in combinations(
                    range(domain.electrons, domain.functions), 2):
                sumb += (db_eris[a, r, b, s] * db_eris[r, t, s, u] *
                        db_eris[a, t, b, u] / (
                        (domain.orbital_energies[a] + domain.orbital_energies[b] -
                         domain.orbital_energies[r] - domain.orbital_energies[s]) *
                        (domain.orbital_energies[a] + domain.orbital_energies[b] -
                         domain.orbital_energies[t] - domain.orbital_energies[u])))

    for a, b, c in product(range(domain.electrons), repeat=3):
        for r, s, t in product(range(domain.electrons, domain.functions),
                repeat=3):
            sumc += (db_eris[a, r, b, s] * db_eris[c, t, s, b] *
                    db_eris[r, a, t, c] / (
                    (domain.orbital_energies[a] + domain.orbital_energies[b] -
                     domain.orbital_energies[r] - domain.orbital_energies[s]) *
                    (domain.orbital_energies[a] + domain.orbital_energies[c] -
                     domain.orbital_energies[r] - domain.orbital_energies[t])))

    return suma + sumb + sumc



