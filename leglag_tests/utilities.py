import copy
import re
from decimal import Decimal
from itertools import accumulate, chain, takewhile
from typing import List, Optional

from leglag.one_d_domain import FinDomain, InfDomain
from leglag.one_d_molecule import OneDMolecule


nuclear_charge = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
}


def hf_energy(molecule: OneDMolecule) -> float:
    """Evaluates the Hartree-Fock energy of a given molecule."""
    return molecule.hf_energy()


def mp2_energy(molecule: OneDMolecule) -> float:
    """Evaluates the second order Moller-Plesset corrected energy of a given
    molecule.
    """
    return molecule.hf_energy() + molecule.mp2_correction()


def mp3_energy(molecule: OneDMolecule) -> float:
    """Evaluates the third order Moller-Plesset corrected energy of a given
    molecule.
    """
    return molecule.hf_energy() + molecule.mp2_correction() + molecule.mp3_correction()


def build_molecule(
    molecule_string: str, distances: List[float], infinite_basis: int, finite_basis: int
) -> OneDMolecule:
    """Creates a molecule from a 'chemical formula' and a geometry.

    This is an extra level of indirection on top of the constructor provided by
    the library. This is to make it easier to alter the interface in the future.

    Arguments:
        molecule_string: the formula for the molecule (e.g. 1H2Li1.)
        distances: the distances between each nucleus.
        infinite_basis: the number of basis functions used in the infinite
            domains.
        finite_basis: the number of basis functions used in the finite domains.

    Returns:
        A OneDMolecule.
    """
    electrons = [int(domain) for domain in re.split(r"[a-zA-Z]+", molecule_string)]
    nuclei = [
        nuclear_charge[nucleus] for nucleus in re.split(r"\d+", molecule_string)[1:-1]
    ]
    positions = list(chain([0], accumulate(distances)))

    domains = [(1, 0, False, 2, electrons[0], infinite_basis)]
    domains.extend(
        [
            (
                idx,
                left,
                right,
                elecs,
                finite_basis,
            )
            for idx, left, right, elecs in zip(
                range(2, len(electrons)), positions, positions[1:], electrons[1:-1]
            )
        ]
    )
    domains.append(
        (len(electrons), positions[-1], True, 2, electrons[-1], infinite_basis)
    )

    return OneDMolecule(
        [(pos, nucleus) for pos, nucleus in zip(positions, nuclei)], domains
    )


def are_same_magnitude(left: Decimal, right: Decimal) -> bool:
    """Checks if two Decimal instances are the same order of magnitude."""
    return left.adjusted() == right.adjusted()


def significant_figures(number: Decimal, figures: int) -> Decimal:
    """Rounds a Decimal instance to a given number of significant figures."""
    return round(number, figures - number.adjusted() - 1)


def matching_figures(left: Decimal, right: Decimal) -> Optional[Decimal]:
    """Determines the matching significant figures of two Decimal instances."""
    if not are_same_magnitude(left, right):
        return None

    left_rounded = (significant_figures(left, figs) for figs in range(1, 20))
    right_rounded = (significant_figures(right, figs) for figs in range(1, 20))
    digits = len(
        [
            _
            for _ in takewhile(
                lambda tup: tup[0] == tup[1], zip(left_rounded, right_rounded)
            )
        ]
    )
    # The last matching digit still has some uncertainty because it could end
    # up rounded to different number if the digit below it is wrong.
    digits -= 1

    if digits < 2:
        return None

    return significant_figures(left, digits - 1)


def translate_molecule(molecule: OneDMolecule, translation: float) -> OneDMolecule:
    """Moves an entire molecule a given distance."""
    molecule = copy.deepcopy(molecule)
    molecule.nuclei = [
        (position + translation, charge) for position, charge in molecule.nuclei
    ]
    for domain in molecule.domains:
        if isinstance(domain, InfDomain):
            domain.position += translation

        if isinstance(domain, FinDomain):
            domain.position = (
                domain.position[0] + translation,
                domain.position[1] + translation,
            )

    return molecule
