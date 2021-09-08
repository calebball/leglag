"""
Tests that calculations are invariant under spatial inversion.
"""
import pytest
import numpy as np

from leglag_tests.utilities import build_molecule, hf_energy


@pytest.mark.parametrize(
    "formula_list",
    [
        ([0, "H", 1]),
        ([1, "He", 1]),
        ([1, "Li", 2]),
    ],
)
def test_hartree_fock_inversion_invariance_on_atoms(
    infinite_basis, finite_basis, formula_list
):
    """Inverting a molecule in space shouldn't affect the Hartree Fock energy."""
    molecule = build_molecule(
        "".join(map(str, formula_list)), [], infinite_basis, finite_basis
    )
    inverted = build_molecule(
        "".join(map(str, reversed(formula_list))), [], infinite_basis, finite_basis
    )
    assert np.isclose(hf_energy(molecule), hf_energy(inverted), atol=0, rtol=1e-8)


@pytest.mark.parametrize(
    "formula_list, distances",
    [
        ([0, "H", 1, "H", 0], [2.636]),
        ([0, "H", 1, "He", 0], [2.025]),
        ([0, "H", 2, "Li", 0], [5.152]),
    ],
)
def test_hartree_fock_inversion_invariance_on_diatomic_ions(
    infinite_basis, finite_basis, formula_list, distances
):
    """Inverting a molecule in space shouldn't affect the Hartree Fock energy."""
    molecule = build_molecule(
        "".join(map(str, formula_list)), distances, infinite_basis, finite_basis
    )
    inverted = build_molecule(
        "".join(map(str, reversed(formula_list))),
        distances,
        infinite_basis,
        finite_basis,
    )
    assert np.isclose(hf_energy(molecule), hf_energy(inverted), atol=0, rtol=1e-8)


@pytest.mark.parametrize(
    "formula_list, distances",
    [
        ([0, "H", 1, "H", 1], [2.636]),
        ([1, "H", 1, "He", 1], [2.025]),
        ([1, "H", 2, "Li", 1], [5.152]),
        ([0, "H", 2, "Li", 2], [5.345]),
    ],
)
def test_hartree_fock_inversion_invariance_on_diatomic_molecules(
    infinite_basis, finite_basis, formula_list, distances
):
    """Inverting a molecule in space shouldn't affect the Hartree Fock energy."""
    molecule = build_molecule(
        "".join(map(str, formula_list)), distances, infinite_basis, finite_basis
    )
    inverted = build_molecule(
        "".join(map(str, reversed(formula_list))),
        distances,
        infinite_basis,
        finite_basis,
    )
    assert np.isclose(hf_energy(molecule), hf_energy(inverted), atol=0, rtol=1e-8)


@pytest.mark.parametrize(
    "formula_list, distances",
    [
        ([0, "H", 1, "H", 1, "H", 1], [2.636, 2.636]),
        ([1, "H", 1, "H", 1, "He", 1], [2.636, 2.052]),
    ],
)
def test_hartree_fock_inversion_invariance_on_triatomic_molecules(
    infinite_basis, finite_basis, formula_list, distances
):
    """Inverting a molecule in space shouldn't affect the Hartree Fock energy."""
    molecule = build_molecule(
        "".join(map(str, formula_list)), distances, infinite_basis, finite_basis
    )
    inverted = build_molecule(
        "".join(map(str, reversed(formula_list))),
        reversed(distances),
        infinite_basis,
        finite_basis,
    )
    assert np.isclose(hf_energy(molecule), hf_energy(inverted), atol=0, rtol=1e-8)
