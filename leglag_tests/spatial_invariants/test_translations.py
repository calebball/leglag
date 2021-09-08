"""
Tests that calculations are invariant under translation.
"""
import pytest
import numpy as np

from leglag_tests.utilities import build_molecule, hf_energy, translate_molecule


translations = [1, 2.5, -2 / 3]


@pytest.mark.parametrize("translation", translations)
def test_hartree_fock_translation_invariance_on_hydrogen(infinite_basis, translation):
    """Translating a molecule in space shouldn't affect the Hartree Fock
    energy.

    A single Hydrogen atom should reveal any issues with single infinite domain
    integrals.
    """
    molecule = build_molecule("0H1", [], infinite_basis, 0)
    translated_molecule = translate_molecule(molecule, translation)
    assert np.isclose(
        hf_energy(molecule), hf_energy(translated_molecule), atol=0, rtol=1e-8
    )


@pytest.mark.parametrize("translation", translations)
def test_hartree_fock_translation_invariance_on_helium(infinite_basis, translation):
    """Translating a molecule in space shouldn't affect the Hartree Fock
    energy.

    A single Helium atom should reveal any issues with two-electron integrals
    across different domains.
    """
    molecule = build_molecule("1He1", [], infinite_basis, 0)
    translated_molecule = translate_molecule(molecule, translation)
    assert np.isclose(
        hf_energy(molecule), hf_energy(translated_molecule), atol=0, rtol=1e-8
    )


@pytest.mark.parametrize("translation", translations)
def test_hartree_fock_translation_invariance_on_lithium(infinite_basis, translation):
    """Translating a molecule in space shouldn't affect the Hartree Fock
    energy.

    A single Lithium atom should reveal any issues with two-electron integrals
    within an infinite domain.
    """
    molecule = build_molecule("1He1", [], infinite_basis, 0)
    translated_molecule = translate_molecule(molecule, translation)
    assert np.isclose(
        hf_energy(molecule), hf_energy(translated_molecule), atol=0, rtol=1e-8
    )


@pytest.mark.parametrize("translation", translations)
def test_hartree_fock_translation_invariance_on_H2_ion(
    infinite_basis, finite_basis, translation
):
    """Translating a molecule in space shouldn't affect the Hartree Fock
    energy.

    An H2+ ion should reveal any issues with single finite domain integrals.
    """
    molecule = build_molecule("0H1H0", [2.636], infinite_basis, finite_basis)
    translated_molecule = translate_molecule(molecule, translation)
    assert np.isclose(
        hf_energy(molecule), hf_energy(translated_molecule), atol=0, rtol=1e-8
    )


@pytest.mark.parametrize("translation", translations)
def test_hartree_fock_translation_invariance_on_HLi_ion(
    infinite_basis, finite_basis, translation
):
    """Translating a molecule in space shouldn't affect the Hartree Fock
    energy.

    An HLi++ ion should reveal any issues with two-electron integrals in a
    finite domain.
    """
    molecule = build_molecule("0H2Li0", [5.142], infinite_basis, finite_basis)
    translated_molecule = translate_molecule(molecule, translation)
    assert np.isclose(
        hf_energy(molecule), hf_energy(translated_molecule), atol=0, rtol=1e-8
    )


@pytest.mark.parametrize("translation", translations)
def test_hartree_fock_translation_invariance_on_HHe(
    infinite_basis, finite_basis, translation
):
    """Translating a molecule in space shouldn't affect the Hartree Fock
    energy.

    An HHe molecule should reveal any issues with two-electron integrals across
    finite domain and infinite domains.
    """
    molecule = build_molecule("1H1He1", [2.025], infinite_basis, finite_basis)
    translated_molecule = translate_molecule(molecule, translation)
    assert np.isclose(
        hf_energy(molecule), hf_energy(translated_molecule), atol=0, rtol=1e-8
    )


@pytest.mark.parametrize("translation", translations)
def test_hartree_fock_translation_invariance_on_HHH_ion(
    infinite_basis, finite_basis, translation
):
    """Translating a molecule in space shouldn't affect the Hartree Fock
    energy.

    An HHH+ molecule should reveal any issues with two-electron integrals
    between finite domains.
    """
    molecule = build_molecule("0H1H1H0", [2.636, 2.636], infinite_basis, finite_basis)
    translated_molecule = translate_molecule(molecule, translation)
    assert np.isclose(
        hf_energy(molecule), hf_energy(translated_molecule), atol=0, rtol=1e-8
    )
