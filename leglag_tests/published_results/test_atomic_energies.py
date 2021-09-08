"""
Tests that compare leglag's output to previously published results.
"""
from decimal import Decimal

import pytest

from leglag_tests.utilities import (
    build_molecule,
    hf_energy,
    matching_figures,
    mp2_energy,
    mp3_energy,
)


@pytest.mark.published
@pytest.mark.parametrize(
    "formula, published_energy",
    [
        ("0H1", Decimal("-0.500000")),
        ("1He1", Decimal("-3.242922")),
        ("1Li2", Decimal("-8.007756")),
        ("2Be2", Decimal("-15.415912")),
        ("2B3", Decimal("-25.35751")),
        ("3C3", Decimal("-38.09038")),
        ("3N4", Decimal("-53.569")),
        ("4O4", Decimal("-71.9293")),
        ("4F5", Decimal("-93.1")),
        ("5Ne5", Decimal("-117.31")),
    ],
)
def test_hartree_fock_energy_of_atom(infinite_basis, formula, published_energy):
    """Tests that the Hartree Fock energy as calculated by leglag matches the
    previously published energies.
    """
    energy = Decimal(
        hf_energy(build_molecule(formula, [], infinite_basis, infinite_basis))
    )
    last_energy = Decimal(
        hf_energy(build_molecule(formula, [], infinite_basis - 1, infinite_basis - 1))
    )

    assert energy.quantize(published_energy) >= published_energy

    converged_energy = matching_figures(energy, last_energy)

    if converged_energy is None:
        pytest.skip("No digits converged")
    assert converged_energy == published_energy.quantize(converged_energy)


@pytest.mark.published
@pytest.mark.parametrize(
    "formula, published_energy",
    [
        ("0H1", Decimal("-0.500000")),
        ("1He1", Decimal("-3.244986")),
        ("1Li2", Decimal("-8.01112")),
        ("2Be2", Decimal("-15.4226")),
        ("2B3", Decimal("-25.3671")),
        ("3C3", Decimal("-38.105")),
        ("3N4", Decimal("-53.59")),
        ("4O4", Decimal("-71.95")),
        ("4F5", Decimal("-93.2")),
        ("5Ne5", Decimal("-117.35")),
    ],
)
def test_second_order_moller_plesset_energy_of_atom(
    infinite_basis, formula, published_energy
):
    """Tests that the second order Moller-Plesset energy as calculated by
    leglag matches the previously published energies.
    """
    energy = Decimal(
        mp2_energy(build_molecule(formula, [], infinite_basis, infinite_basis))
    )
    last_energy = Decimal(
        mp2_energy(build_molecule(formula, [], infinite_basis - 1, infinite_basis - 1))
    )

    assert energy.quantize(published_energy) >= published_energy

    converged_energy = matching_figures(energy, last_energy)

    if converged_energy is None:
        pytest.skip("No digits converged")
    assert converged_energy == published_energy.quantize(converged_energy)


@pytest.mark.published
@pytest.mark.parametrize(
    "formula, published_energy",
    [
        ("0H1", Decimal("-0.500000")),
        ("1He1", Decimal("-3.245611")),
        ("1Li2", Decimal("-8.01179")),
        ("2Be2", Decimal("-15.4236")),
        ("2B3", Decimal("-25.3684")),
        ("3C3", Decimal("-38.107")),
        ("3N4", Decimal("-53.6")),
        ("4O4", Decimal("-71.96")),
        ("4F5", Decimal("-93.2")),
        ("5Ne5", Decimal("-117.35")),
    ],
)
def test_third_order_moller_plesset_energy_of_atom(
    infinite_basis, formula, published_energy
):
    """Tests that the third order Moller-Plesset energy as calculated by
    leglag matches the previously published energies.
    """
    energy = Decimal(
        mp3_energy(build_molecule(formula, [], infinite_basis, infinite_basis))
    )
    last_energy = Decimal(
        mp3_energy(build_molecule(formula, [], infinite_basis - 1, infinite_basis - 1))
    )

    assert energy.quantize(published_energy) >= published_energy

    converged_energy = matching_figures(energy, last_energy)

    if converged_energy is None:
        pytest.skip("No digits converged")
    assert converged_energy == published_energy.quantize(converged_energy)
