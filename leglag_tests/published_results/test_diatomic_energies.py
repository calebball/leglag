"""
Tests that compare leglag's output to previously published results.
"""
from decimal import Decimal

import pytest
import numpy as np

from leglag_tests.utilities import (
    build_molecule,
    hf_energy,
    matching_figures,
    mp2_energy,
    mp3_energy,
)


@pytest.mark.published
@pytest.mark.parametrize(
    "formula, bond_length, published_energy",
    [
        ("0H1H1", 2.636, Decimal("-1.184572")),
        ("1H1He1", 2.025, Decimal("-3.880313")),
        ("0H2Li2", 5.345, Decimal("-8.544163")),
        ("1H2Li1", 5.152, Decimal("-8.681782")),
        ("1H2Be2", 3.966, Decimal("-16.079548")),
        ("1H3B2", 8.880, Decimal("-26.020047")),
        ("1H2B3", 3.298, Decimal("-25.957601")),
        ("0H3B3", 10.349, Decimal("-25.863890")),
        ("1H3C3", 6.666, Decimal("-38.756672")),
        ("1H4N3", 14.316, Decimal("-54.22372")),
        ("1H3N4", 5.407, Decimal("-54.218224")),
        ("0H4N4", 19.20, Decimal("-54.0703")),
        ("1H4O4", 10.468, Decimal("-72.590721")),
        ("1He2Li2", 4.606, Decimal("-11.260655")),
        ("1He3B3", 11.174, Decimal("-28.600892")),
        ("1Li3Li2", 8.693, Decimal("-16.064647")),
        ("2Li3Be2", 7.050, Decimal("-23.452479")),
        ("2Li4B2", 13.330, Decimal("-33.418876")),
        ("1Li4B3", 14.007, Decimal("-33.379031")),
        ("2Li4C3", 10.435, Decimal("-46.140625")),
        ("2Li4N4", 8.956, Decimal("-61.588547")),
        ("2Li5N3", 19.552, Decimal("-61.63192")),
        ("1Li5N4", 21.546, Decimal("-61.5802")),
        ("2Li5O4", 14.943, Decimal("-79.987067")),
        ("2Be4B3", 12.566, Decimal("-40.776885")),
        ("2Be5N4", 20.571, Decimal("-68.9851")),
        ("2B5B3", 19.349, Decimal("-50.733908")),
        ("3B5C3", 16.040, Decimal("-63.457101")),
        ("3B6N3", 26.480, Decimal("-78.949")),
        ("2B6N4", 27.514, Decimal("-78.933")),
        ("3B6O4", 21.138, Decimal("-97.3018")),
        ("3C6N4", 22.880, Decimal("-91.660")),
        ("3N7N4", 30.301, Decimal("-107.14")),
        ("4N7O4", 29.583, Decimal("-125.50")),
    ],
)
def test_hartree_fock_energy_of_diatomic(
    infinite_basis, finite_basis, formula, bond_length, published_energy
):
    """Tests that the Hartree Fock energy as calculated by leglag matches the
    previously published energies.
    """
    energy = Decimal(
        hf_energy(build_molecule(formula, [bond_length], infinite_basis, finite_basis))
    )
    last_energy = Decimal(
        hf_energy(
            build_molecule(formula, [bond_length], infinite_basis - 1, finite_basis - 1)
        )
    )

    assert energy.quantize(published_energy) >= published_energy

    converged_energy = matching_figures(energy, last_energy)

    if converged_energy is None:
        pytest.skip("No digits converged")
    assert converged_energy == published_energy.quantize(converged_energy)


@pytest.mark.published
@pytest.mark.parametrize(
    "formula, bond_length, published_energy",
    [
        ("0H1H1", 2.637, Decimal("-1.185418")),
        ("1H1He1", 2.027, Decimal("-3.882619")),
        ("0H2Li2", 5.323, Decimal("-8.547920")),
        ("1H2Li1", 5.141, Decimal("-8.686367")),
        ("1H2Be2", 3.961, Decimal("-16.08707")),
        ("1H3B2", 8.810, Decimal("-26.0310")),
        ("1H2B3", 3.296, Decimal("-25.96793")),
        ("0H3B3", 10.238, Decimal("-25.8736")),
        ("1H3C3", 6.635, Decimal("-38.7721")),
        ("1H4N3", 14.268, Decimal("-54.244")),
        ("1H3N4", 5.392, Decimal("-54.2379")),
        ("0H4N4", 18.168, Decimal("-54.089")),
        ("1H4O4", 10.383, Decimal("-72.616")),
        ("1He2Li2", 4.586, Decimal("-11.266223")),
        ("1He3B3", 11.170, Decimal("-28.6126")),
        ("1Li3Li2", 8.644, Decimal("-16.07183")),
        ("2Li3Be2", 7.000, Decimal("-23.46286")),
        ("2Li4B2", 13.228, Decimal("-33.4323")),
        ("1Li4B3", 13.999, Decimal("-33.3922")),
        ("2Li4C3", 10.358, Decimal("-46.1588")),
        ("2Li4N4", 8.892, Decimal("-61.6112")),
        ("2Li5N3", 19.258, Decimal("-61.654")),
        ("1Li5N4", 21.092, Decimal("-61.602")),
        ("2Li5O4", 14.769, Decimal("-80.015")),
        ("2Be4B3", 12.566, Decimal("-40.7932")),
        ("2Be5N4", 19.884, Decimal("-69.010")),
        ("2B5B3", 19.233, Decimal("-50.753")),
        ("3B5C3", 16.009, Decimal("-63.481")),
        ("3B6N3", 25.946, Decimal("-78.977")),
        ("2B6N4", 26.939, Decimal("-78.961")),
        ("3B6O4", 20.799, Decimal("-97.34")),
        ("3C6N4", 24.906, Decimal("-91.69")),
        ("3N7N4", 31.892, Decimal("-107.18")),
        ("4N7O4", 28.780, Decimal("-125.54")),
    ],
)
def test_second_order_moller_plesset_energy_of_diatomic(
    infinite_basis, finite_basis, formula, bond_length, published_energy
):
    """Tests that the second order Moller-Plesset energy as calculated by
    leglag matches the previously published energies.
    """
    energy = Decimal(
        mp2_energy(build_molecule(formula, [bond_length], infinite_basis, finite_basis))
    )
    last_energy = Decimal(
        mp3_energy(
            build_molecule(formula, [bond_length], infinite_basis - 1, finite_basis - 1)
        )
    )

    assert energy.quantize(published_energy) >= published_energy

    converged_energy = matching_figures(energy, last_energy)

    if converged_energy is None:
        pytest.skip("No digits converged")
    assert converged_energy == published_energy.quantize(converged_energy)


@pytest.mark.published
@pytest.mark.parametrize(
    "formula, bond_length, published_energy",
    [
        ("0H1H1", 2.638, Decimal("-1.185728")),
        ("1H1He1", 2.027, Decimal("-3.883301")),
        ("0H2Li2", 5.320, Decimal("-8.548659")),
        ("1H2Li1", 5.142, Decimal("-8.687589")),
        ("1H2Be2", 3.962, Decimal("-16.08845")),
        ("1H3B2", 8.806, Decimal("-26.0329")),
        ("1H2B3", 3.296, Decimal("-25.96949")),
        ("0H3B3", 10.235, Decimal("-25.8748")),
        ("1H3C3", 6.633, Decimal("-38.7745")),
        ("1H4N3", 14.257, Decimal("-54.247")),
        ("1H3N4", 5.392, Decimal("-54.2407")),
        ("0H4N4", 18.131, Decimal("-54.091")),
        ("1H4O4", 10.378, Decimal("-72.620")),
        ("1He2Li2", 4.584, Decimal("-11.267543")),
        ("1He3B3", 11.003, Decimal("-28.6145")),
        ("1Li3Li2", 8.637, Decimal("-16.07326")),
        ("2Li3Be2", 6.996, Decimal("-23.46460")),
        ("2Li4B2", 13.157, Decimal("-33.4343")),
        ("1Li4B3", 13.778, Decimal("-33.3941")),
        ("2Li4C3", 10.336, Decimal("-46.1614")),
        ("2Li4N4", 8.884, Decimal("-61.6143")),
        ("2Li5N3", 19.229, Decimal("-61.658")),
        ("1Li5N4", 21.099, Decimal("-61.605")),
        ("2Li5O4", 17.748, Decimal("-80.019")),
        ("2Be4B3", 12.381, Decimal("-40.7955")),
        ("2Be5N4", 19.869, Decimal("-69.014")),
        ("2B5B3", 19.003, Decimal("-50.756")),
        ("3B5C3", 15.779, Decimal("-63.484")),
        ("3B6N3", 25.912, Decimal("-78.981")),
        ("2B6N4", 26.869, Decimal("-78.96")),
        ("3B6O4", 20.735, Decimal("-97.34")),
        ("3C6N4", 24.801, Decimal("-91.70")),
        ("3N7N4", 33.989, Decimal("-107.19")),
        ("4N7O4", 28.727, Decimal("-125.55")),
    ],
)
def test_third_order_moller_plesset_energy_of_diatomic(
    infinite_basis, finite_basis, formula, bond_length, published_energy
):
    """Tests that the third order Moller-Plesset energy as calculated by
    leglag matches the previously published energies.
    """
    energy = Decimal(
        mp3_energy(build_molecule(formula, [bond_length], infinite_basis, finite_basis))
    )
    last_energy = Decimal(
        mp3_energy(
            build_molecule(formula, [bond_length], infinite_basis - 1, finite_basis - 1)
        )
    )

    assert energy.quantize(published_energy) >= published_energy

    converged_energy = matching_figures(energy, last_energy)

    if converged_energy is None:
        pytest.skip("No digits converged")
    assert converged_energy == published_energy.quantize(converged_energy)
