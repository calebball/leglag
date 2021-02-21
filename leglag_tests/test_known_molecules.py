"""
Tests that compare leglag's output to published calculations.
"""
import pytest
import numpy as np

from leglag import OneDMolecule


@pytest.fixture
def infinite_basis():
    return 10


@pytest.fixture
def finite_basis():
    return 10


@pytest.fixture
def domspec_factory(infinite_basis, finite_basis):
    def domspec_factory(nucspec, electrons):
        finite_domains = [
            (idx + 2, left_boundary, right_boundary, electron_count, finite_basis)
            for idx, left_boundary, right_boundary, electron_count in zip(
                range(len(nucspec)),
                [nucleus[0] for nucleus in nucspec],
                [nucleus[0] for nucleus in nucspec[1:]],
                electrons[1:-1],
            )
        ]
        return (
            [(1, nucspec[0][0], False, 2, electrons[0], infinite_basis)]
            + finite_domains
            + [
                (
                    len(nucspec) + 1,
                    nucspec[-1][0],
                    True,
                    2,
                    electrons[-1],
                    infinite_basis,
                )
            ]
        )

    return domspec_factory


def energy_with_basis_convergence(
    nucspec, domspec, energy_function=OneDMolecule.hf_energy
):
    molecule = OneDMolecule(nucspec, domspec)
    molecule_less_one_bf = OneDMolecule(nucspec, domspec)
    for domain in molecule_less_one_bf.domains:
        if domain.functions:
            domain.functions -= 1

    energy = energy_function(molecule)
    energy_less_one_bf = energy_function(molecule_less_one_bf)

    energy_difference = energy_less_one_bf - energy
    converged_tolerance = 10 ** np.ceil(np.log10(energy_difference))

    return energy, converged_tolerance


@pytest.mark.parametrize(
    "nucleus,known_energy",
    [
        (1, -0.5),
        (2, -3.242922),
        (3, -8.007756),
        (4, -15.415912),
        (5, -25.35751),
        (6, -38.09038),
    ],
)
def test_atomic_hartree_fock_energy(infinite_basis, nucleus, known_energy):
    hf_energy, basis_tolerance = energy_with_basis_convergence(
        [(0, nucleus)],
        [
            (1, 0, False, 2, int(np.floor(nucleus / 2)), infinite_basis),
            (2, 0, True, 2, int(np.ceil(nucleus / 2)), infinite_basis),
        ],
    )
    if np.isnan(basis_tolerance) or basis_tolerance > np.log10(abs(hf_energy)):
        pytest.skip("No digits have converged with respect to the basis set")

    assert np.round(hf_energy, decimals=6) >= known_energy
    assert np.isclose(hf_energy, known_energy, atol=basis_tolerance)


@pytest.mark.parametrize(
    "nucspec,electrons,known_energy",
    [
        ([(0, 1), (2.636, 1)], [0, 1, 1], -1.184572),
        ([(0, 1), (2.025, 2)], [1, 1, 1], -3.880313),
        ([(0, 1), (5.345, 3)], [0, 2, 2], -8.544163),
        ([(0, 1), (5.152, 3)], [1, 2, 1], -8.681782),
    ],
)
def test_diatomic_hartree_fock_energies(
    domspec_factory, nucspec, electrons, known_energy
):
    hf_energy, basis_tolerance = energy_with_basis_convergence(
        nucspec, domspec_factory(nucspec, electrons)
    )
    if np.isnan(basis_tolerance) or basis_tolerance > np.log10(abs(hf_energy)):
        pytest.skip("No digits have converged with respect to the basis set")

    assert np.round(hf_energy, decimals=6) >= known_energy
    assert np.isclose(hf_energy, known_energy, atol=basis_tolerance)
