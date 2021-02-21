"""
Tests that calculations are invariant under certain spatial transformations.
"""
import pytest
import numpy as np

from leglag import OneDMolecule


@pytest.fixture
def infinite_basis():
    return 5


@pytest.fixture
def finite_basis():
    return 5


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


@pytest.mark.parametrize(
    "nucspec,electrons,shift",
    [
        ([(0, 1)], [0, 1], 1.414),
        ([(0, 1)], [0, 1], -3.142),
        ([(0, 3)], [2, 1], 1.414),
        ([(0, 3)], [2, 1], -3.142),
    ],
)
def test_hartree_fock_translation_invariance(
    domspec_factory, nucspec, electrons, shift
):
    molecule = OneDMolecule(nucspec, domspec_factory(nucspec, electrons))
    shifted_nucspec = [(nucleus[0] + shift, nucleus[1]) for nucleus in nucspec]
    shifted_molecule = OneDMolecule(
        shifted_nucspec, domspec_factory(shifted_nucspec, electrons)
    )
    assert np.isclose(
        shifted_molecule.hf_energy(), molecule.hf_energy(), atol=0, rtol=1e-8
    )


@pytest.mark.parametrize(
    "nucspec,electrons",
    [
        ([(0, 1)], [0, 1]),
        ([(0, 1), (2.636, 1)], [0, 1, 1]),
        ([(0, 1), (5.152, 3)], [1, 2, 1]),
    ],
)
def test_hartree_fock_inversion_invariance(domspec_factory, nucspec, electrons):
    molecule = OneDMolecule(nucspec, domspec_factory(nucspec, electrons))
    left_boundary = min(nucleus[0] for nucleus in nucspec)
    right_boundary = max(nucleus[0] for nucleus in nucspec)
    inverted_nucspec = sorted(
        [
            (right_boundary - (nucleus[0] - left_boundary), nucleus[1])
            for nucleus in nucspec
        ],
        key=lambda nucleus: nucleus[0],
    )
    inverted_molecule = OneDMolecule(
        inverted_nucspec, domspec_factory(inverted_nucspec, list(reversed(electrons)))
    )
    assert np.isclose(
        inverted_molecule.hf_energy(), molecule.hf_energy(), atol=0, rtol=1e-8
    )
