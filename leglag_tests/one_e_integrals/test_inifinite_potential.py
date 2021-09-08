"""
Tests that the potential energy integrals are correctly evaluated for infinite
domains.
"""
import numpy as np
import pytest
from hypothesis import given, strategies as st

from leglag.one_d_domain import InfDomain
from leglag.one_e_integrals import inf_potential


@pytest.mark.parametrize("side", [True, False])
@given(
    functions=st.integers(min_value=1, max_value=50),
)
def test_potential_matrix_shape(side, functions):
    """Tests that the shape of the one electron potential energy integral
    matrix is correct.
    """
    domain = InfDomain(0, 0, side, 2, 1, functions, None)
    matrix = inf_potential(domain, 0)
    assert matrix.shape == (
        functions,
        functions,
    )


unscaled_potential_matrix_at_0 = np.array(
    [
        [0.5,               np.sqrt(3) / 6,     np.sqrt(6) / 12,    np.sqrt(10) / 20,   np.sqrt(15) / 30],
        [np.sqrt(3) / 6,    0.5,                np.sqrt(2) / 4,     np.sqrt(30) / 20,   np.sqrt(5) / 10],
        [np.sqrt(6) / 12,   np.sqrt(2) / 4,     0.5,                np.sqrt(15) / 10,   np.sqrt(10) / 10],
        [np.sqrt(10) / 20,  np.sqrt(30) / 20,   np.sqrt(15) / 10,   0.5,                np.sqrt(6) / 6],
        [np.sqrt(15) / 30,  np.sqrt(5) / 10,    np.sqrt(10) / 10,   np.sqrt(6) / 6,     0.5],
    ]
)


@pytest.mark.parametrize("side", [True, False])
@given(
    alpha=st.floats(
        min_value=0, max_value=10 ** (np.log10(np.finfo(np.float64).max) / 2)
    ),
)
def test_potential_matrix_values_at_zero(side, alpha):
    """Tests that the kinetic matrix is correctly generated by comparing to an
    analytically constructed example.
    """
    domain = InfDomain(0, 0, side, alpha, 1, 5, None)
    matrix = inf_potential(domain, 0)
    assert np.isclose(
        matrix, 2 * alpha * unscaled_potential_matrix_at_0, atol=0, rtol=1e-14
    ).all()


potential_at_1_with_alpha_2 = np.array(
    [
        [
            0.60306079683378665927,
            0.12784011696937552659,
            0.037486952245518249652,
            0.013108762856806335803,
            0.0051495572630676102668,
        ],
        [
            0.12784011696937552659,
            0.51665968161851543467,
            0.15150171379034025222,
            0.052978434348840739448,
            0.020811687904276340999,
        ],
        [
            0.037486952245518249652,
            0.15150171379034025222,
            0.45911952506799888982,
            0.16054890079162115197,
            0.063068938478793636324,
        ],
        [
            0.013108762856806335803,
            0.052978434348840739448,
            0.16054890079162115197,
            0.41729904920753857386,
            0.16392892092041569535,
        ],
        [
            0.0051495572630676102668,
            0.020811687904276340999,
            0.063068938478793636324,
            0.16392892092041569535,
            0.38514368740123998198,
        ],
    ]
)


@pytest.mark.parametrize("position", [0, -1, 1.414])
def test_potential_matrix_values_at_distance_of_one(position):
    """Tests that the potential matrix is evaluated correctly at a distance of
    1 from the domain border.
    """
    domain = InfDomain(0, position, True, 2, 1, 5, None)
    matrix = inf_potential(domain, position + 1)
    assert np.isclose(matrix, potential_at_1_with_alpha_2, atol=0, rtol=1e-14).all()


potential_at_3_point_5_with_alpha_point_7 = np.array(
    [
        [
            0.18488870402209088733,
            0.034999413128998148669,
            0.0092834496726205206711,
            0.0029640617151554127543,
            0.0010707968626326696883,
        ],
        [
            0.034999413128998148669,
            0.15963467267289742612,
            0.042342437123216826208,
            0.013519284450203167054,
            0.0048839763694181906751,
        ],
        [
            0.0092834496726205206711,
            0.042342437123216826208,
            0.14252116220018659995,
            0.045504799980003726799,
            0.016439062926447980851,
        ],
        [
            0.0029640617151554127543,
            0.013519284450203167054,
            0.045504799980003726799,
            0.12994344770646939481,
            0.046943366736365071840,
        ],
        [
            0.0010707968626326696883,
            0.0048839763694181906751,
            0.016439062926447980851,
            0.046943366736365071840,
            0.12019739670863536310,
        ],
    ]
)


@pytest.mark.parametrize("position", [0, -1, 1.414])
def test_potential_matrix_values_at_distance_of_3_point_5(position):
    """Tests that the potential matrix is evaluated correctly at a distance of
    3.5 from the domain border.
    """
    domain = InfDomain(0, position, True, 0.7, 1, 5, None)
    matrix = inf_potential(domain, position + 3.5)
    assert np.isclose(
        matrix, potential_at_3_point_5_with_alpha_point_7, atol=0, rtol=1e-14
    ).all()