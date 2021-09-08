"""
Tests that the potential energy integrals are correctly evaluated for finite
domains.
"""
import numpy as np
import pytest
from hypothesis import assume, given, strategies as st

from leglag.one_d_domain import FinDomain
from leglag.one_e_integrals import fin_potential


@given(
    functions=st.integers(min_value=1, max_value=50),
)
def test_potential_matrix_shape(functions):
    """Tests that the shape of the one electron potential energy integral
    matrix is correct.
    """
    domain = FinDomain(0, 0, 1, 1, functions, None)
    matrix = fin_potential(domain, 0)
    assert matrix.shape == (functions, functions)


left_side_sign_inversions = np.fromiter(
    ((-1) ** (m + n) for m in range(1, 6) for n in range(1, 6)), int
).reshape((5, 5))


unscaled_potential_matrix_at_border = np.sqrt(
    np.array(
        [
            [25 / 4, 7 / 4, 3 / 4, 11 / 28, 13 / 56],
            [7 / 4, 49 / 4, 21 / 4, 11 / 4, 13 / 8],
            [3 / 4, 21 / 4, 81 / 4, 297 / 28, 351 / 56],
            [11 / 28, 11 / 4, 297 / 28, 121 / 4, 143 / 8],
            [13 / 56, 13 / 8, 351 / 56, 143 / 8, 169 / 4],
        ]
    )
)


def potential_matrix_at_border_wont_overflow(interval):
    return unscaled_potential_matrix_at_border.max() / np.finfo(np.float64).max <= abs(
        interval[1] - interval[0]
    )


@given(
    interval=st.tuples(
        st.floats(allow_infinity=False, allow_nan=False),
        st.floats(allow_infinity=False, allow_nan=False),
    )
    .map(sorted)
    .filter(potential_matrix_at_border_wont_overflow)
)
def test_potential_matrix_at_right_edge(interval):
    center = sum(interval) / 2
    half_width = abs(interval[0] - interval[1]) / 2
    assume(not np.isinf(center))
    assume(not np.isinf(half_width))
    assume(half_width > 0)
    domain = FinDomain(0, interval[0], interval[1], 1, 5, None)
    matrix = fin_potential(domain, center + half_width)
    assert np.isclose(
        matrix,
        0.5 * unscaled_potential_matrix_at_border / half_width,
        atol=0,
        rtol=1e-14,
    ).all()


@given(
    interval=st.tuples(
        st.floats(allow_infinity=False, allow_nan=False),
        st.floats(allow_infinity=False, allow_nan=False),
    )
    .map(sorted)
    .filter(potential_matrix_at_border_wont_overflow)
)
def test_potential_matrix_at_left_edge(interval):
    center = sum(interval) / 2
    half_width = center - interval[0]
    assume(not np.isinf(center))
    assume(not np.isinf(half_width))
    assume(half_width > 0)
    domain = FinDomain(0, interval[0], interval[1], 1, 5, None)
    matrix = fin_potential(domain, center - half_width)
    assert np.isclose(
        matrix,
        0.5 * left_side_sign_inversions
        * unscaled_potential_matrix_at_border
        / half_width,
        atol=0,
        rtol=1e-14,
    ).all()


unscaled_potential_matrix_at_one_width = 4 * np.array(
    [
        [
            5 * (27 * np.log(3) - 28) / 32,
            np.sqrt(7) * (135 * np.log(3) - 148) / 16,
            np.sqrt(3) * (3645 * np.log(3) - 4004) / 64,
            np.sqrt(11 / 7) * 5 * (2079 * np.log(3) - 2284) / 32,
            np.sqrt(13 / 14) * 3 * ((143955 / 16) * np.log(3) - 118613 / 12) / 16,
        ],
        [
            np.sqrt(7) * (135 * np.log(3) - 148) / 16,
            7 * (135 * np.log(3) - 148) / 8,
            np.sqrt(21) * (3645 * np.log(3) - 4004) / 32,
            np.sqrt(11) * 5 * (2079 * np.log(3) - 2284) / 16,
            np.sqrt(13 / 2) * 3 * ((143955 / 16) * np.log(3) - 118613 / 12) / 8,
        ],
        [
            np.sqrt(3) * (3645 * np.log(3) - 4004) / 64,
            np.sqrt(21) * (3645 * np.log(3) - 4004) / 32,
            81 * (3645 * np.log(3) - 4004) / 128,
            np.sqrt(33 / 7) * 135 * (2079 * np.log(3) - 2284) / 64,
            np.sqrt(39 / 14) * 81 * ((143955 / 16) * np.log(3) - 118613 / 12) / 32,
        ],
        [
            np.sqrt(11 / 7) * 5 * (2079 * np.log(3) - 2284) / 32,
            np.sqrt(11) * 5 * (2079 * np.log(3) - 2284) / 16,
            np.sqrt(33 / 7) * 135 * (2079 * np.log(3) - 2284) / 64,
            605 * (2079 * np.log(3) - 2284) / 32,
            np.sqrt(143 / 2) * 33 * ((143955 / 16) * np.log(3) - 118613 / 12) / 16,
        ],
        [
            np.sqrt(13 / 14) * 3 * ((143955 / 16) * np.log(3) - 118613 / 12) / 16,
            np.sqrt(13 / 2) * 3 * ((143955 / 16) * np.log(3) - 118613 / 12) / 8,
            np.sqrt(39 / 14) * 81 * ((143955 / 16) * np.log(3) - 118613 / 12) / 32,
            np.sqrt(143 / 2) * 33 * ((143955 / 16) * np.log(3) - 118613 / 12) / 16,
            5941 * (431865 * np.log(3) - 474452) / 4096,
        ],
    ]
)


def potential_matrix_at_one_width_wont_overflow(interval):
    return unscaled_potential_matrix_at_one_width.max() / np.finfo(
        np.float64
    ).max <= abs(interval[1] - interval[0])


@given(
    interval=st.tuples(
        st.floats(allow_infinity=False, allow_nan=False),
        st.floats(allow_infinity=False, allow_nan=False),
    )
    .map(sorted)
    .filter(potential_matrix_at_one_width_wont_overflow)
)
def test_potential_matrix_at_one_width_to_right(interval):
    center = sum(interval) / 2
    half_width = center - interval[0]
    assume(not np.isinf(center))
    assume(not np.isinf(half_width))
    assume(half_width > 0)
    assume(not np.isinf(center + 2 * half_width))
    domain = FinDomain(0, interval[0], interval[1], 1, 5, None)
    matrix = fin_potential(domain, center + 2 * half_width)
    # The relative error seems to be quite high here at the moment, maybe we
    # can do something about that later?
    assert np.isclose(
        matrix,
        0.5 * unscaled_potential_matrix_at_one_width / half_width,
        atol=0,
        rtol=1e-8,
    ).all()


@given(
    interval=st.tuples(
        st.floats(allow_infinity=False, allow_nan=False),
        st.floats(allow_infinity=False, allow_nan=False),
    )
    .map(sorted)
    .filter(potential_matrix_at_one_width_wont_overflow)
)
def test_potential_matrix_at_one_width_to_left(interval):
    center = sum(interval) / 2
    half_width = center - interval[0]
    assume(not np.isinf(center))
    assume(not np.isinf(half_width))
    assume(half_width > 0)
    assume(not np.isinf(center - 2 * half_width))
    domain = FinDomain(0, interval[0], interval[1], 1, 5, None)
    matrix = fin_potential(domain, center - 2 * half_width)
    assert np.isclose(
        matrix,
        0.5 * left_side_sign_inversions
        * unscaled_potential_matrix_at_one_width
        / half_width,
        atol=0,
        rtol=1e-8,
    ).all()
