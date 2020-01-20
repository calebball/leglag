from collections import deque, defaultdict
from itertools import combinations
from functools import singledispatch
from pkg_resources import resource_filename
import os
import sys
import numpy as np
from numpy import arange, dot, rollaxis
from numpy.linalg import eigh, solve
from scipy.special import eval_genlaguerre, lpmn
import leglag.one_e_integrals
import leglag.two_e_integrals
import leglag.moller_plesset
import leglag.dispersion
from leglag.dft import dft_domain_energy


class OneDDomain:
    """Describes an electron domain within a 1D molecule.
    Abstract class. Use one of the subclasses.
    """

    def __init__(self, num, position, electrons, functions, molecule):
        """Initialises a 1D electron domain.

        Arguments:
            num         (int)   index number in parent molecule
            position    ()  describes interval the domain constitutes
            electrons   (int)   electrons within the domain
            functions   (int)   number of basis functions to describe the domain
            molecule    (OneDMolecule)  reference to parent molecule
        """

        self.num = num
        self.position = position
        self.electrons = electrons
        self.functions = functions if electrons > 0 else 0
        self.molecule = molecule

        self.quad_level = defaultdict(int)

        self.convergence_history = []

    # ==== INTEGRAL STORAGE ====#
    @property
    def kinetic_matrix(self):
        """Retrieve the kinetic energy submatrix for the domain."""
        try:
            return self._kinetic_matrix
        except AttributeError:
            self._kinetic_matrix = build_kinetic_matrix(self)
            return self._kinetic_matrix

    @property
    def mo_kinetic_matrix(self):
        """Retrieve the kinetic energy submatrix for the domain in the MO
        basis."""
        return np.transpose(self.orbital_coefficients).dot(
            self.kinetic_matrix.dot(self.orbital_coefficients)
        )

    @property
    def potential_matrix(self):
        """Retrieve the potential energy submatrix for the domain."""
        try:
            return self._potential_matrix
        except AttributeError:
            self._potential_matrix = build_potential_matrix(self)
            return self._potential_matrix

    @property
    def mo_potential_matrix(self):
        """Retrieve the potential energy submatrix for the domain in the MO
        basis."""
        return np.transpose(self.orbital_coefficients).dot(
            self.potential_matrix.dot(self.orbital_coefficients)
        )

    @property
    def field_matrix(self):
        """Retrieve the submatrix for describing the interaction with an
        external field in the domain."""
        try:
            return self._field_matrix
        except AttributeError:
            self._field_matrix = build_field_matrix(self)
            return self._field_matrix

    @property
    def mo_field_matrix(self):
        """Retrieve the submatrix for describing the interaction with an
        external field in the domain, in the MO basis."""
        return np.transpose(self.orbital_coefficients).dot(
            self.field_matrix.dot(self.orbital_coefficients)
        )

    @property
    def eri_array(self):
        """Retrieve an array of the 2 electron integral subtensors. Each
        element is a subtensor for an electron in this domain paired with one
        in another domain of the parent molecule."""
        try:
            return self._eri_array
        except AttributeError:
            if self.functions > 0:
                self._eri_array = [
                    self.build_eri_array(domain_two)
                    if domain_two.functions > 0
                    else None
                    for domain_two in self.molecule.domains
                ]
            else:
                self._eri_array = [None for domain_two in self.molecule.domains]
            return self._eri_array

    @property
    def double_bar_array(self):
        """Retrieve an array of the 2 electron double-bar integral subtensors.
        Each element is a subtensor for an electron in this domain paired with one
        in another domain of the parent molecule."""
        return [
            eris - np.swapaxes(eris, 1, 3)
            if i == self.molecule.domains.index(self)
            else eris
            for i, eris in enumerate(self.eri_array)
        ]

    @property
    def mo_eri_array(self):
        """Retrieve an array of the 2 electron integral subtensors in the MO
        basis. Each element is a subtensor for an electron in this domain
        paired with one in another domain of the parent molecule."""
        try:
            if self._mo_eri_array_out_of_date:
                if self.functions > 0:
                    self._mo_eri_array = [
                        rollaxis(
                            dot(
                                rollaxis(
                                    dot(
                                        rollaxis(
                                            dot(
                                                rollaxis(
                                                    dot(
                                                        self.eri_array[i],
                                                        domain_two.orbital_coefficients,
                                                    ),
                                                    3,
                                                ),
                                                domain_two.orbital_coefficients,
                                            ),
                                            3,
                                        ),
                                        self.orbital_coefficients,
                                    ),
                                    3,
                                ),
                                self.orbital_coefficients,
                            ),
                            3,
                        )
                        if domain_two.functions > 0
                        else None
                        for i, domain_two in enumerate(self.molecule.domains)
                    ]

                else:
                    self._mo_eri_array = [None for domain_two in self.molecule.domains]

            return self._mo_eri_array
        except AttributeError:
            if self.functions > 0:
                self._mo_eri_array = [
                    rollaxis(
                        dot(
                            rollaxis(
                                dot(
                                    rollaxis(
                                        dot(
                                            rollaxis(
                                                dot(
                                                    self.eri_array[i],
                                                    domain_two.orbital_coefficients,
                                                ),
                                                3,
                                            ),
                                            domain_two.orbital_coefficients,
                                        ),
                                        3,
                                    ),
                                    self.orbital_coefficients,
                                ),
                                3,
                            ),
                            self.orbital_coefficients,
                        ),
                        3,
                    )
                    if domain_two.functions > 0
                    else None
                    for i, domain_two in enumerate(self.molecule.domains)
                ]

            else:
                self._mo_eri_array = [None for domain_two in self.molecule.domains]

            return self._mo_eri_array

    @property
    def mo_double_bar_array(self):
        """Retrieve an array of the 2 electron double-bar integral subtensors
        in the MO basis. Each element is a subtensor for an electron in this
        domain paired with one in another domain of the parent molecule."""
        return [
            eris - np.swapaxes(eris, 1, 3)
            if i == self.molecule.domains.index(self)
            else eris
            for i, eris in enumerate(self.mo_eri_array)
        ]

    # ==== SCF PROPERTIES ====#
    @property
    def orbital_coefficients(self):
        """Retrieve the current orbital coefficient matrix."""
        try:
            return self._orbital_coefficients
        except AttributeError:
            self._density_out_of_date = True
            self._mo_eri_array_out_ouf_date = True
            self._orbital_coefficients = self.molecule.orbital_guess(self)
            return self._orbital_coefficients

    @orbital_coefficients.setter
    def orbital_coefficients(self, value):
        self._density_out_of_date = True
        self._mo_eri_array_out_ouf_date = True
        self._orbital_coefficients = value

    @property
    def density_matrix(self, det=None):
        """Retrieve the current electron density matrix."""
        try:
            if self._density_out_of_date:
                self._density_matrix = dot(
                    self.orbital_coefficients[:, : self.electrons],
                    self.orbital_coefficients.T[: self.electrons, :],
                )
                self._density_out_of_date = False
                return self._density_matrix
            else:
                return self._density_matrix
        except AttributeError:
            self._density_matrix = dot(
                self.orbital_coefficients[:, : self.electrons],
                self.orbital_coefficients.T[: self.electrons, :],
            )
            self._density_out_of_date = False
            return self._density_matrix

    @property
    def density_vector(self):
        """Retrieve the current electron density matrix stacked into a single
        vector."""
        return self.density_matrix.ravel()

    # ==== UTILITIES ====#
    def rho(self, x):
        """Returns the electronic density at a point."""
        bf = self.bf_vector(x)
        return bf.dot(self.density_matrix.dot(bf))

    def weizsaecker(self, x):
        """Returns the Weizsaecker kinetic density at a point."""
        bf = self.bf_vector(x)
        bfp = self.bf_derivative_vector(x)
        return (
            bfp.dot(self.density_matrix.dot(bf)) + bf.dot(self.density_matrix.dot(bfp))
        ) ** 2 / (2 * self.rho(x))

    def tau(self, x):
        """Returns the kinetic energy density at a point."""
        bfp = self.bf_derivative_vector(x)
        return 2 * sum(
            bfp.dot(self.orbital_coefficients[:, k]) ** 2 for k in range(self.electrons)
        )

    def eta(self, x, return_rho=False):
        """Returns the electronic curvature parameter at a point."""
        if self.electrons <= 1:
            return (0, 0) if return_rho else 0
        bf = self.bf_vector(x)
        rho = bf.dot(self.density_matrix.dot(bf))
        if rho < 1e-100:
            return (rho, np.inf) if return_rho else np.inf
        bfp = self.bf_derivative_vector(x)

        tau = 2 * sum(
            bfp.dot(self.orbital_coefficients[:, k]) ** 2 for k in range(self.electrons)
        )

        weizsaecker = (
            bfp.dot(self.density_matrix.dot(bf)) + bf.dot(self.density_matrix.dot(bfp))
        ) ** 2 / (2 * rho)

        if abs(tau - weizsaecker) / abs(tau) < 1e-14:
            return (rho, 0) if return_rho else 0
        if tau - weizsaecker < 0:
            return (rho, 0) if return_rho else 0
        else:
            normalisation = 3 / (2 * np.pi ** 2 * rho ** 3)
            return (
                rho,
                (normalisation * (tau - weizsaecker))
                if return_rho
                else normalisation * (tau - weiasaecker),
            )

    # ==== SCF METHODS ====#
    @property
    def diis_fock(self):
        """Retrieve the current history of Fock matrices used in the DIIS
        acceleration algorithm.

        Stored in a Deque object."""
        try:
            return self._diis_fock
        except AttributeError:
            self._diis_fock = deque([], self.molecule.diis_length)
            return self._diis_fock

    @property
    def diis_error(self):
        """Retrieve the current history of DIIS error matrices.

        Stored in a Deque object."""
        try:
            return self._diis_error
        except AttributeError:
            self._diis_error = deque([], self.molecule.diis_length)
            return self._diis_error

    @property
    def convergence(self):
        """Return the value of the current SCF convergence measure."""
        return np.sqrt(sum(i ** 2 for i in self.diis_error[-1].flatten()))

    def scf_cycle(self, p_array, fns=None):
        """Perform a single SCF cycle for evaluating the HF energy.

        Arguments:
            p_array (list)  list of the current density matrices for each
                domain in the parent molecule
        Optional:
            fns     (int)   number of basis functions to use in the
                calculation.
        """
        if fns == None or fns > self.functions:
            fns = self.functions

        # Store the current fock matrix in the DIIS queue
        self.diis_fock.append(
            self.kinetic_matrix
            + self.potential_matrix
            + sum(
                dot(x[0].reshape(-1, len(x[1])), x[1])
                if type(x[0]) == np.ndarray and type(x[1]) == np.ndarray
                else 0
                for x in zip(self.double_bar_array, p_array)
            ).reshape(self.functions, self.functions)
        )
        self.diis_error.append(
            dot(self.diis_fock[-1][:fns, :fns], self.density_matrix[:fns, :fns])
            - dot(self.density_matrix[:fns, :fns], self.diis_fock[-1][:fns, :fns])
        )

        # Solve the DIIS equation
        diis_mat = np.fromiter(
            (np.trace(dot(i, j)) for i in self.diis_error for j in self.diis_error),
            np.double,
        )
        diis_mat.resize(len(self.diis_error), len(self.diis_error))
        diis_mat = np.vstack((diis_mat, -np.ones(len(self.diis_error), np.double)))
        diis_mat = np.hstack(
            (
                diis_mat,
                ((arange(len(self.diis_error) + 1) // len(self.diis_error)) - 1)[
                    :, None
                ],
            )
        )
        try:
            diis_vec = solve(
                diis_mat, -(arange(len(self.diis_error) + 1) // len(self.diis_error))
            )
        except np.linalg.linalg.LinAlgError as exception:
            self.diis_fock.clear()
            self.diis_error.clear()
            return self.scf_cycle(p_array, fns=fns)

        # Diagonalise the DIIS Fock matrix
        w, v = eigh(sum(x[0] * x[1][:fns, :fns] for x in zip(diis_vec, self.diis_fock)))
        ordering = w.argsort()
        self.orbital_energies = np.pad(
            w[ordering], (0, self.functions - fns), "constant", constant_values=(0, 0)
        )
        tmp = np.zeros((self.functions, self.functions))
        tmp[:fns, :fns] = v[:, ordering]
        self.orbital_coefficients = tmp

        self.convergence_history.append(self.convergence)
        return self.convergence < self.molecule.thresh

    # ==== MOLLER-PLESSET ====#
    def mp2_in_domain(self):
        """Compute the MP2 correlation of the electrons within this domain."""
        return leglag.moller_plesset.mp2_domain_correction(self)

    def mp3_in_domain(self):
        """Compute the MP3 correlation of the electrons within this domain."""
        return leglag.moller_plesset.mp3_domain_correction(self)

    # ==== DENSITY FUNCTIONALS ====#
    def dft_in_domain(self, functional="glda"):
        """Compute the DFT correlation of the electrons within this domain."""
        return dft_domain_energy(self, functional)

    # ==== ABSTRACTS ====#
    def build_eri_array(self, domain_two):
        pass

    def bf_vector(self, x):
        pass

    def bf_derivative_vector(self, x):
        pass

    def dispersion_estimate(self, domain_two):
        pass

    def ele_com(self, i):
        pass


class InfDomain(OneDDomain):
    """Describes an electron domain within occupying a semi-infinite interval
    in a 1D molecule."""

    def __init__(self, num, A, right, alpha, electrons, functions, molecule):
        """Initialises a semi-infinite 1D electron domain.

        Arguments:
            num         (int)   index number in parent molecule
            A           (float) position of the bordering nucleus
            right       (bool)  true if bordered by +\infty
            alpha       (float) exponent parameter of the basis set
            electrons   (int)   electrons within the domain
            functions   (int)   number of basis functions to describe the domain
            molecule    (OneDMolecule)  reference to parent molecule
        """

        super().__init__(num, A, electrons, functions, molecule)
        self.side = right
        self.alpha = alpha

        self.s_star = np.zeros((functions, functions), dtype=np.double)

        with open(
            resource_filename("leglag", "integral_data/LL_bound.dat")
        ) as bound_file:
            for line in bound_file:
                (m, n, value) = line.split()
                m = int(m) - 1
                n = int(n) - 1
                if max(m, n) + 1 > functions:
                    break
                self.s_star[m, n] = value
                self.s_star[n, m] = value

        self.s_star_p = np.zeros((functions, functions), dtype=np.double)
        with open(
            resource_filename("leglag", "integral_data/LL_p_bound.dat")
        ) as bound_file:
            for line in bound_file:
                (m, n, value) = line.split()
                m = int(m) - 1
                n = int(n) - 1
                if max(m, n) + 1 > functions:
                    break
                self.s_star_p[m, n] = value
                self.s_star_p[n, m] = value

    def build_eri_array(self, domain_two):
        """Computes the two electron integral tensor for an electron in this
        domain and another domain.

        Arguments:
            domain_two  (OneDDomain)    domain containing the second electron
        """
        if isinstance(domain_two, FinDomain):
            check = self.molecule.quadrature_check
            if not check:
                start = self.molecule.quadrature_start
            elif self.quad_level[domain_two]:
                start = self.quad_level[domain_two]
                check = False
            else:
                tmp = sorted(
                    val
                    for d in self.molecule.domains
                    for val in d.quad_level.values()
                    if val != 0
                )
                median = tmp[len(tmp) // 2] if len(tmp) > 5 else 0
                start = max(
                    domain_two.quad_level[self], self.molecule.quadrature_start, median
                )

            result, quad = leglag.two_e_integrals.true_inf_fin(
                self, domain_two, quad_check=check, quad_start=start
            )
            self.quad_level[domain_two] = quad

            return result
        elif isinstance(domain_two, InfDomain):
            return (
                leglag.two_e_integrals.true_inf_inf(self, domain_two)
                if self != domain_two
                else leglag.two_e_integrals.quasi_inf_inf(self)
            )

    def dispersion_estimate(self, domain_two):
        """Calculate an estimate of the dispersion energy resulting from
        electrons in this domain interacting with those in another.

        Arguments:
            domain_two  (OneDDomain)    domain containing the second electron
        """
        if isinstance(domain_two, InfDomain):
            return leglag.dispersion.inf_inf_dispersion(self, domain_two)
        elif isinstance(domain_two, FinDomain):
            return leglag.dispersion.inf_fin_dispersion(self, domain_two)

    def bf_vector(self, x):
        """Return a vector containing the values of each basis function at a
        point."""
        a = self.alpha
        z = abs(self.position - x)

        c = np.sqrt(8 * a ** 3) * z * np.exp(-a * z)

        return np.fromiter(
            (
                eval_genlaguerre(mu - 1, 2, 2 * a * z) * c / np.sqrt(mu * (mu + 1))
                for mu in range(1, self.functions + 1)
            ),
            dtype=np.double,
        )

    def bf_derivative_vector(self, x):
        """Return a vector containing the values of the first derivative of
        each basis function at a point."""
        a = self.alpha
        z = abs(self.position - x)
        if self.side:
            return np.fromiter(
                (
                    np.sqrt(8 * a ** 3 / (mu * (mu + 1)))
                    * np.exp(-a * z)
                    * (
                        2 * a * z * eval_genlaguerre(mu - 2, 3, 2 * a * z)
                        - (a * z - 1) * eval_genlaguerre(mu - 1, 2, 2 * a * z)
                    )
                    for mu in range(1, self.functions + 1)
                ),
                dtype=np.double,
            )
        else:
            return np.fromiter(
                (
                    np.sqrt(8 * a ** 3 / (mu * (mu + 1)))
                    * np.exp(-a * z)
                    * (
                        2 * a * z * eval_genlaguerre(mu - 2, 3, 2 * a * z)
                        + (a * z - 1) * eval_genlaguerre(mu - 1, 2, 2 * a * z)
                    )
                    for mu in range(1, self.functions + 1)
                ),
                dtype=np.double,
            )

    def potential(self, x):
        """Return the potential at a point."""
        return leglag.one_e_integrals.inf_potential(self, x)

    def ele_com(self, i):
        """Return the center of mass of one of the current orbitals in the
        domain."""
        one_e_density = dot(
            self.orbital_coefficients[:, i - 1 : i],
            self.orbital_coefficients.T[i - 1 : i, :],
        )

        if self.side:
            return self.position + np.trace(dot(self.field_matrix, one_e_density))
        else:
            return self.position - np.trace(dot(self.field_matrix, one_e_density))


class FinDomain(OneDDomain):
    """Describes an electron domain within occupying a finite interval in a 1D
    molecule."""

    def __init__(self, num, A, B, electrons, functions, molecule):
        """Initialises a finite 1D electron domain.

        Arguments:
            num         (int)   index number in parent molecule
            A           (float) position of the left bordering nucleus
            B           (float) position of the right bordering nucleus
            electrons   (int)   electrons within the domain
            functions   (int)   number of basis functions to describe the domain
            molecule    (OneDMolecule)  reference to parent molecule
        """

        super().__init__(num, (max(A, B), min(A, B)), electrons, functions, molecule)
        self.halfwidth = abs(B - A) / 2

        self.s_star = np.zeros((functions, functions), dtype=np.double)
        with open(
            resource_filename("leglag", "integral_data/PP_bound.dat")
        ) as bound_file:
            for line in bound_file:
                (m, n, value) = line.split()
                m = int(m) - 1
                n = int(n) - 1
                if max(m, n) + 1 > functions:
                    break
                self.s_star[m, n] = value
                self.s_star[n, m] = value

        self.s_star_p = np.zeros((functions, functions), dtype=np.double)
        with open(
            resource_filename("leglag", "integral_data/PP_p_bound.dat")
        ) as bound_file:
            for line in bound_file:
                (m, n, value) = line.split()
                m = int(m) - 1
                n = int(n) - 1
                if max(m, n) + 1 > functions:
                    break
                self.s_star_p[m, n] = value
                self.s_star_p[n, m] = value

        self.s_star_n = np.zeros((functions, functions), dtype=np.double)
        with open(
            resource_filename("leglag", "integral_data/PP_n_bound.dat")
        ) as bound_file:
            for line in bound_file:
                (m, n, value) = line.split()
                m = int(m) - 1
                n = int(n) - 1
                if max(m, n) + 1 > functions:
                    break
                self.s_star_n[m, n] = value
                self.s_star_n[n, m] = value

    def build_eri_array(self, domain_two):
        """Computes the two electron integral tensor for an electron in this
        domain and another domain.

        Arguments:
            domain_two  (OneDDomain)    domain containing the second electron
        """
        if isinstance(domain_two, FinDomain):
            if self == domain_two:
                return leglag.two_e_integrals.quasi_fin_fin(self)
            else:
                check = self.molecule.quadrature_check
                if not check:
                    start = self.molecule.quadrature_start
                elif self.quad_level[domain_two]:
                    start = self.quad_level[domain_two]
                    check = False
                else:
                    tmp = sorted(
                        val
                        for d in self.molecule.domains
                        for val in d.quad_level.values()
                        if val != 0
                    )
                    median = tmp[len(tmp) // 2] if len(tmp) > 5 else 0
                    start = max(
                        domain_two.quad_level[self],
                        self.molecule.quadrature_start,
                        median,
                    )

                result, quad = leglag.two_e_integrals.true_fin_fin(
                    self, domain_two, quad_check=check, quad_start=start
                )
                self.quad_level[domain_two] = quad

                return result
        elif isinstance(domain_two, InfDomain):
            check = self.molecule.quadrature_check
            if not check:
                start = self.molecule.quadrature_start
            elif self.quad_level[domain_two]:
                start = self.quad_level[domain_two]
                check = False
            else:
                tmp = sorted(
                    val
                    for d in self.molecule.domains
                    for val in d.quad_level.values()
                    if val != 0
                )
                median = tmp[len(tmp) // 2] if len(tmp) > 5 else 0
                start = max(
                    domain_two.quad_level[self], self.molecule.quadrature_start, median
                )

            result, quad = leglag.two_e_integrals.true_inf_fin(
                domain_two, self, quad_check=check, quad_start=start
            )
            self.quad_level[domain_two] = quad

            return np.transpose(result, (2, 3, 0, 1))

    def dispersion_estimate(self, domain_two):
        """Calculate an estimate of the dispersion energy resulting from
        electrons in this domain interacting with those in another.

        Arguments:
            domain_two  (OneDDomain)    domain containing the second electron
        """
        if isinstance(domain_two, InfDomain):
            return leglag.dispersion.inf_fin_dispersion(domain_two, self)
        elif isinstance(domain_two, FinDomain):
            return leglag.dispersion.fin_fin_dispersion(self, domain_two)

    def bf_vector(self, x):
        """Return a vector containing the values of each basis function at a
        point."""
        c = sum(self.position) / 2
        w = abs(self.position[0] - self.position[1]) / 2
        normalisation = np.fromiter(
            (
                np.sqrt((mu + 1.5) / (w * mu * (mu + 1) * (mu + 2) * (mu + 3)))
                for mu in range(1, self.functions + 1)
            ),
            dtype=np.double,
        )
        legendre = lpmn(2, self.functions + 1, (x - c) / w)[0][-1, 2:]
        return normalisation * legendre

    def bf_derivative_vector(self, x):
        """Return a vector containing the values of the first derivative of
        each basis function at a point."""
        c = sum(self.position) / 2
        w = abs(self.position[0] - self.position[1]) / 2
        normalisation = np.fromiter(
            (
                np.sqrt((mu + 1.5) / (w * mu * (mu + 1) * (mu + 2) * (mu + 3)))
                for mu in range(1, self.functions + 1)
            ),
            dtype=np.double,
        )
        legendre = lpmn(2, self.functions + 1, (x - c) / w)[1][-1, 2:] / w
        return normalisation * legendre

    def potential(self, x):
        """Return the potential at a point."""
        return leglag.one_e_integrals.fin_potential(self, x)

    def ele_com(self, i):
        """Return the center of mass of one of the current orbitals in the
        domain."""
        one_e_density = dot(
            self.orbital_coefficients[:, i - 1 : i],
            self.orbital_coefficients.T[i - 1 : i, :],
        )

        return (
            self.position[1]
            + self.halfwidth
            + np.trace(dot(self.field_matrix, one_e_density))
        )


# ====== DISPATCH FUNCTIONS ====================================================#


@singledispatch
def build_kinetic_matrix(domain):
    pass


build_kinetic_matrix.register(InfDomain, leglag.one_e_integrals.inf_kinetic)
build_kinetic_matrix.register(FinDomain, leglag.one_e_integrals.fin_kinetic)


@singledispatch
def build_potential_matrix(domain):
    pass


build_potential_matrix.register(
    InfDomain, leglag.one_e_integrals.inf_nuclear_attraction
)
build_potential_matrix.register(
    FinDomain, leglag.one_e_integrals.fin_nuclear_attraction
)


@singledispatch
def build_field_matrix(domain):
    pass


build_field_matrix.register(InfDomain, leglag.one_e_integrals.inf_field)
build_field_matrix.register(FinDomain, leglag.one_e_integrals.fin_field)


# ==============================================================================#
