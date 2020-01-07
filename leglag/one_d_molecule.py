from operator import itemgetter
from itertools import combinations
import numpy as np
from numpy import dot, trace
from numpy.linalg import eigh
import leglag.moller_plesset
from leglag.dft import dft_domain_energy
from leglag.one_d_domain import InfDomain, FinDomain


class OneDMolecule:
    """Describes a 1D molecule according to the model first described by Loos
    et. al. in 2015."""

    def __init__(self, nuc_spec, dom_spec):
        """Initialises a 1D molecule.

        Arguments:
            nuc_spec    (list)  specifies information on the molecule's nuclei
                Each entry consists of tuple with the position of a nucleus
                followed by its charge
            dom_spec    (list)  specifies the molecule's electron domains
                Each entry contains the arguments for the constructor of the
                appropriate OneDDomain object (except the reference to the
                parent molecule). Note that the index number must be ordered
                positionally from -\infty to +\infty, and begin from index 1
        """

        # Setup up domains in the molecule
        self.nuclei = sorted(nuc_spec, key=itemgetter(0))
        self.n_nuclei = len(nuc_spec)
        dom_spec = sorted(dom_spec, key=itemgetter(0))
        self.domains = (
            [InfDomain( *(dom_spec[0] + (self, )) )] +
            [FinDomain( *(d + (self,)) ) for d in dom_spec[1:-1]]+
            [InfDomain( *(dom_spec[-1] + (self,)) )])
        self.non_empty_domains = []
        for d in self.domains:
            if d.electrons > 0 and d.functions > 0:
                self.non_empty_domains.append(d)

        # Set the options for computation
        self.diis_length = 4
        self.thresh = 1.e-10
        self.quadrature_start = 1
        self.quadrature_check = True

        # Initialise anything else
        self._hf_complete = False
        self._hf_fns = None

    @staticmethod
    def orbital_guess(domain, perturb=False):
        """Return an initial guess for the molecular orbitals.

        Optional:
            perturb (bool)  set to perturb the orbital guess
        """
        w, v = eigh(domain.kinetic_matrix + domain.potential_matrix)
        ordering = w.argsort()
        if not perturb:
            return v[:, ordering]
        else:
            rotation = np.identity(domain.functions, np.double)
            rotation[domain.electrons, domain.electrons] = np.cos(0.1)
            rotation[domain.electrons - 1, domain.electrons - 1] = np.cos(0.1)
            rotation[domain.electrons - 1, domain.electrons] = np.sin(0.1)
            rotation[domain.electrons, domain.electrons - 1] = - np.sin(0.1)

            return v[:, ordering].dot(rotation)
            return rotation.dot(v[:, ordering].dot(rotation.T))

    def run_hartree_fock(self, fns=None):
        """Run a SCF procedure to evaluate the Hartree-Fock orbitals.

        Optional:
            fns (int)   maximum number of basis functions to utilise in each
                domain
        """

        try:
            if self._hf_fns != [n for n in fns]:
                for d in self.domains:
                    d.diis_fock.clear()
                    d.diis_error.clear()
        except TypeError:
            if fns == None:
                if self._hf_fns != [d.functions for d in self.domains]:
                    for d in self.domains:
                        d.diis_fock.clear()
                        d.diis_error.clear()
            else:
                if self._hf_fns != [fns for d in self.domains]:
                    for d in self.domains:
                        d.diis_fock.clear()
                        d.diis_error.clear()

        for d in self.domains:
            d.convergence_history = []

        cycle = 0
        converged = False

        try:
            while not converged:
                cycle += 1
                if cycle > 200:
                    tmp = max(d.convergence
                            if d.electrons > 0 else 0 for d in self.domains)
                    break
                elif cycle > 2:
                    tmp = max(d.convergence
                            if d.electrons > 0 else 0 for d in self.domains)
                density_vectors = [dom.density_vector
                                if dom.functions > 0 else None
                                for dom in self.domains]
                converged = [dom.scf_cycle(density_vectors, fns=f)
                            if dom.functions > 0 else True
                            for dom, f in zip(self.domains, fns)]
                converged = not False in converged

        except TypeError:
            while not converged:
                cycle += 1
                if cycle > 200:
                    tmp = max(d.convergence
                            if d.electrons > 0 else 0 for d in self.domains)
                    break
                elif cycle > 2:
                    tmp = max(d.convergence
                            if d.electrons > 0 else 0 for d in self.domains)
                density_vectors = [dom.density_vector
                                if dom.functions > 0 else None
                                for dom in self.domains]
                converged = [dom.scf_cycle(density_vectors, fns=fns)
                            if dom.functions > 0 else True
                            for dom in self.domains]
                converged = not False in converged

        self._hf_complete = True
        try:
            self._hf_fns = [n for n in fns]
        except TypeError:
            if fns != None:
                self._hf_fns = [min(fns, d.functions) for d in self.domains]
            else:
                self._hf_fns = [d.functions for d in self.domains]

    def hf_energy(self, component='full', fns=None):
        """Return an expectation value of the HF wavefunction.

        Optional:
            component   (str)   declare the expectation value to evaluate
                'full'  (default)   total molecular energy
                'kinetic'           electronic kinetic energy
                'potential'         nuclear attraction energy
                'repulsion'         electronic repulsion energy
            fns         (int)   maximum number of basis functions to utilise
                in each domain
        """
        if not self._hf_complete or (self._hf_fns != fns and fns != None):
            self.run_hartree_fock(fns=fns)

        try:
            if self._hf_fns != [n for n in fns]:
                self.run_hartree_fock(fns=fns)
        except TypeError:
            if fns == None:
                if self._hf_fns != [d.functions for d in self.domains]:
                    self.run_hartree_fock()
            else:
                if self._hf_fns != [min(fns, d.functions)
                        for d in self.domains]:
                    self.run_hartree_fock()

        if component == 'full':
            energy = sum(np.sum(trace(dot(d.kinetic_matrix +
                            d.potential_matrix + d.diis_fock[-1],
                            d.density_matrix)))
                        for d in self.non_empty_domains)
            energy = energy / 2 + sum(n[0][1] * n[1][1] /
                        abs(n[0][0] - n[1][0])
                        for n in combinations(self.nuclei, 2))
        if component == 'kinetic':
            energy = sum(np.sum(trace(dot(d.kinetic_matrix, d.density_matrix)))
                         for d in self.non_empty_domains)
        if component == 'attraction':
            energy = sum(np.sum(trace(dot(d.potential_matrix,
                            d.density_matrix)))
                         for d in self.non_empty_domains)
        if component == 'repulsion':
            energy = sum(np.sum(trace(dot(d.diis_fock[-1] - d.potential_matrix -
                                          d.kinetic_matrix, d.density_matrix)))
                         for d in self.non_empty_domains)
            energy /= 2
        return energy

    def mp2_correction(self):
        """Return the MP2 correlation estimate for the whole molecule."""
        if not self._hf_complete: self.run_hartree_fock()
        return leglag.moller_plesset.mp2_correction(self, self._hf_fns)

    def mp3_correction(self):
        """Return the MP3 correlation correction for the whole molecule.

        Add mp2_correction for the full MP3 correlation energy"""
        if not self._hf_complete: self.run_hartree_fock()
        return leglag.moller_plesset.mp3_correction(self, self._hf_fns)

    def dft_correction(self, functional="glda", parameters=None, fit=None):
        """Return a post-HF DFT correlation estimate for the whole molecule.

        Optional:
            functional  (str)   correlation functional to utilise
                'lda'
                'glda'  (default)
                '0lda'
                'alphalda'
                'sblda'
                'gsblda'
            parameters  ()      additional parameters for specifying some
                functionals
            fit         (str)   nominate a specific fit of the declared
                functional type
            """
        if not self._hf_complete: self.run_hartree_fock()
        if functional != "lda" and functional != 'sblda':
            return sum(dft_domain_energy(d, functional,
                        parameters=parameters, fit=fit)
                    for d in self.domains if d.electrons > 1)
        else:
            return sum(dft_domain_energy(d, functional,
                        parameters=parameters, fit=fit)
                    for d in self.domains if d.electrons > 0)

    def dispersion_estimate(self):
        """Return an estimate of the dispersion interactions between electrons
        in separate domains across the whole molecule."""
        return sum(d1.dispersion_estimate(d2)
                for d1, d2 in combinations(self.domains, 2))

