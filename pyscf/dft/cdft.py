#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Finite molecular UKS with one Hirshfeld population constraint.

The physical functional is minimized at fixed spin electron counts and
Tr[D W] = target. The propagated energy excludes the constraint potential.
Analytic forces use the stationary Lagrangian, including moving atomic
reference densities, basis functions and quadrature. Atomic units throughout.
No density fitting, ECP, NLC, periodicity, symmetry or spin constraints.
'''

from dataclasses import dataclass
import hashlib

import numpy as np
from scipy.optimize import brentq
from pyscf import gto, lib
from pyscf.dft import numint, uks
from pyscf.grad import rks as rks_grad
from pyscf.scf import atom_hf


class ConvergenceError(RuntimeError):
    '''An inner SCF or the population constraint did not converge.'''


def atomic_references(mol):
    '''Neutral spherical atomic-HF densities in the molecular spherical basis.

    Returns a mapping from atom symbol to (isolated atom Mole, density).
    References are independent of molecular positions and the constrained state.
    '''
    if mol.cart or mol.has_ecp() or mol._pseudo:
        raise ValueError('Hirshfeld references require all-electron spherical AOs')
    references = {}
    for ia in range(mol.natm):
        symbol = mol.atom_symbol(ia)
        if symbol in references:
            continue
        charge = int(mol.atom_charge(ia))
        if charge <= 0:
            raise ValueError('Ghost atoms are not admitted in the constraint')
        atom = gto.M(atom=[(symbol, (0, 0, 0))], basis={symbol: mol._basis[symbol]},
                     spin=charge % 2, charge=0, verbose=0)
        method = atom_hf.AtomHF1e(atom) if charge == 1 else atom_hf.AtomSphAverageRHF(atom)
        method.conv_tol = 1e-12
        method.max_cycle = 100
        method.kernel()
        if not method.converged:
            raise ConvergenceError('The neutral spherical atomic reference did not converge')
        density = np.asarray(method.make_rdm1())
        if density.ndim == 3:
            density = density.sum(axis=0)
        references[symbol] = (atom, density)
    return references


class Hirshfeld:
    '''One fixed-target signed fragment population on moving molecular grids.

    atom_weights: real coefficients per QM atom, e.g. +1/-1 on two fragments.
    grids: unpruned PySCF molecular grids with full Becke response available.
    references: frozen neutral spherical atomic-HF densities from
        atomic_references(mol). They are not optimized in the molecular field.
    '''

    def __init__(self, mol, atom_weights, grids, references=None):
        self.mol = mol
        self.atom_weights = np.array(atom_weights, dtype=float, copy=True)
        if (self.atom_weights.shape != (mol.natm,) or
                not np.isfinite(self.atom_weights).all() or
                np.ptp(self.atom_weights) == 0):
            raise ValueError('A nonconstant finite weight per QM atom is required')
        if mol.cart or mol.has_ecp() or mol._pseudo or mol.symmetry:
            raise ValueError('Constraint requires finite all-electron spherical AOs without symmetry')
        if grids.mol is not mol or grids.prune is not None:
            raise ValueError('Constraint requires this molecule and an unpruned grid')
        self.grids = grids
        self.references = atomic_references(mol) if references is None else references
        self.matrix = self._matrix()

    @property
    def reference_sha256(self):
        h = hashlib.sha256()
        for symbol in sorted(self.references):
            atom, dm = self.references[symbol]
            h.update(symbol.encode())
            h.update(repr(atom._basis).encode())
            h.update(np.asarray(dm, dtype='<f8').tobytes())
        return h.hexdigest()

    def _blocks(self, deriv):
        mol = self.mol
        for owner, (coords, weights, weight1) in enumerate(
                rks_grad.grids_response_cc(self.grids)):
            for start, stop in lib.prange(0, len(coords), 1024):
                xyz = coords[start:stop]
                ao = numint.eval_ao(mol, xyz, deriv=deriv)
                pro = np.empty((mol.natm, len(xyz)))
                pro_grad = np.empty((mol.natm, 3, len(xyz))) if deriv else None
                for ia in range(mol.natm):
                    atom, dm = self.references[mol.atom_symbol(ia)]
                    pao = numint.eval_ao(atom, xyz-mol.atom_coord(ia), deriv=deriv)
                    p0 = pao[0] if deriv else pao
                    prodm = p0 @ dm
                    pro[ia] = np.einsum('gi,gi->g', prodm, p0)
                    if deriv:
                        pro_grad[ia] = 2*np.einsum('gi,xgi->xg', prodm, pao[1:4])
                denominator = pro.sum(axis=0)
                supported = denominator > 1e-200
                inverse = np.zeros_like(denominator)
                inverse[supported] = 1/denominator[supported]
                w = (self.atom_weights @ pro)*inverse
                yield (owner, ao, weights[start:stop],
                       weight1[:, :, start:stop], w, inverse, pro_grad)

    def _matrix(self):
        wmat = np.zeros((self.mol.nao, self.mol.nao))
        for _, ao, weight, _, w, _, _ in self._blocks(0):
            wmat += ao.T @ ((weight*w)[:, None]*ao)
        return (wmat+wmat.T)*.5

    def population(self, density):
        '''Spin-summed signed population in electrons.'''
        dm = np.asarray(density)
        if dm.ndim == 3:
            dm = dm.sum(axis=0)
        return float(np.einsum('ij,ji', dm, self.matrix))

    def density_gradient(self, density):
        '''Partial dC/dR at fixed AO density, including grid and AO movement.

        Returns (natom, 3) in electrons/Bohr. Orbital response is treated by
        the constrained Lagrangian and energy-weighted Pulay term, not here.
        '''
        dm = np.asarray(density)
        if dm.ndim == 3:
            dm = dm.sum(axis=0)
        gradient = np.zeros((self.mol.natm, 3))
        slices = self.mol.aoslice_by_atom()
        for owner, ao, weight, weight1, w, inverse, pro_grad in self._blocks(1):
            aodm = ao[0] @ dm
            rho = np.einsum('gi,gi->g', ao[0], aodm)
            drho_grid = 2*np.einsum('gi,xgi->xg', aodm, ao[1:4])
            # Nuclear movement of each proatom at fixed grid position.
            dw = -(self.atom_weights[:, None, None]-w[None, None, :])*pro_grad
            dw *= inverse[None, None, :]
            # Points in this block move with their owner atom.
            dw[owner] -= dw.sum(axis=0)
            gradient += np.einsum('axg,g->ax', weight1, w*rho)
            gradient += np.einsum('axg,g->ax', dw, weight*rho)
            gradient[owner] += np.einsum('xg,g->x', drho_grid, weight*w)
            for ia, (_, _, p0, p1) in enumerate(slices):
                drho_ao = -2*np.einsum('gi,xgi->xg', aodm[:, p0:p1],
                                       ao[1:4, :, p0:p1])
                gradient[ia] += np.einsum('xg,g->x', drho_ao, weight*w)
        return gradient


@dataclass
class Result:
    '''Converged constrained density and physical energy, in atomic units.'''
    method: object
    constraint: Hirshfeld
    target: float
    multiplier: float
    population: float
    energy: float
    iterations: list

    @property
    def residual(self):
        return self.population-self.target

    @property
    def residual_energy(self):
        return self.multiplier*self.residual

    @property
    def density(self):
        return self.method.make_rdm1()

    def gradients(self):
        '''Full QM energy derivative in Hartree/Bohr.'''
        grad = self.method.nuc_grad_method()
        grad.grid_response = True
        base = grad.kernel()
        return base+self.multiplier*self.constraint.density_gradient(self.density)

    def environment_gradients(self):
        '''Fixed point-charge source derivatives in Hartree/Bohr.

        The constraint depends only on QM nuclei. Stationarity removes density
        response; official qmmm electronic plus nuclear reaction forces remain.
        '''
        if not hasattr(self.method, 'mm_mol'):
            raise ValueError('No point-charge environment is present')
        grad = self.method.nuc_grad_method()
        return grad.grad_hcore_mm(self.density.sum(axis=0))+grad.grad_nuc_mm()

    def diagnostics(self):
        '''State, population, physical/optimizer energy and convergence data.'''
        overlap = self.method.get_ovlp()
        eig, vectors = np.linalg.eigh(overlap)
        if eig.min() <= 1e-8:
            raise ConvergenceError('AO overlap is outside the admitted independent-basis domain')
        root = (vectors*np.sqrt(eig)) @ vectors.T
        dm = self.density
        lowdin = np.diag(root @ dm.sum(axis=0) @ root)
        spin = np.diag(root @ (dm[0]-dm[1]) @ root)
        slices = self.method.mol.aoslice_by_atom()
        atomic_pop = np.array([lowdin[p0:p1].sum() for _, _, p0, p1 in slices])
        atomic_spin = np.array([spin[p0:p1].sum() for _, _, p0, p1 in slices])
        traces = np.einsum('sij,ji->s', dm, overlap)
        if not np.allclose(traces, self.method.mol.nelec, atol=1e-8, rtol=0):
            raise ConvergenceError('Constrained density violates the fixed electron counts')
        return dict(target=self.target, population=self.population,
                    density_electron_counts=traces.tolist(),
                    lowdin_atom_populations=atomic_pop.tolist(),
                    lowdin_atom_spins=atomic_spin.tolist(),
                    lowdin_signed_population=float(self.constraint.atom_weights @ atomic_pop),
                    lowdin_signed_spin=float(self.constraint.atom_weights @ atomic_spin),
                    residual=self.residual, multiplier_hartree=self.multiplier,
                    residual_energy_hartree=self.residual_energy,
                    physical_energy_hartree=self.energy,
                    augmented_energy_hartree=self.energy+self.residual_energy,
                    spin_square=float(self.method.spin_square()[0]),
                    spin_population=float(np.einsum(
                        'ij,ji', self.density[0]-self.density[1], self.constraint.matrix)),
                    electron_counts=list(self.method.mol.nelec),
                    reference_sha256=self.constraint.reference_sha256,
                    iterations=self.iterations)


def solve(method, constraint, target, *, dm0=None, multiplier0=0.,
          population_tol=1e-8, max_multiplier=1., max_iterations=40,
          inner_solver="newton", inner_lindep=1e-24):
    '''Solve one population-constrained UKS state without mutating the input.

    One bounded scalar multiplier is solved with fully converged inner UKS.
    inner_solver selects official second-order SCF (newton) or DIIS explicitly.
    inner_lindep is the Newton AH subspace cutoff; its default retains directions
    needed for tight population residuals. No silent solver fallback is used.
    Inner SCF failure and missing brackets raise ConvergenceError; neither an
    unconstrained state nor an earlier force is silently returned.
    '''
    if (not isinstance(method, uks.UKS) or method.mol is not constraint.mol or
            method.do_nlc() or getattr(method, 'with_df', None) is not None or
            method.mol.symmetry or method.small_rho_cutoff != 0):
        raise ValueError('Only finite UKS, no DF/NLC/symmetry or density pruning is supported')
    if (not np.isfinite([target, multiplier0, population_tol, max_multiplier]).all()
            or population_tol <= 0 or max_multiplier <= 0
            or abs(multiplier0) > max_multiplier or type(max_iterations) is not int
            or max_iterations < 2):
        raise ValueError('Invalid fixed target or bounded constraint solver controls')
    if not (-method.mol.nelectron*max(abs(constraint.atom_weights)) < target <
            method.mol.nelectron*max(abs(constraint.atom_weights))):
        raise ValueError('Population target exceeds the electron-number bound')
    if (inner_solver not in ("newton", "diis") or not np.isfinite(inner_lindep)
            or not 0 < inner_lindep <= 1e-14):
        raise ValueError('Invalid inner UKS solver or AH linear-dependence cutoff')
    orbital_limit = (method.conv_tol_grad if method.conv_tol_grad is not None
                     else np.sqrt(method.conv_tol))
    mf = method.copy()
    # Official SCF's final extra diagonalization permits a looser gradient.
    # Solve more tightly internally, then enforce the requested bound explicitly.
    mf.conv_tol_grad = min(orbital_limit * .1, population_tol)
    mf.grids = method.grids.copy()
    mf.chkfile = None
    hcore = method.get_hcore()
    density = None if dm0 is None else np.array(dm0, copy=True)
    records = []
    results = {}

    class PopulationConverged(Exception):
        def __init__(self, multiplier):
            self.multiplier = multiplier

    def evaluate(multiplier):
        nonlocal density, mf
        multiplier = float(multiplier)
        mf.get_hcore = lambda mol=None: hcore+multiplier*constraint.matrix
        if inner_solver == "newton":
            inner = mf.newton()
            inner.ah_lindep = inner_lindep
            # A relative AH residual as large as the current orbital gradient
            # can accept a zero step in a nearly orthogonal trial subspace.
            # Resolve the inner equation below the requested orbital accuracy.
            inner.ah_conv_tol = min(inner.ah_conv_tol, (mf.conv_tol_grad*.1)**2)
            cycles = 0
            callback = mf.callback

            def count_cycle(env):
                nonlocal cycles
                cycles = env['imacro']+1
                if callable(callback):
                    callback(env)

            inner.callback = count_cycle
            inner.kernel(dm0=density)
            mf = inner.undo_soscf()
            mf.callback = callback
            mf.cycles = cycles
        else:
            mf.kernel(dm0=density)
        orbital_residual = np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ))
        if not mf.converged:
            raise ConvergenceError('UKS %s did not converge at multiplier %.12g; '
                                   'orbital residual %.12g after %d cycles' %
                                   (inner_solver, multiplier, orbital_residual, mf.cycles))
        if not np.isfinite(orbital_residual) or orbital_residual > orbital_limit:
            raise ConvergenceError('UKS orbital residual %.6g exceeds %.6g' %
                                   (orbital_residual, orbital_limit))
        density = mf.make_rdm1()
        population = constraint.population(density)
        physical = float(mf.e_tot-multiplier*population)
        residual = population-target
        records.append(dict(multiplier_hartree=multiplier, population=population,
                            physical_energy_hartree=physical,
                            orbital_residual=float(orbital_residual),
                            scf_cycles=int(mf.cycles)))
        lib.logger.debug(mf, 'cDFT lambda %.12g residual %.6g', multiplier, residual)
        results[multiplier] = (population, physical)
        if abs(residual) <= population_tol:
            raise PopulationConverged(multiplier)
        return residual

    try:
        initial = evaluate(multiplier0)
        direction = 1 if initial > 0 else -1
        left = float(multiplier0)
        fleft = initial
        step = .025
        while True:
            right = float(np.clip(multiplier0+direction*step, -max_multiplier, max_multiplier))
            fright = evaluate(right)
            if fleft*fright <= 0:
                break
            if abs(right) == max_multiplier:
                raise ConvergenceError('Population target is not bracketed within the multiplier bound')
            left, fleft = right, fright
            step *= 2
        root = brentq(evaluate, min(left, right), max(left, right),
                      xtol=1e-12, rtol=1e-12, maxiter=max_iterations)
        evaluate(root)
    except PopulationConverged as converged:
        root = converged.multiplier
    population, energy = results[root]
    if abs(population-target) > population_tol:
        raise ConvergenceError('Final cDFT population residual %.12g exceeds %.12g at lambda %.12g' %
                               (population-target, population_tol, root))
    return Result(mf, constraint, float(target), root, population, energy, records)

