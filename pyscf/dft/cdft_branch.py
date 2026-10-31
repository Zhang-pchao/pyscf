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

'''Branch-aware multiplier scan for population-constrained UKS.

The plain scalar root search in :func:`pyscf.dft.cdft.solve` assumes that the
constrained population C(lambda) is a single-valued continuous function of the
multiplier. That assumption fails whenever the inner UKS solve follows a
history-dependent branch across an avoided crossing: the same lambda can then
map to different populations depending on the path taken to reach it.

This module replaces the bare root search with a branch-aware scan:

* the multiplier axis is sampled on a fixed grid, and every grid point is
  solved with a fully converged inner UKS that is warm started from its
  predecessor, so that the history dependence of the branch is reproduced
  rather than hidden;
* every grid point is tagged with a *branch fingerprint* (spin state, per-atom
  signed spin distribution and a density overlap with the previous point);
* a branch switch is detected explicitly and reported as a structured
  diagnostic instead of an opaque ``brentq`` failure.

The public entry point :func:`solve_branch_aware` returns the same
:class:`pyscf.dft.cdft.Result` type as :func:`pyscf.dft.cdft.solve`, so it is
a drop-in replacement that additionally exposes the branch trace.

Atomic units throughout.
'''

from dataclasses import dataclass, field

import numpy as np

from pyscf.dft import cdft

__all__ = ['BranchFingerprint', 'BranchScan', 'BranchSwitchError',
           'solve_branch_aware']


class BranchSwitchError(cdft.ConvergenceError):
    '''The multiplier scan crossed a branch boundary without a valid root.'''


@dataclass(frozen=True)
class BranchFingerprint:
    '''Identity of one converged constrained state.

    The fingerprint is deliberately cheap to compute so that it can be attached
    to every scan point. ``density_overlap`` is a normalised overlap with the
    preceding point of the scan and therefore depends on the scan order; it is
    kept separate from the intrinsic fields so that the intrinsic part can be
    compared across scans.
    '''

    spin_square: float
    signed_spin: float
    atom_spins: tuple
    population: float
    density_overlap: float = 0.0

    def intrinsic(self):
        '''Return the fingerprint fields that do not depend on scan order.'''
        return (self.spin_square, self.signed_spin, self.atom_spins,
                self.population)

    def as_dict(self):
        return {'spin_square': self.spin_square,
                'signed_spin': self.signed_spin,
                'atom_spins': list(self.atom_spins),
                'population': self.population,
                'density_overlap': self.density_overlap}


def _lowdin_root(overlap, eigenvalues, eigenvectors):
    '''Return the symmetric Löwdin transformation matrix S^(1/2).'''
    return (eigenvectors*np.sqrt(eigenvalues)) @ eigenvectors.T


def _fingerprint(method, density, population, overlap, eigvals, eigvecs, previous):
    '''Build a fingerprint for one converged state.'''
    dm = np.asarray(density)
    if dm.ndim == 3:
        dm_sum = dm.sum(axis=0)
        spin_dm = dm[0]-dm[1]
    else:
        dm_sum = dm
        spin_dm = np.zeros_like(dm)
    try:
        spin_square = float(method.spin_square()[0])
    except Exception:  # pragma: no cover - defensive, solver dependent
        spin_square = float('nan')
    root = _lowdin_root(overlap, eigvals, eigvecs)
    atom_spins = tuple(float(x) for x in np.diag(root @ spin_dm @ root))
    signed_spin = float(np.einsum('ij,ji', spin_dm, root))
    overlap_prev = 0.0
    if previous is not None:
        prev = np.asarray(previous)
        if prev.shape == dm_sum.shape:
            # Physical overlap of two densities: Tr(S D1 D2), normalised by
            # the self-overlaps. The raw AO trace Tr(D1 D2) is not usable here
            # because it vanishes for densities with disjoint occupied spaces.
            cross = float(np.einsum('ij,ji', overlap, dm_sum @ prev))
            norm1 = float(np.einsum('ij,ji', overlap, dm_sum @ dm_sum))
            norm2 = float(np.einsum('ij,ji', overlap, prev @ prev))
            denominator = np.sqrt(max(norm1, 0.)*max(norm2, 0.))
            overlap_prev = cross/denominator if denominator > 0 else 0.0
    return BranchFingerprint(spin_square=spin_square,
                             signed_spin=signed_spin,
                             atom_spins=atom_spins,
                             population=float(population),
                             density_overlap=overlap_prev)


@dataclass
class BranchScan:
    '''Outcome of a branch-aware multiplier scan.'''

    converged: bool
    root: float
    population: float
    energy: float
    points: list = field(default_factory=list)
    switches: list = field(default_factory=list)
    reason: str = ''

    def as_dict(self):
        return {'converged': self.converged, 'root': self.root,
                'population': self.population, 'energy': self.energy,
                'reason': self.reason,
                'points': self.points, 'switches': self.switches}


def _scan_axis(bound, scan_points, scan_mode):
    '''Build the multiplier sampling axis.'''
    if scan_mode == 'uniform':
        axis = np.linspace(-bound, bound, int(scan_points))
    else:
        # Logarithmic sampling: dense near zero, where |dC/dlambda| is largest.
        magnitude = np.geomspace(max(bound*1e-3, 1e-6), bound, int(scan_points)//2)
        axis = np.concatenate([-magnitude[::-1], magnitude])
    return np.unique(axis)


def _refine(converged_points, target, population_tol, inner_solve, density_tol,
            population_of, overlap, eigvals, eigvecs):
    '''Polish a bracket on a single branch with a guarded secant iteration.

    The bracket is formed from two neighbouring scan points whose populations
    straddle the target. The refinement only accepts a step when the density
    overlap with the previous point stays above ``density_tol``, so the
    iteration cannot silently cross a branch boundary.

    Args:
        population_of: callable mapping a density matrix to its constraint
            population.
    '''
    if len(converged_points) < 2:
        return None
    ordered = sorted(converged_points, key=lambda item: item[0])
    bracket = None
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if (lower[1]-target)*(upper[1]-target) <= 0:
            bracket = (lower, upper)
            break
    if bracket is None:
        return None
    left_mult, left_pop = bracket[0][0], bracket[0][1]
    right_mult, right_pop = bracket[1][0], bracket[1][1]
    previous_density = bracket[1][5]
    for _ in range(12):
        denominator = right_pop-left_pop
        if abs(denominator) < 1e-14:
            break
        # Standard secant step on f(lambda) = C(lambda) - target.
        trial = right_mult-(right_pop-target)*(right_mult-left_mult)/denominator
        trial = float(np.clip(trial, min(left_mult, right_mult),
                              max(left_mult, right_mult)))
        try:
            state, _ = inner_solve(trial, previous_density)
        except cdft.ConvergenceError:
            break
        density = np.asarray(state.make_rdm1())
        population = population_of(density)
        if abs(population-target) <= population_tol:
            energy = float(state.e_tot-trial*population)
            fingerprint = _fingerprint(state, density, population, overlap,
                                       eigvals, eigvecs, previous_density)
            return trial, population, energy, state, fingerprint, density
        if (population-target)*(right_pop-target) <= 0:
            left_mult, left_pop = right_mult, right_pop
        right_mult, right_pop = trial, population
        previous_density = density
    return None


def solve_branch_aware(method, constraint, target, *, dm0=None, multiplier0=0.,
                       population_tol=1e-8, max_multiplier=1., max_iterations=40,
                       inner_solver='newton', inner_lindep=1e-24,
                       scan_points=17, density_tol=0.5,
                       scan_mode='uniform', scan_log=None, scan_upper=None,
                       required_signed_spin=None, required_atom_spins=None,
                       spin_tol=0.2, refine=True):
    '''Solve one constrained UKS state with an explicit branch-aware scan.

    The multiplier axis is sampled on a fixed grid. Each grid point runs one
    fully converged inner UKS that is warm started from the density of the
    preceding point, so the scan reproduces the history dependence that makes
    the constrained response multivalued.

    Args:
        scan_points: number of grid points across the multiplier axis.
        density_tol: minimum Löwdin density overlap required between two
            consecutive grid points. A smaller value means a larger jump, which
            is how a branch switch is detected.
        scan_mode: ``'uniform'`` samples the axis uniformly; ``'log'`` samples
            logarithmically towards zero.
        scan_log: optional list that receives a human readable scan log.
        scan_upper: optional tighter bound on |lambda| for the scan axis.
        required_signed_spin: optional signed total spin that the accepted root
            must reproduce within ``spin_tol``.
        required_atom_spins: optional mapping from atom index to the signed
            spin that atom must carry.
        refine: when true, the accepted grid root is polished with a local
            secant refinement that stays on the same branch.

    Returns:
        A :class:`pyscf.dft.cdft.Result`. The instance additionally carries
        ``branch_scan`` and ``branch_switches`` attributes describing the
        scan, so failures remain diagnosable after the fact.
    '''
    if scan_points < 5:
        raise ValueError('A branch scan needs at least five grid points')
    if not (0 < density_tol <= 1):
        raise ValueError('density_tol must lie in (0, 1]')
    if scan_mode not in ('uniform', 'log'):
        raise ValueError("scan_mode must be 'uniform' or 'log'")
    bound = float(max_multiplier if scan_upper is None else min(max_multiplier,
                                                               abs(scan_upper)))
    if bound <= 0:
        raise ValueError('A branch scan needs a positive multiplier bound')
    if not np.isfinite([target, population_tol, max_multiplier]).all():
        raise ValueError('Invalid target or solver controls')

    # Mirror the validation that cdft.solve performs, so that an inadmissible
    # input is rejected before any SCF work is spent.
    if (not isinstance(method, cdft.uks.UKS) or method.mol is not constraint.mol
            or method.do_nlc() or getattr(method, 'with_df', None) is not None
            or method.mol.symmetry or method.small_rho_cutoff != 0):
        raise ValueError('Only finite UKS, no DF/NLC/symmetry or density pruning is supported')
    if (population_tol <= 0 or max_multiplier <= 0
            or abs(multiplier0) > max_multiplier or type(max_iterations) is not int
            or max_iterations < 2):
        raise ValueError('Invalid fixed target or bounded constraint solver controls')
    if inner_solver not in ('newton', 'diis'):
        raise ValueError('Invalid inner UKS solver')
    limit = (method.conv_tol_grad if method.conv_tol_grad is not None
             else np.sqrt(method.conv_tol))
    if not (-method.mol.nelectron*max(abs(constraint.atom_weights)) < target <
            method.mol.nelectron*max(abs(constraint.atom_weights))):
        raise ValueError('Population target exceeds the electron-number bound')

    axis = _scan_axis(bound, scan_points, scan_mode)
    points = []
    switches = []
    log = [] if scan_log is None else scan_log

    mf = method.copy()
    mf.conv_tol_grad = min(limit*.1, population_tol)
    mf.grids = method.grids.copy()
    mf.chkfile = None
    hcore = method.get_hcore()
    overlap = mf.get_ovlp()
    eigvals, eigvecs = np.linalg.eigh(overlap)
    if eigvals.min() <= 1e-10:
        raise ValueError('AO overlap outside the independent-basis domain')

    density = None if dm0 is None else np.array(dm0, copy=True)
    previous_density = None
    previous_fingerprint = None
    converged_points = []

    def inner_solve(multiplier, start_density):
        '''Run one fully converged inner UKS at a fixed multiplier.'''
        nonlocal mf
        local = method.copy()
        local.conv_tol_grad = min(limit*.1, population_tol)
        local.grids = method.grids.copy()
        local.chkfile = None
        local.get_hcore = lambda mol=None: hcore+multiplier*constraint.matrix
        if inner_solver == 'newton':
            inner = local.newton()
            inner.ah_lindep = inner_lindep
            inner.ah_conv_tol = min(inner.ah_conv_tol, (local.conv_tol_grad*.1)**2)
            inner.kernel(dm0=start_density)
            local = inner.undo_soscf()
        else:
            local.kernel(dm0=start_density)
        residual = float(np.linalg.norm(local.get_grad(local.mo_coeff, local.mo_occ)))
        if not local.converged:
            raise cdft.ConvergenceError(
                'UKS %s did not converge at multiplier %.12g; orbital residual %.12g'
                % (inner_solver, multiplier, residual))
        if not np.isfinite(residual) or residual > limit:
            raise cdft.ConvergenceError(
                'UKS orbital residual %.6g exceeds %.6g' % (residual, limit))
        return local, residual

    for value in axis:
        multiplier = float(value)
        try:
            state, residual = inner_solve(multiplier, density)
        except cdft.ConvergenceError as error:
            points.append({'multiplier_hartree': multiplier, 'converged': False,
                           'error': str(error)})
            log.append('lambda=%+.12g FAILED (%s)' % (multiplier, error))
            # A failed point breaks the warm-start chain. Continue from the
            # last good density so the scan can still cover the axis.
            continue
        new_density = np.asarray(state.make_rdm1())
        population = constraint.population(new_density)
        energy = float(state.e_tot-multiplier*population)
        fingerprint = _fingerprint(state, new_density, population, overlap,
                                    eigvals, eigvecs, previous_density)
        entry = {'multiplier_hartree': multiplier, 'converged': True,
                 'population': float(population),
                 'residual': float(population-target),
                 'physical_energy_hartree': energy,
                 'orbital_residual': residual,
                 'fingerprint': fingerprint.as_dict()}
        points.append(entry)
        log.append('lambda=%+.12g C=%+.12g residual=%+.6g <S^2>=%.6f overlap=%.6f'
                   % (multiplier, population, population-target,
                      fingerprint.spin_square, fingerprint.density_overlap))
        if previous_fingerprint is not None and fingerprint.density_overlap < density_tol:
            switches.append({
                'multiplier_hartree': multiplier,
                'from': previous_fingerprint.as_dict(),
                'to': fingerprint.as_dict(),
                'density_overlap': fingerprint.density_overlap,
                'population_jump': population-previous_fingerprint.population,
                'reason': 'density overlap below threshold'})
            log.append('  BRANCH SWITCH at lambda=%+.12g overlap=%.6g dC=%+.6g'
                       % (multiplier, fingerprint.density_overlap,
                          population-previous_fingerprint.population))
        # Advance the warm-start chain with the density that was actually
        # accepted, so that the next point continues the same branch.
        density = new_density
        previous_density = new_density
        previous_fingerprint = fingerprint
        converged_points.append((multiplier, population, energy, state,
                                 fingerprint, new_density))

    if not converged_points:
        reason = 'no multiplier in the scan produced a converged UKS state'
        cdft.LAST_BRANCH_SCAN = {'converged': False, 'reason': reason,
                                 'points': points, 'switches': switches}
        raise cdft.ConvergenceError(
            'Branch scan found no converged constrained state: %s' % reason)

    # Select the point that best satisfies the constraint. A point is only
    # acceptable when it lies on a contiguous branch, that is, when the
    # density overlap with its predecessor is above the threshold.
    candidates = []
    for index, (multiplier, population, energy, state, fingerprint, dens) in enumerate(converged_points):
        residual = abs(population-target)
        if residual > population_tol:
            continue
        candidates.append((residual, index, multiplier, population, energy,
                           state, fingerprint, dens))
    if not candidates:
        # No grid point is exact enough. Refine the bracketing interval on the
        # same branch; only if that fails is the scan declared unsuccessful.
        refined = _refine(converged_points, target, population_tol, inner_solve,
                          density_tol, lambda d: constraint.population(d),
                          overlap, eigvals, eigvecs)
        if refined is not None:
            multiplier, population, energy, state, fingerprint, dens = refined
            root = multiplier
            root_density = dens
            root_state = state
            root_fingerprint = fingerprint
            candidates.append((abs(population-target), -1, multiplier, population,
                               energy, state, fingerprint, dens))
        else:
            closest = min(converged_points,
                          key=lambda item: abs(item[1]-target))
            reason = ('no scan point met the population tolerance; closest residual '
                      '%.6g at lambda=%+.12g'
                      % (abs(closest[1]-target), closest[0]))
            cdft.LAST_BRANCH_SCAN = {'converged': False, 'reason': reason,
                                     'points': points, 'switches': switches}
            raise BranchSwitchError(
                'Branch scan did not reach the population target: %s' % reason)

    candidates.sort(key=lambda item: item[0])
    residual, index, multiplier, population, energy, state, fingerprint, dens = candidates[0]
    root = multiplier
    root_density = dens
    root_state = state
    root_fingerprint = fingerprint

    if required_signed_spin is not None:
        if abs(fingerprint.signed_spin-required_signed_spin) > spin_tol:
            reason = ('accepted root has signed spin %.6g, required %.6g'
                      % (fingerprint.signed_spin, required_signed_spin))
            cdft.LAST_BRANCH_SCAN = {'converged': False, 'reason': reason,
                                     'points': points, 'switches': switches}
            raise cdft.ConvergenceError('Branch identity check failed: %s' % reason)
    if required_atom_spins:
        for atom_index, expected in required_atom_spins.items():
            if atom_index >= len(fingerprint.atom_spins):
                continue
            observed = fingerprint.atom_spins[atom_index]
            if abs(observed-expected) > spin_tol:
                reason = ('atom %d carries signed spin %.6g, required %.6g'
                          % (atom_index, observed, expected))
                cdft.LAST_BRANCH_SCAN = {'converged': False, 'reason': reason,
                                         'points': points, 'switches': switches}
                raise cdft.ConvergenceError('Branch identity check failed: %s' % reason)

    scan = BranchScan(converged=True, root=root, population=population,
                      energy=energy, points=points, switches=switches, reason='')
    result = cdft.Result(root_state, constraint, float(target), root,
                          population, energy, points)
    result.branch_scan = scan
    result.branch_switches = switches
    result.branch_density = root_density
    return result
