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
        # The caller passes the raw make_rdm1() array, which is 3D for a UKS
        # state. Reduce it to the same total-density representation that is
        # used for the current state, otherwise the shape comparison below
        # never matches and the overlap is silently reported as zero.
        if prev.ndim == 3:
            prev = prev.sum(axis=0)
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


def _branch_mismatch(fingerprint, required_signed_spin, required_atom_spins,
                     spin_tol):
    '''Return a reason string when the fingerprint violates the required identity.

    ``None`` means the fingerprint satisfies every requirement. The check is
    deliberately a pure function of the fingerprint so that it can be applied
    to every scan point, to every refinement trial and to the finally
    accepted root with exactly the same semantics.
    '''
    if required_signed_spin is not None:
        if abs(fingerprint.signed_spin-required_signed_spin) > spin_tol:
            return ('signed total spin %.6g does not match the required %.6g'
                    % (fingerprint.signed_spin, required_signed_spin))
    if required_atom_spins:
        for atom_index in sorted(required_atom_spins):
            expected = required_atom_spins[atom_index]
            if atom_index >= len(fingerprint.atom_spins):
                return ('atom %d carries no fingerprint in this state' % atom_index)
            observed = fingerprint.atom_spins[atom_index]
            if abs(observed-expected) > spin_tol:
                return ('atom %d carries signed spin %.6g, required %.6g'
                        % (atom_index, observed, expected))
    return None


def _branch_summary(converged_points):
    '''Summarise the distinct branches that the scan visited.

    The summary is what makes a failed branch requirement actionable: it
    states which signed spins were actually reachable, so that the operator
    can state the intended branch identity instead of guessing a tolerance.
    '''
    clusters = []
    for multiplier, population, energy, state, fingerprint, density in converged_points:
        placed = False
        for cluster in clusters:
            if abs(cluster['signed_spin']-fingerprint.signed_spin) <= 0.2:
                cluster['multiplier_range'][0] = min(cluster['multiplier_range'][0],
                                                     multiplier)
                cluster['multiplier_range'][1] = max(cluster['multiplier_range'][1],
                                                     multiplier)
                cluster['population_range'][0] = min(cluster['population_range'][0],
                                                     population)
                cluster['population_range'][1] = max(cluster['population_range'][1],
                                                     population)
                cluster['n_points'] += 1
                placed = True
                break
        if not placed:
            clusters.append({'signed_spin': fingerprint.signed_spin,
                             'spin_square': fingerprint.spin_square,
                             'atom_spins': list(fingerprint.atom_spins),
                             'multiplier_range': [multiplier, multiplier],
                             'population_range': [population, population],
                             'n_points': 1})
    return clusters


def _refine(converged_points, target, population_tol, inner_solve, density_tol,
            population_of, overlap, eigvals, eigvecs, max_iterations=40, emit=None,
            branch_check=None):
    '''Polish a bracket on a single branch with a guarded secant iteration.

    The bracket is formed from two neighbouring scan points whose populations
    straddle the target. The refinement only accepts a step when the density
    overlap with the previous point stays above ``density_tol``, so the
    iteration cannot silently cross a branch boundary.

    The bracket is maintained explicitly. A secant step is taken when it stays
    inside the bracket, and a bisection is used when the secant step is
    degenerate or when the inner solve at the secant point fails. A failed
    inner solve is retried from the other bracket endpoint before the
    iteration gives up, because the warm start density decides which branch
    the inner solver follows.

    Args:
        population_of: callable mapping a density matrix to its constraint
            population.
        branch_check: optional callable mapping a fingerprint to a reason string
            when the fingerprint violates the required branch identity, and to
            ``None`` when it satisfies it. When given, only branch-consistent
            scan points may form the bracket, and a refinement trial whose
            fingerprint violates the identity is rejected instead of being
            accepted as the root.
    '''
    if len(converged_points) < 2:
        return None
    ordered = sorted(converged_points, key=lambda item: item[0])
    # When a branch identity is required, the bracket may only be formed from
    # points on a branch that satisfies it. Otherwise the refinement could
    # walk onto a different electronic branch and return a state that does
    # not carry the requested identity.
    if branch_check is not None:
        allowed = [item for item in ordered
                   if branch_check(item[4]) is None]
        ordered = allowed
        if len(ordered) < 2:
            return None
    bracket = None
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if (lower[1]-target)*(upper[1]-target) <= 0:
            bracket = (lower, upper)
            break
    if bracket is None:
        return None
    left_mult, left_pop = bracket[0][0], bracket[0][1]
    right_mult, right_pop = bracket[1][0], bracket[1][1]
    # The warm start density decides which branch the inner solver follows.
    # Start from the endpoint that is closer in multiplier to the target, so
    # that the iteration stays on the branch that brackets the root.
    left_density = np.asarray(bracket[0][5])
    right_density = np.asarray(bracket[1][5])
    previous_density = right_density
    force_bisect = False
    for _ in range(max_iterations):
        denominator = right_pop-left_pop
        if abs(denominator) < 1e-14:
            break
        # Standard secant step on f(lambda) = C(lambda) - target.
        trial = right_mult-(right_pop-target)*(right_mult-left_mult)/denominator
        clipped = float(np.clip(trial, min(left_mult, right_mult),
                                 max(left_mult, right_mult)))
        # A secant step that leaves the bracket, or that degenerates, is
        # replaced by a bisection. This keeps the bracket contracting even
        # when the response curve is strongly nonlinear. A trial that was
        # rejected for crossing a branch boundary also forces a bisection, so
        # that the iteration approaches the bracket centre instead of
        # repeatedly stepping across the same discontinuity.
        secant_is_usable = (not force_bisect
                            and abs(trial-clipped) < 1e-12
                            and min(left_mult, right_mult) < clipped
                            < max(left_mult, right_mult))
        if not secant_is_usable:
            clipped = 0.5*(left_mult+right_mult)
        force_bisect = False
        # Warm start from the bracket endpoint nearer to the trial.
        if abs(clipped-left_mult) <= abs(clipped-right_mult):
            start_density = left_density
        else:
            start_density = right_density
        state = None
        for candidate in (start_density, left_density, right_density):
            try:
                state, _ = inner_solve(clipped, candidate)
                break
            except cdft.ConvergenceError:
                continue
        if state is None:
            if emit is not None:
                emit('  REFINE lambda=%+.12g FAILED (no inner solve converged)'
                     % clipped)
            break
        density = np.asarray(state.make_rdm1())
        population = population_of(density)
        trial_fingerprint = _fingerprint(state, density, population, overlap,
                                         eigvals, eigvecs, previous_density)
        # The branch-boundary guard. A refinement step that crosses a branch
        # boundary is rejected, because the warm start may follow the other
        # branch and the trial would then report a population that belongs to
        # a different electronic state. Without this guard the secant step
        # oscillates across the discontinuity of the multivalued response and
        # never converges, while the iteration still looks healthy.
        crossed = (trial_fingerprint.density_overlap < density_tol)
        if crossed:
            if emit is not None:
                emit('  REFINE lambda=%+.12g REJECTED (density overlap %.6g '
                     'below threshold %.6g)'
                     % (clipped, trial_fingerprint.density_overlap, density_tol))
            # The trial crossed the boundary, so it carries no information
            # about the branch that is bracketed. The bracket is not updated
            # and the next iteration bisects instead of taking a secant step,
            # which moves the trial point towards the bracket centre and away
            # from the boundary. The warm start is taken from the endpoint
            # that is known to be on the bracketed branch.
            force_bisect = True
        else:
            force_bisect = False
        if not crossed and branch_check is not None:
            mismatch = branch_check(trial_fingerprint)
            if mismatch is not None:
                # The trial landed on a different electronic branch. It is
                # rejected rather than accepted, because accepting it would
                # return a state that does not carry the requested identity.
                if emit is not None:
                    emit('  REFINE lambda=%+.12g REJECTED (%s)'
                         % (clipped, mismatch))
                # The bracket endpoint that produced this branch is the one
                # that carries the required identity, so the refinement keeps
                # its bracket and tries a different trial point.
                break
        if emit is not None and not crossed:
            emit('  REFINE lambda=%+.12g C=%+.12g residual=%+.6g'
                 % (clipped, population, population-target))
        if not crossed and abs(population-target) <= population_tol:
            energy = float(state.e_tot-clipped*population)
            fingerprint = trial_fingerprint
            return clipped, population, energy, state, fingerprint, density
        if not crossed:
            # Update the bracket so that it always straddles the target.
            if (population-target)*(left_pop-target) <= 0:
                right_mult, right_pop = left_mult, left_pop
                right_density = left_density
                left_mult, left_pop = clipped, population
                left_density = density
            else:
                left_mult, left_pop = clipped, population
                left_density = density
            previous_density = density
        # A crossed trial is discarded: it carries no information about the
        # bracketed branch, so neither the bracket nor the warm-start
        # reference is updated from it.
    return None


def solve_branch_aware(method, constraint, target, *, dm0=None, multiplier0=0.,
                       population_tol=1e-8, max_multiplier=1., max_iterations=40,
                       inner_solver='newton', inner_lindep=1e-24,
                       scan_points=17, density_tol=0.5,
                       scan_mode='uniform', scan_log=None, scan_upper=None,
                       required_signed_spin=None, required_atom_spins=None,
                       spin_tol=0.2, refine=True, scf_tol_grad=None):
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
        scan_log: optional scan log sink. Either a list that is filled in
            place, or a callable that is called with each record as it is
            produced, so that a long scan stays observable.
        scan_upper: optional tighter bound on |lambda| for the scan axis.
        required_signed_spin: optional signed total spin that the accepted root
            must reproduce within ``spin_tol``. When it is given, the scan
            points that do not carry that identity are *filtered out* rather
            than being accepted and then rejected, so that a solution which
            exists elsewhere on the multiplier axis is still found.
        required_atom_spins: optional mapping from atom index to the signed
            spin that atom must carry. It acts as a filter in the same way as
            ``required_signed_spin``.
        refine: when true, the accepted grid root is polished with a local
            secant refinement that stays on the same branch.
        scf_tol_grad: optional explicit orbital-gradient tolerance for the
            inner SCF. It is deliberately separate from ``population_tol``:
            the latter bounds the constraint residual in electrons, the
            former bounds the inner SCF in an orbital gradient. Reusing one
            for the other silently drives the inner solve to an unreachable
            tolerance, because the auxiliary-space tolerance is the square of
            the gradient tolerance.

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
    # The scan log may be a list that is filled in place, or a callable that
    # is called for every record. A callable lets a caller stream the log
    # while the scan is still running. In both cases the records are also
    # kept in an internal list, so that the completed scan can always report
    # its own log.
    records = []
    if scan_log is None:
        def emit(message):
            records.append(message)
    elif callable(scan_log):
        def emit(message):
            records.append(message)
            scan_log(message)
    else:
        def emit(message):
            records.append(message)
            scan_log.append(message)

    mf = method.copy()
    # The inner SCF gradient tolerance is independent of the constraint
    # residual tolerance. It defaults to the method's own limit, and never
    # to population_tol, which lives on a different scale entirely.
    inner_grad_tol = limit*.1 if scf_tol_grad is None else float(scf_tol_grad)
    mf.conv_tol_grad = inner_grad_tol
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
        local.conv_tol_grad = inner_grad_tol
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
            emit('lambda=%+.12g FAILED (%s)' % (multiplier, error))
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
        emit('lambda=%+.12g C=%+.12g residual=%+.6g <S^2>=%.6f overlap=%.6f'
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
            emit('  BRANCH SWITCH at lambda=%+.12g overlap=%.6g dC=%+.6g'
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
                                 'points': points, 'switches': switches,
                                 'log': list(records)}
        raise cdft.ConvergenceError(
            'Branch scan found no converged constrained state: %s' % reason)

    # The required branch identity, if any, is applied as a filter over the
    # scan points and not only as a check on the finally accepted root. A
    # filter is required because the constrained response is multivalued: the
    # scan may well visit the requested branch somewhere else on the
    # multiplier axis, and rejecting the whole solve would then discard a
    # solution that does exist.
    branch_check = None
    if required_signed_spin is not None or required_atom_spins:
        def branch_check(fingerprint, _ss=required_signed_spin,
                         _as=required_atom_spins, _tol=spin_tol):
            return _branch_mismatch(fingerprint, _ss, _as, _tol)

    branches = _branch_summary(converged_points)

    # Select the point that best satisfies the constraint. A point is only
    # acceptable when it lies on a contiguous branch, that is, when the
    # density overlap with its predecessor is above the threshold, and when
    # it carries the required branch identity.
    candidates = []
    rejected_by_branch = []
    for index, (multiplier, population, energy, state, fingerprint, dens) in enumerate(converged_points):
        residual = abs(population-target)
        if residual > population_tol:
            continue
        if branch_check is not None:
            mismatch = branch_check(fingerprint)
            if mismatch is not None:
                rejected_by_branch.append({
                    'multiplier_hartree': multiplier,
                    'residual': float(population-target),
                    'signed_spin': fingerprint.signed_spin,
                    'reason': mismatch})
                continue
        candidates.append((residual, index, multiplier, population, energy,
                           state, fingerprint, dens))
    if not candidates:
        # No grid point is exact enough on the required branch. Refine the
        # bracketing interval on the same branch; only if that fails is the
        # scan declared unsuccessful.
        refined = _refine(converged_points, target, population_tol, inner_solve,
                          density_tol, lambda d: constraint.population(d),
                          overlap, eigvals, eigvecs, max_iterations, emit,
                          branch_check)
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
                                     'points': points, 'switches': switches,
                                     'branches': branches,
                                     'rejected_by_branch': rejected_by_branch,
                                     'log': list(records)}
            raise BranchSwitchError(
                'Branch scan did not reach the population target: %s' % reason)

    candidates.sort(key=lambda item: item[0])
    residual, index, multiplier, population, energy, state, fingerprint, dens = candidates[0]
    root = multiplier
    root_density = dens
    root_state = state
    root_fingerprint = fingerprint

    if branch_check is not None:
        # The filter already guarantees this, but the accepted root is
        # checked once more so that a future change to the filter cannot
        # silently return a state with the wrong branch identity.
        mismatch = branch_check(fingerprint)
        if mismatch is not None:
            reason = ('accepted root has signed spin %.6g, required %.6g'
                      % (fingerprint.signed_spin, required_signed_spin))
            cdft.LAST_BRANCH_SCAN = {'converged': False, 'reason': reason,
                                     'points': points, 'switches': switches,
                                     'branches': branches,
                                     'rejected_by_branch': rejected_by_branch,
                                     'log': list(records)}
            raise cdft.ConvergenceError('Branch identity check failed: %s' % reason)

    scan = BranchScan(converged=True, root=root, population=population,
                      energy=energy, points=points, switches=switches, reason='')
    result = cdft.Result(root_state, constraint, float(target), root,
                          population, energy, points)
    result.branch_scan = scan
    result.branch_switches = switches
    result.branch_density = root_density
    result.branch_clusters = branches
    result.branch_rejected = rejected_by_branch
    return result
