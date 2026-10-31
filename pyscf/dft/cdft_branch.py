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
           'BranchGapError', 'solve_branch_aware']


class BranchSwitchError(cdft.ConvergenceError):
    '''The multiplier scan crossed a branch boundary without a valid root.'''


class BranchGapError(BranchSwitchError):
    '''The target population lies in a gap between two electronic branches.

    This is a statement about the model rather than about the numerics: the
    constrained response is genuinely multivalued and the requested
    population is not attained by any reachable branch. Reporting it
    separately keeps a physically unreachable target from being mistaken for a
    solver that failed to converge.
    '''


@dataclass(frozen=True)
class BranchFingerprint:
    '''Identity of one converged constrained state.

    The fingerprint is deliberately cheap to compute so that it can be attached
    to every scan point. ``density_overlap`` is a normalised overlap with the
    preceding point of the scan and therefore depends on the scan order; it is
    kept separate from the intrinsic fields so that the intrinsic part can be
    compared across scans.

    The spin quantities are physical observables:

    ``total_spin``
        the total spin ``N_alpha - N_beta`` of the state, i.e. the trace of the
        spin density matrix in the AO metric. It is bounded by the electron
        count and is therefore a usable branch identity.
    ``atom_spins``
        the Loewdin signed spin of each *atom*, summed over the basis
        functions that belong to that atom. The sum over all atoms reproduces
        ``total_spin``.
    ``fragment_spins``
        the signed spin of each named fragment, as a tuple of ``(name, spin)``
        pairs. A fragment spin is the signed contraction of the atomic spins
        with the signed fragment weights of the constraint. This is the
        quantity that distinguishes a hole localised on the donor from a hole
        delocalised into the solvent, which is exactly the identity that a
        donor-definition comparison has to resolve.
    '''

    spin_square: float
    total_spin: float
    atom_spins: tuple
    population: float
    fragment_spins: tuple = ()
    density_overlap: float = 0.0

    def intrinsic(self):
        '''Return the fingerprint fields that do not depend on scan order.'''
        return (self.spin_square, self.total_spin, self.atom_spins,
                self.fragment_spins, self.population)

    def as_dict(self):
        return {'spin_square': self.spin_square,
                'total_spin': self.total_spin,
                'atom_spins': list(self.atom_spins),
                'fragment_spins': list(self.fragment_spins),
                'population': self.population,
                'density_overlap': self.density_overlap}

def _lowdin_root(overlap, eigenvalues, eigenvectors):
    '''Return the symmetric Löwdin transformation matrix S^(1/2).'''
    return (eigenvectors*np.sqrt(eigenvalues)) @ eigenvectors.T


def _fingerprint(method, density, population, overlap, eigvals, eigvecs, previous,
                 fragment_weights=None):
    '''Build a fingerprint for one converged state.

    The spin quantities are physical observables:

    * the total spin is ``Tr(spin_dm @ S)`` in the AO metric, which reproduces
      ``N_alpha - N_beta`` exactly for a converged state;
    * the atomic spins are the Loewdin diagonal summed over the basis
      functions of each atom, so that they sum to the total spin;
    * the fragment spins contract the atomic spins with the signed fragment
      weights of the constraint.

    An earlier version of this helper reported ``Tr(spin_dm @ S^(1/2))`` as the
    total spin and the raw Loewdin diagonal as the atomic spins. Both were
    wrong: the first is not the total spin and is not bounded by the electron
    count, and the second is a per basis function quantity that silently grew
    with the basis size. A branch identity built on those numbers could accept
    a state whose reported spin was physically impossible.
    '''
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
    # The total spin is the trace of the spin density in the AO metric.
    total_spin = float(np.einsum('ij,ji', spin_dm, overlap))
    # The Loewdin diagonal is a per basis function quantity. Sum it over the
    # basis functions of each atom so that the atomic spins are per atom and
    # add up to the total spin.
    per_basis = np.diag(root @ spin_dm @ root)
    slices = method.mol.aoslice_by_atom()
    atom_spins = tuple(float(per_basis[p0:p1].sum()) for _, _, p0, p1 in slices)
    fragment_spins = ()
    if fragment_weights is not None:
        # ``fragment_weights`` maps a fragment name to the signed weight of
        # each atom: {name: [w0, w1, ...]}. The fragment spin is the signed
        # contraction of the atomic spins with that weight vector.
        names = list(fragment_weights)
        rows = []
        for name in names:
            weights = np.asarray(fragment_weights[name], dtype=float)
            if weights.shape != (len(atom_spins),):
                raise ValueError('Fragment weights must match the number of atoms')
            rows.append((str(name), float(weights @ np.asarray(atom_spins))))
        fragment_spins = tuple(rows)
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
                             total_spin=total_spin,
                             atom_spins=atom_spins,
                             fragment_spins=fragment_spins,
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

    ``required_signed_spin`` constrains the total spin ``N_alpha - N_beta``.
    ``required_atom_spins`` constrains the Loewdin signed spin of individual
    atoms, and the keys are *atom* indices, not basis function indices.
    '''
    if required_signed_spin is not None:
        if abs(fingerprint.total_spin-required_signed_spin) > spin_tol:
            return ('total spin %.6g does not match the required %.6g'
                    % (fingerprint.total_spin, required_signed_spin))
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


def _branch_gap(converged_points, target, discontinuity_evidence=None):
    '''Describe a gap in the reachable population range that contains the target.

    The constrained response can be multivalued: two electronic branches may
    cover disjoint intervals of the population axis, so that a target between
    them is attained by neither branch. The scan then reports "no point met
    the tolerance", which reads like a numerical failure and invites a
    tolerance that is then loosened.

    This helper detects the structural situation instead. A gap is reported
    only when the target falls strictly between two separated intervals that
    no reachable point covers. A target outside the overall reachable range
    is *not* a gap: it is simply out of range.

    The evidence for a discontinuity is the refinement's own observation: a
    trial on the bracketed branch must agree with the linear interpolation of
    the bracket, and a trial that does not agree has crossed into another
    branch. This is the reliable signal, because it is measured *on* the
    branch rather than inferred from a coarse grid.

    A coarse scan alone cannot settle the question. On the V2-04c D1 case
    the response moves by 1.9 electrons across a scan step of 0.125 Hartree
    in the smooth part and by 0.14 electrons across a step of 0.0005 Hartree
    at the true discontinuity, so no threshold on the jump alone separates
    them. The refinement resolves the discontinuity because it measures the
    response at the scale of the bracket it is refining.

    Measured on the V2-04c D1 case, the two limits of the discontinuity at
    lambda ~ 0.0236 carried the same total spin to 1e-6, the same spin square
    to 0.001 and a density overlap of 0.998, while the constrained population
    differed by 0.142 electrons. An identity built on the total spin and the
    spin square alone merges them and reports no gap.

    The limits of the gap are the populations that the two branches actually
    reach, and they are taken from the *observed* trials, not from the coarse
    scan bracket. This distinction decides the outcome on the V2-04c D1 case.
    The coarse bracket that straddles the fold spans the populations 14.173
    and 12.259, and the target 14.0 lies below the lower limit, so the coarse
    bracket reports "out of range". The observed limits are 14.0528 on the
    lower branch and 13.9145 on the upper branch, and the target 14.0 lies
    strictly between them. Only the observed limits identify the target as
    unattainable; the coarse bracket hides the gap entirely.

    The comparison is direction agnostic, because the constrained population
    may decrease or increase with the multiplier depending on the sign of the
    constraint weights.

    Args:
        converged_points: the converged scan points.
        target: the requested population.
        discontinuity_evidence: the discontinuities that the refinement
            observed, as recorded by :func:`_refine`.

    Returns a description string, or ``None`` when the target is not inside a
    structural gap.
    '''
    if not converged_points:
        return None
    # Each observed discontinuity separates two branches. The two limits of
    # the discontinuity are the populations that the branches reach: the
    # population of the trial that was rejected and the population that the
    # bracket predicted for it. A target that falls strictly between those
    # limits is attained by neither branch, because the response does not
    # take the intermediate values.
    for evidence in (discontinuity_evidence or []):
        # A fold that was confirmed by the strict multivaluedness test gives
        # the two limits of the gap directly: those are the populations that
        # the two branches actually reach.
        fold = evidence.get('fold')
        if fold is not None:
            limits = [float(x) for x in fold['limits']]
        else:
            # The population that the bracketed branch predicts at the trial
            # point is the limit of the lower branch, and the measured
            # population of the rejected trial is the limit of the upper
            # branch. The coarse bracket endpoints are *not* used: they are
            # the populations of two scan points that lie far away from the
            # fold.
            limits = []
            if 'bracket_prediction' in evidence:
                limits.append(float(evidence['bracket_prediction']))
            if 'population' in evidence:
                limits.append(float(evidence['population']))
            if len(limits) < 2:
                limits = [float(x) for x in evidence.get('bracket_populations', [])]
        if len(limits) < 2:
            continue
        lower, upper = min(limits), max(limits)
        if lower < target < upper:
            return ('reachable populations stop at %.6f and resume at %.6f '
                    '(the two branches are separated by %.6f electrons)'
                    % (lower, upper, upper-lower))
    return None


def _branch_summary(converged_points, branch_tol=0.2):
    '''Summarise the distinct branches that the scan visited.

    The summary is what makes a failed branch requirement actionable: it
    states which spin identities were actually reachable, so that the operator
    can state the intended identity instead of guessing a tolerance.

    A branch is identified by the total spin, the spin square *and* the
    fragment spin distribution. The total spin alone is not sufficient: a
    spin-contaminated state and a well-behaved state can carry the same total
    spin while differing in ``<S^2>`` by several units, and those are
    different electronic branches. Clustering on the total spin alone would
    merge them and hide exactly the distinction that the branch audit has to
    make.

    The fragment spins are included for a second reason. A hole that migrates
    from the donor to the acceptor leaves the total spin and the spin square
    almost unchanged, because the total spin is a property of the whole
    system and the hole migration is a redistribution of it. Measured on the
    V2-04c D1 case, the two sides of the discontinuity at lambda ~ 0.0236
    carried the *same* total spin (+2.000000) and the same spin square to
    0.001, while the donor fragment spin differed by 0.079 electrons and the
    constrained population jumped by 0.142 electrons.

    The tolerance is deliberately loose here. This function *labels* the
    branches that the scan visited; deciding where one branch ends and the
    next begins is the job of :func:`_refine`, which observes the
    discontinuity of the response directly while it polishes the bracket.
    Using a tight tolerance here would split a single smooth branch into
    many pieces, because the donor fragment spin varies continuously by more
    than one electron across the scan.
    '''
    clusters = []
    for multiplier, population, energy, state, fingerprint, density in converged_points:
        placed = False
        for cluster in clusters:
            if _same_branch(cluster['_fingerprint'], fingerprint, branch_tol):
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
            clusters.append({'total_spin': fingerprint.total_spin,
                             'spin_square': fingerprint.spin_square,
                             'atom_spins': list(fingerprint.atom_spins),
                             'fragment_spins': list(fingerprint.fragment_spins),
                             'multiplier_range': [multiplier, multiplier],
                             'population_range': [population, population],
                             'n_points': 1,
                             '_fingerprint': fingerprint})
    # The internal fingerprint is not part of the reported record.
    for cluster in clusters:
        cluster.pop('_fingerprint', None)
    return clusters


def _branch_identity(fingerprint):
    '''The intrinsic electronic identity of a converged state.

    Two states belong to the same branch when they carry the same total spin,
    the same spin square and the same fragment spin distribution. The
    population is deliberately excluded, because it is the quantity that is
    being solved for and varies continuously along a branch.
    '''
    return (fingerprint.total_spin, fingerprint.spin_square,
            tuple(fingerprint.fragment_spins))


def _same_branch(first, second, tol):
    '''Whether two fingerprints describe the same electronic branch.

    The comparison is on the branch identity and is deliberately strict: a
    trial that lands on a different electronic state must be rejected, because
    its population belongs to a different branch of the multivalued
    constrained response and would otherwise be reported as a root.
    '''
    if abs(first.total_spin-second.total_spin) > tol:
        return False
    if abs(first.spin_square-second.spin_square) > tol:
        return False
    if len(first.fragment_spins) != len(second.fragment_spins):
        return False
    for (_, a), (_, b) in zip(first.fragment_spins, second.fragment_spins):
        if abs(a-b) > tol:
            return False
    return True


def _probe_multivalued(multiplier, left_density, right_density, inner_solve,
                       population_of, overlap, eigvals, eigvecs,
                       fragment_weights=None, tolerance=1e-6):
    '''Test whether the constrained response is multivalued at one multiplier.

    This is the *definition* of a fold of the constrained response: at one
    fixed multiplier, two different initial densities both converge to
    different self-consistent solutions. Nothing weaker establishes it. A
    population that jumps between two neighbouring multipliers may be a
    genuine discontinuity, or it may be an artefact of the warm-start
    history; the difference is decided here, by solving the same multiplier
    from both sides and comparing the two answers.

    The two solutions differ in the localisation of the electronic hole, so
    their fingerprint differs in the fragment spin distribution while their
    total spin and spin square agree. The populations are what the caller
    needs: they are the two limits of the fold.

    Returns a description of the fold, or ``None`` when the response is
    single valued at that multiplier.
    '''
    solutions = []
    for start in (left_density, right_density):
        if start is None:
            continue
        try:
            state, _residual = inner_solve(multiplier, start)
        except cdft.ConvergenceError:
            continue
        density = np.asarray(state.make_rdm1())
        population = float(population_of(density))
        solutions.append((population, density, state))
    if len(solutions) < 2:
        return None
    first, second = solutions[0], solutions[1]
    if abs(first[0]-second[0]) <= tolerance:
        # The two warm starts agree: the response is single valued here.
        return None
    lower, upper = sorted((first[0], second[0]))
    # The fingerprints of the two solutions are reported, because the
    # distinction between them is what makes the fold physical.
    fingerprints = []
    for population, density, state in solutions:
        fingerprint = _fingerprint(state, density, population, overlap,
                                   eigvals, eigvecs, None, fragment_weights)
        fingerprints.append(fingerprint)
    return {'multiplier_hartree': float(multiplier),
            'populations': [float(first[0]), float(second[0])],
            'limits': [float(lower), float(upper)],
            'separation_electrons': float(upper-lower),
            'total_spins': [float(f.total_spin) for f in fingerprints],
            'spin_squares': [float(f.spin_square) for f in fingerprints],
            'fragment_spins': [list(f.fragment_spins) for f in fingerprints]}


def _refine(converged_points, target, population_tol, inner_solve, density_tol,
            population_of, overlap, eigvals, eigvecs, max_iterations=40, emit=None,
            branch_check=None, fragment_weights=None, branch_identity_tol=0.2):
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
        return None, []
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
            return None, []
    bracket = None
    for lower, upper in zip(ordered[:-1], ordered[1:]):
        if (lower[1]-target)*(upper[1]-target) <= 0:
            bracket = (lower, upper)
            break
    if bracket is None:
        return None, []
    left_mult, left_pop = bracket[0][0], bracket[0][1]
    right_mult, right_pop = bracket[1][0], bracket[1][1]
    # The warm start density decides which branch the inner solver follows.
    # Start from the endpoint that is closer in multiplier to the target, so
    # that the iteration stays on the branch that brackets the root.
    left_density = np.asarray(bracket[0][5])
    right_density = np.asarray(bracket[1][5])
    previous_density = right_density
    # The branch identity of the bracketed points. A refinement trial that
    # carries a different identity is on a different electronic branch and
    # must not be accepted as a root.
    bracket_identity = bracket[0][4]
    # The tolerance of the branch identity check. It is used only as a
    # secondary criterion; the primary criterion below detects a crossing
    # directly from the response itself.
    branch_tol = branch_identity_tol
    # The evidence for a discontinuity of the constrained response. The
    # refinement observes it directly: a trial that lies on the bracketed
    # branch must agree with the linear interpolation of the bracket, and a
    # trial that does not agree has crossed into another branch.
    discontinuity_evidence = []
    force_bisect = False
    # The multipliers that were rejected because they crossed a branch
    # boundary. The bisection must not return to them, because the bisection
    # of an unchanged bracket is a fixed point and the iteration would
    # recompute the same rejected multiplier for ever. On the V2-04c D1 case
    # that produced five identical rejections in a row and the refinement
    # spun until the child process was killed.
    rejected_multipliers = []
    # Whether a confirmed fold has already shown that the target lies in a
    # structural gap. When it has, the refinement stops immediately.
    structural_gap = False
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
            # The bisection point must not be one that was already rejected.
            # The interval is halved towards the endpoint that is closer to
            # the trial, which is the side that the response says carries
            # the target.
            low = min(left_mult, right_mult)
            high = max(left_mult, right_mult)
            for _attempt in range(64):
                clipped = 0.5*(low+high)
                if all(abs(clipped-old) > 1e-12 for old in rejected_multipliers):
                    break
                # Move the trial towards the endpoint whose population is on
                # the same side of the target as the other one.
                if (left_pop-target)*(right_pop-target) < 0:
                    low = clipped
                else:
                    high = clipped
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
                                         eigvals, eigvecs, previous_density,
                                         fragment_weights)
        # The branch-boundary guard. A refinement step that crosses a branch
        # boundary is rejected, because the warm start may follow the other
        # branch and the trial would then report a population that belongs to
        # a different electronic state. Without this guard the secant step
        # oscillates across the discontinuity of the multivalued response and
        # never converges, while the iteration still looks healthy.
        #
        # The primary criterion is *internal to the response*. The bracket
        # spans one branch, so the population on it must vary smoothly with
        # the multiplier. A trial whose measured residual disagrees with the
        # linear interpolation of the bracket is therefore not on that
        # branch, and it is rejected. This detects a genuine discontinuity
        # even when the wavefunction barely changes: on the V2-04c D1 case
        # the density overlap across the discontinuity was 0.998 and the
        # total spin was identical, so an overlap test could not see it.
        #
        # The density overlap is retained as a secondary criterion, because a
        # wavefunction jump without a corresponding population anomaly is
        # still a change of branch.
        #
        # The branch identity is a third criterion, used only when the
        # required identity is known. It is applied with a loose tolerance,
        # because the fragment spin distribution varies continuously along a
        # branch and a tight tolerance would reject valid trials.
        span = right_mult-left_mult
        predicted_population = (left_pop+(right_pop-left_pop)
                                *(clipped-left_mult)/span) if span else left_pop
        # The scale of the anomaly that counts as a crossing: a small
        # multiple of the residual that the bracket itself spans.
        anomaly = abs(population-predicted_population)
        bracket_span = abs(right_pop-left_pop)
        crossing_scale = max(0.05*bracket_span, 10.*population_tol)
        discontinuity = anomaly > crossing_scale
        if discontinuity:
            # A population anomaly is only a *candidate* discontinuity. The
            # population may also jump because the warm start followed the
            # other solution of a single valued problem. The fold is
            # therefore confirmed by the strict test: at the same multiplier,
            # solve from both bracket endpoints and check that the two
            # self-consistent answers differ.
            #
            # The confirmation is what makes the gap reportable. Without it,
            # a numerical multivaluedness would be reported as a physical
            # gap, and a physical gap would be reported as a solver failure.
            fold = _probe_multivalued(clipped, left_density, right_density,
                                      inner_solve, population_of, overlap,
                                      eigvals, eigvecs, fragment_weights)
            evidence = {
                'multiplier_hartree': float(clipped),
                'population': float(population),
                'bracket_prediction': float(predicted_population),
                'anomaly_electrons': float(anomaly),
                'bracket_multipliers': [float(left_mult), float(right_mult)],
                'bracket_populations': [float(left_pop), float(right_pop)],
                'density_overlap': float(trial_fingerprint.density_overlap),
                'total_spin': float(trial_fingerprint.total_spin),
                'spin_square': float(trial_fingerprint.spin_square),
                'multivalued_confirmed': fold is not None,
            }
            if fold is not None:
                evidence['fold'] = fold
                # The two limits of the fold are the populations that the two
                # branches actually reach. They are the quantities that
                # decide whether the requested target is attainable.
                evidence['population'] = float(fold['limits'][0])
                evidence['bracket_prediction'] = float(fold['limits'][1])
                if emit is not None:
                    emit('  FOLD CONFIRMED at lambda=%+.12g: the response is '
                         'multivalued, the branches reach %.9f and %.9f'
                         % (clipped, fold['limits'][0], fold['limits'][1]))
                # A confirmed fold whose two limits straddle the target is a
                # structural gap, and no further refinement can reach the
                # target: the response does not take the intermediate values.
                # The iteration therefore stops at once and reports the gap,
                # instead of bisecting towards a root that provably does not
                # exist.
                fold_lower, fold_upper = fold['limits']
                if fold_lower < target < fold_upper:
                    structural_gap = True
                    if emit is not None:
                        emit('  STRUCTURAL GAP: the target %.9g lies between '
                             'the branch limits %.9f and %.9f; the refinement '
                             'stops without a root' % (target, fold_lower,
                                                      fold_upper))
            discontinuity_evidence.append(evidence)
        if structural_gap:
            # The evidence for the gap is complete. The refinement ends here,
            # and the caller turns the evidence into a BranchGapError.
            return None, discontinuity_evidence
        crossed = (trial_fingerprint.density_overlap < density_tol)
        identity_changed = False
        if not crossed and not discontinuity and bracket_identity is not None:
            # The branch identity is a *diagnostic*, not a rejection
            # criterion. The fragment spin distribution drifts continuously
            # along one and the same branch, so a fixed tolerance cannot
            # separate "the hole moved to another fragment" from "the hole
            # moved a little further along the same branch".
            #
            # Measured on the V2-04c D1 case: the donor fragment spin grows
            # from 0.003 to 0.337 electrons while the multiplier runs from 0
            # to 0.03, entirely within one branch. Across the fold the donor
            # spin changes by 0.079 electrons. Any fixed threshold that
            # separates those two numbers also rejects most of the first
            # sequence. Using the identity as a rejection criterion therefore
            # rejected perfectly good trials and made the refinement spin.
            #
            # The rejection criteria are therefore the two that are
            # intrinsic to the *response*: the population must agree with
            # the linear interpolation of the bracket (a discontinuity of the
            # constrained response), and the density overlap must stay high.
            # The identity is recorded as evidence and reported.
            identity_changed = not _same_branch(trial_fingerprint,
                                               bracket_identity, branch_tol)
        if crossed or discontinuity:
            if emit is not None:
                if crossed:
                    emit('  REFINE lambda=%+.12g REJECTED (density overlap %.6g '
                         'below threshold %.6g)'
                         % (clipped, trial_fingerprint.density_overlap,
                            density_tol))
                else:
                    emit('  REFINE lambda=%+.12g REJECTED (population %.9g '
                         'disagrees with the bracket prediction %.9g by %.3g)'
                         % (clipped, population, predicted_population, anomaly))
            # The trial crossed the boundary, so it carries no information
            # about the branch that is bracketed. The bracket is not updated
            # from it and the next iteration bisects instead of taking a
            # secant step, which moves the trial point towards the bracket
            # centre and away from the boundary. The warm start is taken from
            # the endpoint that is known to be on the bracketed branch.
            #
            # The trial point itself must, however, be excluded from the
            # bracket. Without that exclusion the bisection recomputes the
            # same point on every iteration, because the bisection of an
            # unchanged bracket is a fixed point. On the V2-04c D1 case that
            # produced the same rejected multiplier five times in a row and
            # the refinement spun until the child process was killed. The
            # exclusion is done by shrinking the bracket onto the half that
            # keeps the rejected point out, which is exactly what a bisection
            # does; the rejected point is therefore remembered and the next
            # bisection is taken of the *remaining* interval.
            force_bisect = True
            rejected_multipliers.append(clipped)
        else:
            force_bisect = False
        if not crossed and not discontinuity and branch_check is not None:
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
        if emit is not None and not crossed and not discontinuity:
            if identity_changed:
                # The identity moved, but the population stayed on the
                # branch. This is a drift, not a crossing, and it is reported
                # so that the operator can see the hole migrating.
                emit('  REFINE lambda=%+.12g C=%+.12g residual=%+.6g '
                     '(identity drift: donor spin %.6g -> %.6g)'
                     % (clipped, population, population-target,
                        dict(bracket_identity.fragment_spins).get('donor', 0.),
                        dict(trial_fingerprint.fragment_spins).get('donor', 0.)))
            else:
                emit('  REFINE lambda=%+.12g C=%+.12g residual=%+.6g'
                     % (clipped, population, population-target))
        if not crossed and not discontinuity and abs(population-target) <= population_tol:
            energy = float(state.e_tot-clipped*population)
            fingerprint = trial_fingerprint
            return (clipped, population, energy, state, fingerprint,
                    density), discontinuity_evidence
        if not crossed and not discontinuity:
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
    return None, discontinuity_evidence


def solve_branch_aware(method, constraint, target, *, dm0=None, multiplier0=0.,
                       population_tol=1e-8, max_multiplier=1., max_iterations=40,
                       inner_solver='newton', inner_lindep=1e-24,
                       scan_points=17, density_tol=0.5,
                       scan_mode='uniform', scan_log=None, scan_upper=None,
                       required_signed_spin=None, required_atom_spins=None,
                       spin_tol=0.2, refine=True, scf_tol_grad=None,
                       fragment_weights=None, branch_identity_tol=0.2):
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
        required_signed_spin: optional total spin ``N_alpha - N_beta`` that the
            accepted root must reproduce within ``spin_tol``. When it is given,
            the scan points that do not carry that identity are *filtered out*
            rather than being accepted and then rejected, so that a solution
            which exists elsewhere on the multiplier axis is still found.
        required_atom_spins: optional mapping from atom index to the signed
            spin that atom must carry. It acts as a filter in the same way as
            ``required_signed_spin``.
        fragment_weights: optional per-atom signed weights used to report the
            signed spin of each named fragment in the fingerprint. The
            fragment spin is what distinguishes a hole localised on the donor
            from a hole delocalised into the solvent.
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
                                    eigvals, eigvecs, previous_density,
                                    fragment_weights)
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
                    'total_spin': fingerprint.total_spin,
                    'spin_square': fingerprint.spin_square,
                    'reason': mismatch})
                continue
        candidates.append((residual, index, multiplier, population, energy,
                           state, fingerprint, dens))
    if not candidates:
        # No grid point is exact enough on the required branch. Refine the
        # bracketing interval on the same branch; only if that fails is the
        # scan declared unsuccessful. The refinement also reports the
        # discontinuities that it observed, which is the evidence for a
        # structural gap between two electronic branches.
        refined, discontinuity_evidence = _refine(
            converged_points, target, population_tol, inner_solve,
            density_tol, lambda d: constraint.population(d),
            overlap, eigvals, eigvecs, max_iterations, emit,
            branch_check, fragment_weights, branch_identity_tol)
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
            # The target may be unreachable not because the solver is
            # inaccurate but because the constrained response is genuinely
            # multivalued and the target falls into the gap between two
            # electronic branches. That is a statement about the model, not
            # about the numerics, and it must be reported as such.
            gap = _branch_gap(converged_points, target, discontinuity_evidence)
            if gap is not None:
                reason = ('the target population %.6g lies in a gap between two '
                          'electronic branches: %s' % (target, gap))
            cdft.LAST_BRANCH_SCAN = {'converged': False, 'reason': reason,
                                     'points': points, 'switches': switches,
                                     'branches': branches,
                                     'rejected_by_branch': rejected_by_branch,
                                     'branch_gap': gap,
                                     'discontinuities': discontinuity_evidence,
                                     'log': list(records)}
            if gap is not None:
                raise BranchGapError(
                    'Branch scan did not reach the population target: %s' % reason)
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
            reason = ('accepted root has total spin %.6g, required %.6g'
                      % (fingerprint.total_spin, required_signed_spin))
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
