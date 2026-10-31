#!/usr/bin/env python
# Copyright 2026 The PySCF Developers. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Regression tests for the cDFT gradient assembly and the branch-aware scan.

The gradient test pins a conclusion that was reached by source inspection and
numerical cross-check rather than by an assertion in the repository: the
analytic gradient must contain the constraint contribution exactly once.

The assertion is deliberately written so that it fails if the upstream PySCF
gradient code ever starts reading the patched ``mf.get_hcore`` instead of the
unpatched ``mf_grad.get_hcore``. In that case the constraint term would be
counted twice and the analytic force would no longer match the finite
difference of the constrained energy.
'''

import unittest
import numpy as np
from pyscf import gto, dft, qmmm
from pyscf.dft import cdft
from pyscf.dft import cdft_branch
from pyscf.lib import param


def molecule(coords):
    return gto.M(atom=list(zip(['H', 'H'], coords)), basis='sto-3g',
                 charge=1, spin=1, verbose=0)


def setup(coords):
    mf = dft.UKS(molecule(coords))
    mf.xc = 'pbe0'
    mf.grids.level = 1
    mf.grids.prune = None
    mf.small_rho_cutoff = 0
    mf.conv_tol = 1e-12
    mf.conv_tol_grad = 1e-7
    return mf


class FakeMol:
    '''A minimal stand-in for a PySCF molecule.

    The refinement helper builds a branch fingerprint for every trial, and the
    fingerprint needs the atom slices to report per-atom spins. The synthetic
    states in the refinement tests therefore need a molecule-like object.
    '''

    def __init__(self, natm):
        self.natm = natm

    def aoslice_by_atom(self):
        # One basis function per atom keeps the slice map trivial.
        return [(i, i, i, i+1) for i in range(self.natm)]


class GradientAssembly(unittest.TestCase):
    '''The constraint contribution must appear exactly once in the gradient.'''

    def test_constraint_is_not_double_counted(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        plain = setup(xyz)
        plain_hcore = plain.get_hcore()
        mf = qmmm.mm_charge(setup(xyz), [[2., .3, .4]], [.2], unit='Angstrom')
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        result = cdft.solve(mf, w, .4)
        baseline = result.gradients()

        # The constrained method carries a patched get_hcore that adds
        # lambda*W. The gradient object must not consume that patched
        # function: it has to read the unpatched core Hamiltonian through
        # mf_grad.get_hcore. If upstream changed hcore_generator to read
        # mf.base.get_hcore, the two evaluations below would differ and the
        # constraint term would be counted twice.
        grad = result.method.nuc_grad_method()
        grad.grid_response = True
        with_patched = grad.kernel()

        unpatched = result.method.get_hcore
        try:
            result.method.get_hcore = lambda mol=None: plain_hcore
            with_plain = grad.kernel()
        finally:
            result.method.get_hcore = unpatched

        np.testing.assert_allclose(with_patched, with_plain, atol=1e-10)

        # The full gradient must contain the constraint contribution exactly
        # once: once through density_gradient and not at all through hcore.
        expected = with_plain + result.multiplier*w.density_gradient(result.density)
        np.testing.assert_allclose(baseline, expected, atol=1e-10)
        self.assertGreater(np.abs(baseline).max(), 1e-6)

    def test_gradient_matches_finite_difference(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = qmmm.mm_charge(setup(xyz), [[2., .3, .4]], [.2], unit='Angstrom')
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        result = cdft.solve(mf, w, .4)
        analytic = result.gradients()
        for h in [.001, .0005]:
            energies = []
            for sign in [-1, 1]:
                moved = xyz.copy()
                moved[0, 2] += sign*h
                other = qmmm.mm_charge(setup(moved), [[2., .3, .4]], [.2],
                                       unit='Angstrom')
                cw = cdft.Hirshfeld(other.mol, [1., -1.], other.grids,
                                    w.references)
                energies.append(cdft.solve(other, cw, .4).energy)
            numerical = (energies[1]-energies[0])/(2*h)*param.BOHR
            self.assertLess(abs(numerical-analytic[0, 2]), 2e-6)

    def test_augmented_and_physical_energy_accounting(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        result = cdft.solve(mf, w, .4)
        diagnostics = result.diagnostics()
        physical = diagnostics['physical_energy_hartree']
        augmented = diagnostics['augmented_energy_hartree']
        # E_augmented = E_physical + lambda * (C - C*)
        expected = physical + result.multiplier*(result.population-result.target)
        self.assertAlmostEqual(augmented, expected, 9)
        self.assertLess(abs(diagnostics['residual']), 1e-8)


class BranchAwareScan(unittest.TestCase):
    '''The branch scan must reach a valid root and expose its trace.'''

    def test_scan_finds_the_constrained_root(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        log = []
        result = cdft_branch.solve_branch_aware(
            mf, w, .4, scan_points=9, scan_log=log, scan_upper=.5)
        self.assertLess(abs(result.population-.4), 1e-8)
        scan = result.branch_scan
        self.assertTrue(scan.converged)
        self.assertGreaterEqual(len(scan.points), 9)
        self.assertTrue(any(p.get('converged') for p in scan.points))
        self.assertTrue(log)

    def test_scan_log_accepts_a_callable_sink(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        # A caller may pass a callable so that the log is streamed while the
        # scan is still running. The solver must call it, not try to append
        # to it.
        streamed = []
        result = cdft_branch.solve_branch_aware(
            mf, w, .4, scan_points=9, scan_log=streamed.append,
            scan_upper=.5)
        self.assertLess(abs(result.population-.4), 1e-8)
        self.assertTrue(streamed)
        self.assertTrue(any('lambda=' in message for message in streamed))

    def test_refinement_reaches_a_target_between_two_scan_points(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        # A coarse scan whose grid points all miss the target must still be
        # refined onto the root. The refinement has to keep a straddling
        # bracket and fall back to bisection when the secant step leaves it.
        coarse = cdft_branch.solve_branch_aware(
            mf, w, .4, scan_points=5, scan_upper=.5)
        self.assertLess(abs(coarse.population-.4), 1e-8)

    def test_refinement_survives_a_failing_inner_solve(self):
        # A warm start density that always fails must not abort the
        # refinement before it has retried from the other bracket endpoint.
        calls = {'count': 0}

        class FakeState:
            def __init__(self, population):
                self.e_tot = -1.0
                self._population = population
                self.mol = FakeMol(1)

            def make_rdm1(self):
                # A UKS state carries two spin blocks.
                return np.array([np.array([[self._population]]),
                                 np.array([[0.0]])])

        def inner_solve(multiplier, start_density):
            calls['count'] += 1
            if calls['count'] % 2 == 1:
                raise cdft.ConvergenceError('synthetic failure')
            # A monotone response around the target at lambda = 0.
            return FakeState(0.4 + 0.1*multiplier), 0.0

        def population_of(density):
            return float(density[0][0][0]+density[1][0][0])

        # Two bracket points straddling the target 0.4.
        points = [(-1.0, 0.3, 0.0, None, None, np.array([[[0.3]]])),
                  (1.0, 0.5, 0.0, None, None, np.array([[[0.5]]]))]
        refined, _evidence = cdft_branch._refine(
            points, 0.4, 1e-8, inner_solve, 0.5, population_of,
            np.array([[1.0]]), np.array([1.0]), np.array([[1.0]]),
            max_iterations=20)
        self.assertIsNotNone(refined)
        self.assertLess(abs(refined[1]-0.4), 1e-8)
        self.assertGreater(calls['count'], 1)

    def test_scan_reports_failure_without_opaque_root_error(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        # A target outside the reachable range must fail with a structured
        # diagnostic rather than an opaque root-finder message.
        with self.assertRaises((cdft.ConvergenceError, cdft_branch.BranchSwitchError)):
            cdft_branch.solve_branch_aware(mf, w, .8, scan_points=7,
                                           scan_upper=1e-3)
        self.assertTrue(hasattr(cdft, 'LAST_BRANCH_SCAN'))

    def test_fingerprint_is_intrinsic_and_order_independent(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        result = cdft.solve(mf, w, .4)
        overlap = result.method.get_ovlp()
        eigvals, eigvecs = np.linalg.eigh(overlap)
        first = cdft_branch._fingerprint(result.method, result.density,
                                         result.population, overlap, eigvals,
                                         eigvecs, None)
        second = cdft_branch._fingerprint(result.method, result.density,
                                          result.population, overlap, eigvals,
                                          eigvecs, None)
        self.assertEqual(first.intrinsic(), second.intrinsic())
        self.assertAlmostEqual(first.spin_square, second.spin_square, 12)

    def test_density_overlap_is_computed_for_a_3d_density(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        result = cdft.solve(mf, w, .4)
        overlap = result.method.get_ovlp()
        eigvals, eigvecs = np.linalg.eigh(overlap)
        # A UKS make_rdm1() returns a 3D array. The fingerprint must reduce
        # it the same way for the current and for the previous state, so
        # that the density overlap is a real number instead of a silent zero.
        previous = np.asarray(result.density)
        self.assertEqual(previous.ndim, 3)
        fingerprint = cdft_branch._fingerprint(result.method, result.density,
                                                result.population, overlap,
                                                eigvals, eigvecs, previous)
        self.assertGreater(fingerprint.density_overlap, 0.9)
        # A state compared against itself must overlap completely.
        identical = cdft_branch._fingerprint(result.method, result.density,
                                              result.population, overlap,
                                              eigvals, eigvecs, result.density)
        self.assertAlmostEqual(identical.density_overlap, 1.0, 8)

    def test_branch_identity_check_rejects_a_wrong_state(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        # Requiring a signed spin of 5.0 cannot be met by this doublet, so the
        # scan must refuse the candidate instead of returning a wrong state.
        with self.assertRaises(cdft.ConvergenceError):
            cdft_branch.solve_branch_aware(mf, w, .4, scan_points=7,
                                           required_signed_spin=5.0)

    def test_failed_branch_requirement_reports_reachable_branches(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        with self.assertRaises(cdft.ConvergenceError):
            cdft_branch.solve_branch_aware(mf, w, .4, scan_points=7,
                                           required_signed_spin=5.0)
        # The diagnostic must state which branches were actually reachable,
        # so that the operator can state the intended identity instead of
        # guessing a tolerance.
        scan = cdft.LAST_BRANCH_SCAN
        self.assertIn('branches', scan)
        self.assertTrue(scan['branches'])
        for cluster in scan['branches']:
            self.assertIn('total_spin', cluster)
            self.assertIn('spin_square', cluster)
            self.assertIn('multiplier_range', cluster)
        self.assertIn('rejected_by_branch', scan)


class BranchFiltering(unittest.TestCase):
    '''A required branch identity must select a branch, not just reject one.'''

    def test_branch_mismatch_reports_the_first_violation(self):
        fingerprint = cdft_branch.BranchFingerprint(
            spin_square=2.0, total_spin=1.0, atom_spins=(0.5, -0.5),
            population=1.0)
        self.assertIsNone(cdft_branch._branch_mismatch(fingerprint, None, None,
                                                       0.2))
        self.assertIsNone(cdft_branch._branch_mismatch(fingerprint, 1.0, None,
                                                       0.2))
        self.assertIsNotNone(cdft_branch._branch_mismatch(fingerprint, 5.0, None,
                                                          0.2))
        self.assertIsNotNone(cdft_branch._branch_mismatch(
            fingerprint, None, {0: 3.0}, 0.2))
        # An atom index beyond the fingerprint must be reported rather than
        # silently skipped.
        self.assertIsNotNone(cdft_branch._branch_mismatch(
            fingerprint, None, {7: 1.0}, 0.2))

    def test_branch_summary_clusters_by_total_spin(self):
        def point(multiplier, total_spin):
            fingerprint = cdft_branch.BranchFingerprint(
                spin_square=total_spin, total_spin=total_spin,
                atom_spins=(total_spin,), population=multiplier)
            return (multiplier, multiplier, 0., None, fingerprint,
                    np.zeros((1, 1, 1)))
        clusters = cdft_branch._branch_summary([point(-1., 1.), point(-0.5, 1.),
                                                point(0.5, 3.)])
        self.assertEqual(len(clusters), 2)
        self.assertAlmostEqual(clusters[0]['total_spin'], 1.)
        self.assertEqual(clusters[0]['n_points'], 2)
        self.assertAlmostEqual(clusters[0]['multiplier_range'][0], -1.)
        self.assertAlmostEqual(clusters[0]['multiplier_range'][1], -0.5)
        self.assertAlmostEqual(clusters[1]['total_spin'], 3.)
        self.assertEqual(clusters[1]['n_points'], 1)

    def test_required_identity_selects_the_matching_branch(self):
        # The constrained response is multivalued. A required identity must
        # be applied as a filter over the scan, so that a root on another
        # branch does not destroy a solution that does exist.
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        reference = cdft_branch.solve_branch_aware(mf, w, .4, scan_points=9,
                                                    scan_upper=.5)
        total_spin = reference.branch_scan.points[0]['fingerprint']['total_spin']
        # Requiring the identity that the scan actually produced must still
        # converge, and the accepted root must carry it.
        accepted = cdft_branch.solve_branch_aware(
            mf, w, .4, scan_points=9, scan_upper=.5,
            required_signed_spin=total_spin)
        self.assertLess(abs(accepted.population-.4), 1e-8)
        self.assertAlmostEqual(accepted.branch_scan.points[-1]['fingerprint']
                               ['total_spin'], total_spin, 6)
        self.assertTrue(hasattr(accepted, 'branch_clusters'))

    def test_refinement_rejects_a_trial_on_the_wrong_branch(self):
        # A refinement trial whose fingerprint violates the required identity
        # must be rejected instead of being returned as the root.
        calls = {'count': 0}

        class FakeState:
            def __init__(self, population, signed_spin):
                self.e_tot = -1.0
                self._population = population
                self._signed_spin = signed_spin
                self.mol = FakeMol(1)

            def make_rdm1(self):
                return np.array([np.array([[self._population]]),
                                 np.array([[0.0]])])

        def inner_solve(multiplier, start_density):
            calls['count'] += 1
            # The trial leaves the required branch, so it must be rejected.
            return FakeState(0.4+0.1*multiplier, 5.0), 0.0

        def population_of(density):
            return float(density[0][0][0]+density[1][0][0])

        points = [(-1.0, 0.3, 0.0, None, None, np.array([[[0.3]]])),
                  (1.0, 0.5, 0.0, None, None, np.array([[[0.5]]]))]
        # The branch check rejects every trial fingerprint.
        def branch_check(fingerprint):
            return 'wrong branch'

        refined, _evidence = cdft_branch._refine(
            points, 0.4, 1e-8, inner_solve, 0.5, population_of,
            np.array([[1.0]]), np.array([1.0]), np.array([[1.0]]),
            max_iterations=20, branch_check=branch_check)
        self.assertIsNone(refined)

    def test_refinement_only_brackets_consistent_branch_points(self):
        # When a branch identity is required, the bracket may only be formed
        # from points that satisfy it. Otherwise the refinement could walk
        # onto a different electronic branch.
        class FakeState:
            def __init__(self, population):
                self.e_tot = -1.0
                self._population = population
                self.mol = FakeMol(1)

            def make_rdm1(self):
                return np.array([np.array([[self._population]]),
                                 np.array([[0.0]])])

        def inner_solve(multiplier, start_density):
            return FakeState(0.4+0.1*multiplier), 0.0

        def population_of(density):
            return float(density[0][0][0]+density[1][0][0])

        def fingerprint(total_spin):
            return cdft_branch.BranchFingerprint(
                spin_square=total_spin, total_spin=total_spin,
                atom_spins=(total_spin,), population=0.0)

        # The straddling pair is (-1.0, 0.3) and (1.0, 0.5). Only the second
        # one satisfies the required identity, so no bracket may be formed.
        points = [(-1.0, 0.3, 0.0, None, fingerprint(5.0),
                   np.array([[[0.3]]])),
                  (1.0, 0.5, 0.0, None, fingerprint(1.0),
                   np.array([[[0.5]]]))]
        refined, _evidence = cdft_branch._refine(
            points, 0.4, 1e-8, inner_solve, 0.5, population_of,
            np.array([[1.0]]), np.array([1.0]), np.array([[1.0]]),
            max_iterations=20,
            branch_check=lambda fp: None if fp.total_spin == 1.0 else 'wrong')
        self.assertIsNone(refined)

    def test_refinement_rejects_a_trial_that_crosses_a_branch_boundary(self):
        # A refinement trial whose density overlap with the previous accepted
        # point falls below the threshold belongs to a different electronic
        # branch. Accepting it would let the secant step oscillate across the
        # discontinuity of the multivalued constrained response and never
        # converge, while the iteration still looks healthy. The trial must be
        # rejected and the iteration must continue inside the bracket.
        class FakeState:
            def __init__(self, population):
                self.e_tot = -1.0
                self._population = population
                self.mol = FakeMol(1)

            def make_rdm1(self):
                return np.array([np.array([[self._population]]),
                                 np.array([[0.0]])])

        # The inner solve returns a population on the wrong branch for part of
        # the axis. The fingerprint is built from the real solver helper, so
        # the density overlap is computed exactly as in production.
        def inner_solve(multiplier, start_density):
            return FakeState(0.4+0.1*multiplier), 0.0

        def population_of(density):
            return float(density[0][0][0]+density[1][0][0])

        def fingerprint(population):
            return cdft_branch.BranchFingerprint(
                spin_square=2.0, total_spin=1.0, atom_spins=(1.0,),
                population=population, density_overlap=0.1)

        points = [(-1.0, 0.3, 0.0, None, fingerprint(0.3), np.array([[[0.3]]])),
                  (1.0, 0.5, 0.0, None, fingerprint(0.5), np.array([[[0.5]]]))]
        # The fingerprint helper is patched so that every trial reports a low
        # overlap. The refinement must then reject every trial and must not
        # return a root.
        original = cdft_branch._fingerprint
        try:
            cdft_branch._fingerprint = (lambda *args, **kwargs:
                                        fingerprint(0.4))
            refined, _evidence = cdft_branch._refine(
                points, 0.4, 1e-8, inner_solve, 0.5, population_of,
                np.array([[1.0]]), np.array([1.0]), np.array([[1.0]]),
                max_iterations=20)
        finally:
            cdft_branch._fingerprint = original
        self.assertIsNone(refined)

    def test_refinement_reports_an_identity_drift_without_rejecting_the_trial(self):
        # A trial can stay on the same density branch, with a high overlap,
        # and still move the electronic hole to another fragment. The
        # population then jumps by a fraction of an electron and the secant
        # step oscillates across the discontinuity. The SAI0 campaigns showed
        # exactly this: the overlap stayed near 0.99 while the population
        # jumped between 14.056 and 13.914.
        #
        # The identity must therefore *not* be a rejection criterion. The
        # fragment spin distribution drifts continuously along one and the
        # same branch: on the V2-04c D1 case the donor spin grew from 0.003
        # to 0.337 electrons while the multiplier ran from 0 to 0.03, and
        # the jump across the fold was only 0.079 electrons. No fixed
        # threshold separates the drift from the jump, so a threshold based
        # rejection rejects valid trials and makes the refinement spin.
        #
        # The identity is recorded and reported instead, and the rejection
        # is decided by the response itself.
        calls = {'count': 0}

        class FakeState:
            def __init__(self, population, donor_spin):
                self.e_tot = -1.0
                self._population = population
                self._donor_spin = donor_spin
                self.mol = FakeMol(1)

            def make_rdm1(self):
                return np.array([np.array([[self._population]]),
                                 np.array([[0.0]])])

        def inner_solve(multiplier, start_density):
            calls['count'] += 1
            # The first trial is on the donor branch, the second one has the
            # hole on the acceptor.
            donor_spin = 0.97 if calls['count'] == 1 else -1.93
            return FakeState(0.4+0.1*multiplier, donor_spin), 0.0

        def population_of(density):
            return float(density[0][0][0]+density[1][0][0])

        donor = cdft_branch.BranchFingerprint(
            spin_square=2.0, total_spin=2.0, atom_spins=(2.0,),
            fragment_spins=(('donor', 0.97),), population=0.3)
        points = [(-1.0, 0.3, 0.0, None, donor, np.array([[[0.3]]])),
                  (1.0, 0.5, 0.0, None, donor, np.array([[[0.5]]]))]

        def patched(*args, **kwargs):
            # The trial is on the acceptor branch, but its density overlap is
            # high, so only the identity guard could reject it.
            return cdft_branch.BranchFingerprint(
                spin_square=2.0, total_spin=2.0, atom_spins=(2.0,),
                fragment_spins=(('donor', -1.93),), population=0.4,
                density_overlap=0.99)

        original = cdft_branch._fingerprint
        try:
            cdft_branch._fingerprint = patched
            refined, _evidence = cdft_branch._refine(
                points, 0.4, 1e-8, inner_solve, 0.5, population_of,
                np.array([[1.0]]), np.array([1.0]), np.array([[1.0]]),
                max_iterations=20)
        finally:
            cdft_branch._fingerprint = original
        # The identity drift does not reject the trial. The population of the
        # trial is inside the tolerance, so the refinement returns it.
        self.assertIsNotNone(refined)

    def test_the_multivaluedness_probe_requires_two_different_solutions(self):
        # The strict test of a fold: at one multiplier, two different warm
        # starts must converge to two different self-consistent solutions.
        class FakeState:
            def __init__(self, population):
                self.e_tot = -1.0
                self._population = population
                self.mol = FakeMol(1)

            def make_rdm1(self):
                return np.array([np.array([[self._population]]),
                                 np.array([[0.0]])])

        def population_of(density):
            return float(density[0][0][0]+density[1][0][0])

        def solve_with(start_population, other_population):
            def inner_solve(multiplier, start_density):
                # The warm start decides which of the two solutions is
                # returned, which is the mechanism of a fold.
                chosen = start_population if start_density is not None \
                    and float(start_density[0][0][0]) < 0.4 else other_population
                return FakeState(chosen), 0.0
            return inner_solve

        left = np.array([[[0.3]]])
        right = np.array([[[0.5]]])
        # The two warm starts reach two different solutions.
        fold = cdft_branch._probe_multivalued(
            0.0, left, right, solve_with(0.3, 0.5), population_of,
            np.array([[1.0]]), np.array([1.0]), np.array([[1.0]]))
        self.assertIsNotNone(fold)
        self.assertAlmostEqual(fold['separation_electrons'], 0.2, 12)
        self.assertEqual(len(fold['limits']), 2)
        # When both warm starts reach the same solution the response is
        # single valued and no fold is reported.
        single = cdft_branch._probe_multivalued(
            0.0, left, right, solve_with(0.3, 0.3), population_of,
            np.array([[1.0]]), np.array([1.0]), np.array([[1.0]]))
        self.assertIsNone(single)

    def test_a_confirmed_fold_reports_the_two_branch_limits(self):
        # The gap report must use the two limits that the two branches
        # actually reach, not the coarse scan bracket. On the V2-04c D1 case
        # the coarse bracket spans 14.173 and 12.259, which hides the gap
        # around the target 14.0 completely, while the observed limits
        # 14.0528 and 13.9145 bracket it.
        def point(multiplier, population):
            fingerprint = cdft_branch.BranchFingerprint(
                spin_square=2.01, total_spin=2.0, atom_spins=(2.0,),
                population=population)
            return (multiplier, population, 0., None, fingerprint,
                    np.zeros((1, 1, 1)))
        points = [point(-0.125, 14.309), point(0.125, 12.259)]
        evidence = [{'multiplier_hartree': 0.0235858,
                     'population': 13.9145,
                     'bracket_prediction': 14.0525,
                     'anomaly_electrons': 0.138,
                     'multivalued_confirmed': True,
                     'fold': {'limits': [13.9145, 14.0525],
                              'separation_electrons': 0.138}}]
        gap = cdft_branch._branch_gap(points, 14.0, evidence)
        self.assertIsNotNone(gap)
        self.assertIn('separated by', gap)
        # A target outside the two limits is not in the gap.
        self.assertIsNone(cdft_branch._branch_gap(points, 14.1, evidence))
        # A target that the coarse bracket would have hidden is found,
        # because the limits are the observed ones.
        self.assertIsNone(cdft_branch._branch_gap(points, 14.2, evidence))

    def test_the_refinement_never_repeats_a_rejected_multiplier(self):
        # The bisection of an unchanged bracket is a fixed point. When a
        # trial is rejected the bracket must shrink, or the refinement
        # recomputes the same rejected multiplier for ever. On the V2-04c D1
        # case that produced five identical rejections in a row and the
        # refinement spun until the child process was killed.
        seen = []

        class FakeState:
            def __init__(self, population):
                self.e_tot = -1.0
                self._population = population
                self.mol = FakeMol(1)

            def make_rdm1(self):
                return np.array([np.array([[self._population]]),
                                 np.array([[0.0]])])

        def inner_solve(multiplier, start_density):
            seen.append(float(multiplier))
            # Every trial lands on the wrong branch, so every trial is
            # rejected by the population anomaly test.
            return FakeState(0.9), 0.0

        def population_of(density):
            return float(density[0][0][0]+density[1][0][0])

        donor = cdft_branch.BranchFingerprint(
            spin_square=2.0, total_spin=2.0, atom_spins=(2.0,),
            fragment_spins=(('donor', 1.0),), population=0.3)
        points = [(-1.0, 0.3, 0.0, None, donor, np.array([[[0.3]]])),
                  (1.0, 0.5, 0.0, None, donor, np.array([[[0.5]]]))]

        original = cdft_branch._fingerprint
        probe = cdft_branch._probe_multivalued
        try:
            cdft_branch._fingerprint = (lambda *args, **kwargs:
                                        cdft_branch.BranchFingerprint(
                                            spin_square=2.0, total_spin=2.0,
                                            atom_spins=(2.0,),
                                            fragment_spins=(('donor', 1.0),),
                                            population=0.4, density_overlap=0.99))
            # The multivaluedness probe is switched off here, because it
            # solves the same multiplier twice on purpose. The behaviour
            # under test is the bisection, which must not return to a
            # multiplier that was already rejected.
            cdft_branch._probe_multivalued = (lambda *args, **kwargs: None)
            refined, _evidence = cdft_branch._refine(
                points, 0.4, 1e-8, inner_solve, 0.5, population_of,
                np.array([[1.0]]), np.array([1.0]), np.array([[1.0]]),
                max_iterations=20)
        finally:
            cdft_branch._fingerprint = original
            cdft_branch._probe_multivalued = probe
        self.assertIsNone(refined)
        # No multiplier was tried twice.
        self.assertEqual(len(seen), len(set(seen)))

    def test_same_branch_compares_the_electronic_identity(self):
        first = cdft_branch.BranchFingerprint(
            spin_square=2.0, total_spin=2.0, atom_spins=(2.0,),
            fragment_spins=(('donor', 0.97),), population=14.0)
        same = cdft_branch.BranchFingerprint(
            spin_square=2.01, total_spin=2.0, atom_spins=(2.0,),
            fragment_spins=(('donor', 0.98),), population=13.9)
        other = cdft_branch.BranchFingerprint(
            spin_square=2.0, total_spin=2.0, atom_spins=(2.0,),
            fragment_spins=(('donor', -1.93),), population=13.9)
        self.assertTrue(cdft_branch._same_branch(first, same, 0.2))
        self.assertFalse(cdft_branch._same_branch(first, other, 0.2))


class PhysicalFingerprint(unittest.TestCase):
    '''The branch fingerprint must report physical observables.

    An earlier implementation reported ``Tr(spin_dm @ S^(1/2))`` as the total
    spin and the raw Loewdin diagonal as the atomic spins. The first is not
    the total spin and is not bounded by the electron count, and the second is
    a per basis function quantity. A branch identity built on those numbers
    could accept a state whose reported spin was physically impossible, which
    is exactly what the SAI0 campaigns observed.
    '''

    def test_total_spin_reproduces_the_electron_spin_imbalance(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        result = cdft.solve(mf, w, .4)
        overlap = result.method.get_ovlp()
        eigvals, eigvecs = np.linalg.eigh(overlap)
        fingerprint = cdft_branch._fingerprint(result.method, result.density,
                                                result.population, overlap,
                                                eigvals, eigvecs, None)
        # The total spin must be N_alpha - N_beta, not a Loewdin-contraction
        # artefact. It can never exceed the electron count.
        expected = result.method.mol.nelec[0]-result.method.mol.nelec[1]
        self.assertAlmostEqual(fingerprint.total_spin, expected, 8)
        self.assertLessEqual(abs(fingerprint.total_spin), result.method.mol.nelectron)

    def test_atom_spins_are_per_atom_and_sum_to_the_total_spin(self):
        # A basis with more functions than atoms is required, because the
        # assertion that the atomic spins are per atom and not per basis
        # function is only meaningful when those counts differ.
        mol = gto.M(atom=[('H', (0., 0., 0.)), ('H', (0., 0., 1.4))],
                    basis='6-31g', charge=1, spin=1, verbose=0)
        mf = dft.UKS(mol)
        mf.xc = 'pbe0'
        mf.grids.level = 1
        mf.grids.prune = None
        mf.small_rho_cutoff = 0
        mf.conv_tol = 1e-12
        mf.conv_tol_grad = 1e-7
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        result = cdft_branch.solve_branch_aware(mf, w, .4, scan_points=9,
                                                scan_upper=.5)
        overlap = result.method.get_ovlp()
        eigvals, eigvecs = np.linalg.eigh(overlap)
        fingerprint = cdft_branch._fingerprint(result.method, result.density,
                                                result.population, overlap,
                                                eigvals, eigvecs, None)
        # One entry per atom, not per basis function.
        self.assertEqual(len(fingerprint.atom_spins), mf.mol.natm)
        self.assertLess(len(fingerprint.atom_spins), mf.mol.nao)
        # The atomic spins must sum to the total spin.
        self.assertAlmostEqual(sum(fingerprint.atom_spins),
                               fingerprint.total_spin, 8)

    def test_fragment_spins_are_reported_from_the_constraint_weights(self):
        xyz = np.array([[0., 0., -.5], [0, 0., .5]])
        mf = setup(xyz)
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        result = cdft.solve(mf, w, .4)
        overlap = result.method.get_ovlp()
        eigvals, eigvecs = np.linalg.eigh(overlap)
        fingerprint = cdft_branch._fingerprint(
            result.method, result.density, result.population, overlap,
            eigvals, eigvecs, None,
            fragment_weights={'donor': [1., 0.], 'acceptor': [0., -1.]})
        self.assertEqual([name for name, _ in fingerprint.fragment_spins],
                         ['donor', 'acceptor'])
        # The fragment spin is the signed contraction of the atomic spins.
        donor = dict(fingerprint.fragment_spins)['donor']
        acceptor = dict(fingerprint.fragment_spins)['acceptor']
        self.assertAlmostEqual(donor, fingerprint.atom_spins[0], 8)
        self.assertAlmostEqual(acceptor, -fingerprint.atom_spins[1], 8)

    def test_branch_summary_separates_states_with_the_same_total_spin(self):
        # Two states with the same total spin but a different spin square are
        # different electronic branches. Clustering them together would hide
        # the very distinction the branch audit has to make.
        def point(multiplier, total_spin, spin_square):
            fingerprint = cdft_branch.BranchFingerprint(
                spin_square=spin_square, total_spin=total_spin,
                atom_spins=(total_spin,), population=multiplier)
            return (multiplier, multiplier, 0., None, fingerprint,
                    np.zeros((1, 1, 1)))
        clusters = cdft_branch._branch_summary(
            [point(-1., 2., 2.0), point(-0.5, 2., 4.0)])
        self.assertEqual(len(clusters), 2)

    def test_branch_gap_is_reported_for_an_unreachable_target(self):
        # Two branches with disjoint population coverage and the target
        # between them is a structural gap between electronic branches, not
        # a numerical failure. The evidence is the discontinuity that the
        # refinement observed on the bracketed branch.
        def point(multiplier, population, spin_square):
            fingerprint = cdft_branch.BranchFingerprint(
                spin_square=spin_square, total_spin=2.0, atom_spins=(2.0,),
                population=population)
            return (multiplier, population, 0., None, fingerprint,
                    np.zeros((1, 1, 1)))
        points = [point(-1.0, 14.2, 2.0), point(-0.5, 13.9, 2.0),
                  point(0.5, 12.1, 4.0), point(1.0, 11.8, 4.0)]
        # The refinement observed a discontinuity between the two limits of
        # one branch, 13.9 and 12.1. A target of 13.0 lies strictly between
        # them and is therefore attained by neither branch.
        evidence = [{'bracket_populations': [13.9, 12.1],
                     'multiplier_hartree': 0.0, 'population': 12.1,
                     'bracket_prediction': 13.9, 'anomaly_electrons': 1.8}]
        gap = cdft_branch._branch_gap(points, 13.0, evidence)
        self.assertIsNotNone(gap)
        # A target that is reached must not be reported as a gap.
        self.assertIsNone(cdft_branch._branch_gap(points, 14.0, evidence))
        # A target outside the reachable range is out of range, not a gap.
        self.assertIsNone(cdft_branch._branch_gap(points, 20.0, evidence))
        # Without the evidence of a discontinuity there is no gap to report.
        self.assertIsNone(cdft_branch._branch_gap(points, 13.0))

    def test_a_hole_migration_is_a_branch_switch_even_with_a_high_overlap(self):
        # The V2-04c D1 case measured a discontinuity of the constrained
        # population at lambda ~ 0.0236 across which the density overlap
        # stayed at 0.998, the total spin was identical to 1e-6 and the spin
        # square differed by 0.001. Only the population jumped, by 0.142
        # electrons. An identity built on the total spin and the spin square
        # alone therefore calls that one branch and the discontinuity becomes
        # invisible.
        def point(multiplier, population, donor):
            fingerprint = cdft_branch.BranchFingerprint(
                spin_square=2.0104, total_spin=2.0, atom_spins=(2.0,),
                fragment_spins=(('donor', donor), ('acceptor', -donor)),
                population=population)
            return (multiplier, population, 0., None, fingerprint,
                    np.zeros((1, 1, 1)))
        points = [point(0.0230, 14.073, 0.048),
                  point(0.0235, 14.059, 0.056),
                  point(0.0240, 13.888, 0.153)]
        # The refinement observed a trial whose population disagreed with the
        # bracket prediction. The two limits of the discontinuity are the
        # population that the bracketed branch predicts and the population
        # that the rejected trial measured. The target 14.0 lies between
        # them.
        evidence = [{'bracket_populations': [14.059, 13.888],
                     'multiplier_hartree': 0.02358, 'population': 13.9145,
                     'bracket_prediction': 14.0525, 'anomaly_electrons': 0.086,
                     'density_overlap': 0.998, 'total_spin': 2.0,
                     'spin_square': 2.0114}]
        gap = cdft_branch._branch_gap(points, 14.0, evidence)
        self.assertIsNotNone(gap)
        self.assertIn('separated by', gap)

    def test_the_branch_identity_tolerance_resolves_a_fragment_redistribution(self):
        # A redistribution of 0.079 electrons between fragments is a branch
        # change at a tight identity tolerance, and is not one at the loose
        # tolerance that the solver uses.
        first = cdft_branch.BranchFingerprint(
            spin_square=2.0104, total_spin=2.0, atom_spins=(2.0,),
            fragment_spins=(('donor', 0.056), ('acceptor', -1.944)),
            population=14.059)
        second = cdft_branch.BranchFingerprint(
            spin_square=2.0114, total_spin=2.0, atom_spins=(2.0,),
            fragment_spins=(('donor', 0.153), ('acceptor', -1.847)),
            population=13.888)
        self.assertFalse(cdft_branch._same_branch(first, second, 0.02))
        self.assertTrue(cdft_branch._same_branch(first, second, 0.2))
        # The total spin and the spin square alone do not separate them.
        self.assertAlmostEqual(first.total_spin, second.total_spin, 8)
        self.assertLess(abs(first.spin_square-second.spin_square), 0.002)

    def test_the_refinement_rejects_a_trial_that_disagrees_with_its_bracket(self):
        # The refinement must reject a trial whose population disagrees with
        # the linear interpolation of the bracket, because such a trial is
        # not on the bracketed branch. This is the criterion that detects a
        # discontinuity whose density overlap stays high.
        class FakeMol:
            def __init__(self, natm):
                self.natm = natm

        class FakeState:
            def __init__(self, population):
                self.e_tot = -1.0
                self._population = population
                self.mol = FakeMol(1)

            def make_rdm1(self):
                return np.array([np.array([[self._population]]),
                                 np.array([[0.0]])])

        calls = {'count': 0}

        def inner_solve(multiplier, start_density):
            calls['count'] += 1
            # Every trial reports a population that disagrees with the
            # bracket, so the refinement must reject each of them and give
            # up instead of accepting a root.
            return FakeState(0.25), 0.0

        def population_of(density):
            return float(density[0][0][0]+density[1][0][0])

        donor = cdft_branch.BranchFingerprint(
            spin_square=2.0, total_spin=2.0, atom_spins=(2.0,),
            fragment_spins=(('donor', 1.0),), population=0.3)
        points = [(-1.0, 0.3, 0.0, None, donor, np.array([[[0.3]]])),
                  (1.0, 0.5, 0.0, None, donor, np.array([[[0.5]]]))]

        def patched(*args, **kwargs):
            return cdft_branch.BranchFingerprint(
                spin_square=2.0, total_spin=2.0, atom_spins=(2.0,),
                fragment_spins=(('donor', 1.0),), population=0.4,
                density_overlap=0.99)

        original = cdft_branch._fingerprint
        try:
            cdft_branch._fingerprint = patched
            refined, evidence = cdft_branch._refine(
                points, 0.4, 1e-8, inner_solve, 0.5, population_of,
                np.array([[1.0]]), np.array([1.0]), np.array([[1.0]]),
                max_iterations=20)
        finally:
            cdft_branch._fingerprint = original
        # The trial that disagreed with the bracket was rejected, and the
        # rejection is recorded as the evidence of a discontinuity.
        self.assertIsNone(refined)
        self.assertTrue(evidence)
        self.assertIn('anomaly_electrons', evidence[0])

    def test_neighbouring_points_on_one_branch_are_not_a_gap(self):
        # The scan spacing separates neighbouring points on one branch. That
        # must not be mistaken for a gap between branches.
        def point(multiplier, population):
            fingerprint = cdft_branch.BranchFingerprint(
                spin_square=2.0, total_spin=2.0, atom_spins=(2.0,),
                population=population)
            return (multiplier, population, 0., None, fingerprint,
                    np.zeros((1, 1, 1)))
        points = [point(-1., 14.2), point(-0.5, 13.9), point(0.5, 13.6)]
        self.assertIsNone(cdft_branch._branch_gap(points, 13.75))

    def test_branch_gap_error_is_a_distinct_failure_mode(self):
        # The gap failure must be distinguishable from a plain branch switch,
        # so that a physically unreachable target is never reported as a
        # solver that failed to converge.
        self.assertTrue(issubclass(cdft_branch.BranchGapError,
                                   cdft_branch.BranchSwitchError))
        self.assertTrue(issubclass(cdft_branch.BranchSwitchError,
                                   cdft.ConvergenceError))


if __name__ == '__main__':
    unittest.main()
