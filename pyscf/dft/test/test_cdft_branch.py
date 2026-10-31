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


if __name__ == '__main__':
    unittest.main()
