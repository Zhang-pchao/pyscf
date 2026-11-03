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

"""Production-path regression of cache projection and bounded inference."""
import unittest

import numpy as np

from pyscf import dft, gto
from pyscf.dft import cdft, cdft_branch
from pyscf.dft.cdft_branch_cache import _scan_cache_key


class Maintenance(unittest.TestCase):
    def setup_model(self):
        mol = gto.M(atom="H 0 0 0; H 0 0 1", basis="sto-3g",
                    charge=1, spin=1, verbose=0)
        mf = dft.UKS(mol)
        mf.xc = "pbe"
        mf.grids.level = 0
        mf.grids.prune = None
        mf.small_rho_cutoff = 0
        mf.conv_tol = 1e-11
        mf.conv_tol_grad = 1e-7
        return mf, cdft.Hirshfeld(mol, [1., -1.], mf.grids)

    def test_cache_keys_include_seed_field_spin_and_solver_controls(self):
        mf, constraint = self.setup_model()
        axis = np.linspace(-.1, .1, 5)

        def key(**kw):
            return _scan_cache_key(constraint, mf, axis, "newton", 1e-8,
                                   .5, "uniform", {"all": [1., 1.]}, **kw)

        reference = key()
        self.assertEqual(reference, key())
        self.assertNotEqual(reference, key(dm0=np.zeros((2, 2, 2))))
        self.assertNotEqual(reference, key(inner_lindep=1e-20))
        mf.direct_scf_tol = 1e-15
        self.assertNotEqual(reference, key())
        reference = key()
        original = mf.get_hcore()
        mf.get_hcore = lambda *args: original + np.eye(2)*.01
        self.assertNotEqual(reference, key())
        reference = key()
        mf.mol.spin = -1
        self.assertNotEqual(reference, key())

    def test_real_pbe_cache_projects_new_target_without_mutating_first(self):
        mf, constraint = self.setup_model()
        cache = {}
        kwargs = dict(scan_points=5, scan_upper=.2, population_tol=1e-8,
                      scan_cache=cache)
        first = cdft_branch.solve_branch_aware(mf, constraint, .1, **kwargs)
        before = [dict(p) for p in first.iterations]
        log = []
        second = cdft_branch.solve_branch_aware(mf, constraint, .2,
                                               scan_log=log, **kwargs)
        self.assertTrue(any("SCAN REUSED" in line for line in log))
        self.assertEqual(first.iterations, before)
        self.assertIsInstance(second, cdft_branch.BranchResult)
        for point in second.iterations:
            if point["converged"]:
                self.assertAlmostEqual(point["residual"], point["population"]-.2)
        self.assertLess(abs(second.residual), 1e-8)
        self.assertEqual(second.branch_diagnostics, cdft.LAST_BRANCH_SCAN)
        self.assertIsNot(second.branch_diagnostics, cdft.LAST_BRANCH_SCAN)
        uncached = cdft_branch.solve_branch_aware(
            mf, constraint, .2, scan_points=5, scan_upper=.2, population_tol=1e-8)
        self.assertAlmostEqual(second.energy, uncached.energy, places=10)
        np.testing.assert_allclose(second.density, uncached.density, atol=1e-7, rtol=0)
        np.testing.assert_allclose(second.gradients(), uncached.gradients(), atol=1e-7, rtol=0)

    def test_smooth_multivalued_counterexample_calls_production_classifier(self):
        # A=-lambda, B=2-lambda; at zero populations 0,2 do not exclude C=1.
        evidence = [{"fold": {"limits": [0., 2.], "multiplier_hartree": 0.}}]
        observed = [(0., 0.), (0., 2.)]
        description = cdft_branch._branch_gap(observed, 1., evidence)
        self.assertIn("continuation remains unresolved", description)
        # Follow the same smooth A branch to lambda=-1: target reached.
        self.assertIsNone(cdft_branch._branch_gap(observed+[(-1., 1.)], 1., evidence))
        self.assertIn("untested", cdft_branch._branch_unreachable(evidence, 3.))


if __name__ == "__main__":
    unittest.main()
