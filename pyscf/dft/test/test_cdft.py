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

import unittest
import numpy as np
from pyscf import gto, dft, qmmm
from pyscf.dft import cdft
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


class KnownValues(unittest.TestCase):
    def test_atomic_reference_spin_sum(self):
        mol = molecule([[0, 0, 0], [0, 0, 1]])
        atom, dm = cdft.atomic_references(mol)['H']
        self.assertEqual(dm.shape, (atom.nao, atom.nao))
        self.assertAlmostEqual(np.einsum('ij,ji', dm, atom.intor('int1e_ovlp')), 1., 10)

    def test_weight_moving_grid_and_ao_derivative(self):
        xyz = np.array([[.1, .2, -.6], [-.2, -.1, .6]])
        mf = setup(xyz)
        weight = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        dm = np.array([[.6, .1], [.1, .4]])
        derivative = weight.density_gradient(dm)
        np.testing.assert_allclose(derivative.sum(axis=0), 0, atol=1e-12)
        for atom, axis in np.ndindex(2, 3):
            h = 1e-4
            values = []
            for sign in [-1, 1]:
                moved = xyz.copy()
                moved[atom, axis] += sign*h
                candidate = setup(moved)
                moved_weight = cdft.Hirshfeld(candidate.mol, [1., -1.],
                                              candidate.grids, weight.references)
                values.append(moved_weight.population(dm))
            numerical = (values[1]-values[0])/(2*h)*param.BOHR
            self.assertAlmostEqual(derivative[atom, axis], numerical, 7)

    def test_physical_energy_and_full_gradient(self):
        xyz = np.array([[0., 0., -.5], [0., 0., .5]])
        mf = setup(xyz)
        mf = qmmm.mm_charge(mf, [[2., .3, .4]], [.2], unit='Angstrom')
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        result = cdft.solve(mf, w, .4)
        physical = mf.energy_tot(result.density)
        self.assertAlmostEqual(physical, result.energy, 10)
        self.assertLess(abs(result.residual), 1e-8)
        qm_gradient = result.gradients()
        mm_gradient = result.environment_gradients()
        np.testing.assert_allclose(qm_gradient.sum(axis=0)+mm_gradient.sum(axis=0),
                                   0, atol=2e-7)
        for h in [.001, .0005]:
            energies = []
            for sign in [-1, 1]:
                moved = xyz.copy()
                moved[0, 2] += sign*h
                other = qmmm.mm_charge(setup(moved), [[2., .3, .4]], [.2], unit='Angstrom')
                cw = cdft.Hirshfeld(other.mol, [1., -1.], other.grids, w.references)
                energies.append(cdft.solve(other, cw, .4).energy)
            numerical = (energies[1]-energies[0])/(2*h)*param.BOHR
            self.assertLess(abs(numerical-qm_gradient[0, 2]), 2e-6)

    def test_fail_closed_and_no_input_mutation(self):
        mf = setup([[0, 0, -.5], [0, 0, .5]])
        w = cdft.Hirshfeld(mf.mol, [1., -1.], mf.grids)
        hcore = mf.get_hcore()
        with self.assertRaises(cdft.ConvergenceError):
            cdft.solve(mf, w, .8, max_multiplier=1e-10)
        np.testing.assert_array_equal(mf.get_hcore(), hcore)
        with self.assertRaises(ValueError):
            cdft.solve(mf, w, 2.)


if __name__ == '__main__':
    unittest.main()

