#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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

import unittest
import numpy
from pyscf import lib, gto

def make_ab(n):
    numpy.random.seed(1)
    a = numpy.random.random((n,n)) + numpy.eye(n) * 2
    b = numpy.random.random(n)

    adiag = a.diagonal()
    x0 = b / adiag
    arest = a - numpy.diag(adiag)
    return a, b, adiag, arest, x0

class KnownValues(unittest.TestCase):
    def test_with_errvec(self):
        a, b, adiag, arest, x = make_ab(16)
        ad = lib.diis.DIIS()
        for i in range(20):
            x = (b - arest.dot(x)) / adiag
            x = ad.update(x)
        self.assertAlmostEqual(abs(a.dot(x) - b).max(), 0, 6)
        self.assertAlmostEqual(abs(x - numpy.linalg.solve(a,b)).max(), 0, 6)

    def test_without_errvec(self):
        a, b, adiag, arest, x = make_ab(16)
        ad = lib.diis.DIIS()
        for i in range(20):
            e = b - a.dot(x)
            x = (b - arest.dot(x)) / adiag
            x = ad.update(x, xerr=e)
        self.assertAlmostEqual(abs(a.dot(x) - b).max(), 0, 6)
        self.assertAlmostEqual(abs(x - numpy.linalg.solve(a,b)).max(), 0, 6)

    def test_restore(self):
        a, b, adiag, arest, x = make_ab(16)
        lib.diis.INCORE_SIZE, bak = 4, lib.diis.INCORE_SIZE
        ftmp = lib.NamedTemporaryFile()
        ad = lib.diis.DIIS(filename=ftmp.name)
        for i in range(8):
            x = (b - arest.dot(x)) / adiag
            x = ad.update(x)

        ad = lib.diis.DIIS().restore(ftmp.name, inplace=False)
        x = ad.extrapolate()
        for i in range(12):
            e = b - a.dot(x)
            x = (b - arest.dot(x)) / adiag
            x = ad.update(x, xerr=e)
        lib.diis.INCORE_SIZE = bak
        self.assertAlmostEqual(abs(a.dot(x) - b).max(), 0, 6)
        self.assertAlmostEqual(abs(x - numpy.linalg.solve(a,b)).max(), 0, 6)

        ad = lib.diis.restore(ftmp.name)
        x = ad.extrapolate()
        for i in range(12):
            x = (b - arest.dot(x)) / adiag
            x = ad.update(x)
        self.assertAlmostEqual(abs(a.dot(x) - b).max(), 0, 6)
        self.assertAlmostEqual(abs(x - numpy.linalg.solve(a,b)).max(), 0, 6)

    def test_dependent_error_vectors(self):
        ad = lib.diis.DIIS()
        ad.update(numpy.array([1., 2.]), xerr=numpy.array([1., 2.]))
        result = ad.update(numpy.array([3., 4.]), xerr=numpy.array([1., 2.]))
        numpy.testing.assert_allclose(result, [2., 3.], rtol=0, atol=1e-12)

    def test_complex_dependent_error_vectors(self):
        ad = lib.diis.DIIS()
        error = numpy.array([1., 1j])
        ad.update(numpy.array([1.+1j, 2.-1j]), xerr=error)
        result = ad.update(numpy.array([3.+1j, 4.-1j]), xerr=error)
        numpy.testing.assert_allclose(result, [2.+1j, 3.-1j], rtol=0, atol=1e-12)

    def test_nonfinite_error_vectors_rejected(self):
        ad = lib.diis.DIIS()
        ad.update(numpy.array([1., 2.]), xerr=numpy.array([1., 2.]))
        with self.assertRaises(ValueError):
            ad.update(numpy.array([3., 4.]), xerr=numpy.array([numpy.nan, 2.]))

    def test_extrapolate(self):
        a, b, adiag, arest, x = make_ab(16)
        ad = lib.diis.DIIS()
        for i in range(20):
            e = b - a.dot(x)
            x = (b - arest.dot(x)) / adiag
            x = ad.update(x, xerr=e)

        x = ad.extrapolate(4)
        self.assertAlmostEqual(abs(a.dot(x) - b).max(), 0, 6)
        self.assertAlmostEqual(abs(x - numpy.linalg.solve(a,b)).max(), 0, 6)


if __name__ == "__main__":
    print("Full Tests for lib.diis")
    unittest.main()
