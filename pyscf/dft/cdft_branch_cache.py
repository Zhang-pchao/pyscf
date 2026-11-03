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

"""Process-local, target-independent scan identity and trace projection."""
import hashlib
import json
from copy import deepcopy

import numpy as np


def _scan_cache_key(constraint, method, axis, inner_solver, inner_grad_tol,
                    density_tol, scan_mode, fragment_weights, *, dm0=None,
                    inner_lindep=1e-24):
    """Conservative same-object identity plus numerical inputs.

    This is an in-memory cache, never a portable model identifier. The full
    molecule, Hamiltonian, overlap, integrals, grid, seed and SCF controls
    detect in-place changes. Independently decorated objects cannot share it.
    """
    h = hashlib.sha256()

    def add(name, value):
        h.update(name.encode() + b'\0')
        if isinstance(value, np.ndarray):
            h.update(str((value.shape, value.dtype)).encode())
            h.update(np.ascontiguousarray(value).tobytes())
        else:
            h.update(json.dumps(value, sort_keys=True, default=repr).encode())
        h.update(b'\0')

    add('object', id(method))
    add('constraint', np.asarray(constraint.matrix))
    add('weights', np.asarray(constraint.atom_weights))
    add('reference', getattr(constraint, 'reference_sha256', None))
    add('axis', np.asarray(axis))
    add('seed', None if dm0 is None else np.asarray(dm0))
    add('controls', [inner_solver, inner_grad_tol, inner_lindep, density_tol, scan_mode])
    add('fragments', fragment_weights)
    for name in ('xc', 'conv_tol', 'conv_tol_grad', 'max_cycle', 'direct_scf',
                 'direct_scf_tol', 'small_rho_cutoff', 'init_guess', 'level_shift',
                 'damp', 'diis', 'diis_space', 'diis_start_cycle', 'diis_damp',
                 'omega', 'max_memory'):
        add(name, getattr(method, name, None))
    mol = method.mol
    # Do not serialize lazy caches (e.g. energy_nuc initializes mol.enuc).
    add('molecule', [mol.natm, mol.nelectron, getattr(mol, 'nelec', None)])
    for name in ('_atm', '_bas', '_env', '_ecpbas', 'cart', 'charge', 'spin'):
        add('mol.' + name, getattr(mol, name, None))
    add('nelec', getattr(method, 'nelec', getattr(mol, 'nelec', None)))
    for name in ('get_hcore', 'get_ovlp', 'energy_nuc'):
        getter = getattr(method, name, None)
        add(name, getter() if getter is not None else None)
    add('eri', getattr(method, '_eri', None))
    for name in ('get_veff', 'get_jk', 'get_occ', 'get_init_guess'):
        callback = getattr(method, name, None)
        add(name, id(getattr(callback, '__func__', callback)))
    for name in ('level', 'prune', 'atom_grid', 'radi_method', 'becke_scheme',
                 'radii_adjust', 'atomic_radii', 'alignment', 'cutoff',
                 'coords', 'weights'):
        add('grid.' + name, getattr(method.grids, name, None))
    return h.hexdigest()


def project_points(points, target):
    """Independent target-dependent view; never mutate cached observations."""
    projected = deepcopy(points)
    for point in projected:
        if point.get('converged'):
            point['residual'] = float(point['population'] - target)
    return projected


def record_points(points, emit):
    """Replay measured populations with residuals for the current target."""
    for point in points:
        value = point['multiplier_hartree']
        if not point.get('converged'):
            emit('lambda=%+.12g FAILED (%s)' % (value, point['error']))
            continue
        fp = point['fingerprint']
        emit('lambda=%+.12g C=%+.12g residual=%+.6g <S^2>=%.6f overlap=%.6f'
             % (value, point['population'], point['residual'],
                fp['spin_square'], fp['density_overlap']))
