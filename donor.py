""" Donor object for modelling single donors in a silicon lattice using emtpy

Author: Daniel Parker
"""

import emtpy

import numpy as np
import os
import matplotlib.pyplot as plt
import dill

from joblib import Memory

dir_path = os.path.dirname(os.path.realpath(__file__))
mem = Memory(cachedir = dir_path + '/caches/', verbose = 0)

# CONSTANTS
m0 = 9.10938215e-31 # Electron mass, [kg]
hbar = 1.054571726e-34 # hbar in [J] [s]
q = 1.6021765653e-19 # Elementary charge, [C]  
mt = 0.19
ml = 0.92
eps0 = 8.854e-12
epsb = 11.7 # XXX: 11.45 at low temperatures

donor_energies = {'P': [-45.59] + 3*[-33.89] + 2*[-32.58],\
                  'As':[-53.76] + 3*[-32.67] + 2*[-31.26],\
                  'Sb':[-42.74] + 3*[-32.90] + 2*[-30.47],\
                  'Bi':[-70.98] + 3*[-32.39] + 2*[-30.10]}
donor_A0 = {'P': 117.53, 'As': 198.35, 'Sb': 186.80, 'Bi': 1475.4} # MHz

states = ['A1'] + 3*['T2'] + 2*['E']
valleys = ['+x', '-x', '+y', '-y', '+z', '-z']

class Donor(object):
    def __init__(self, donor, material_path, basis_path, central_cell_path):
        """ Complete model setup. """
        if donor not in ['P', 'As', 'Sb', 'Bi']:
            raise RuntimeError('Only P, As, Sb and Bi donors are supported.')

        self.coulombPrefactor = q**2/(4*np.pi*eps0*epsb)*1e9/q*1e3 # ~ 123
        self.energies_exp = np.array(donor_energies[donor])
        self.A0_exp = donor_A0[donor]

        with open(material_path, 'rb') as f:
            self.material = dill.load(f)
        with open(central_cell_path, 'rb') as f:
            self.central_cell = dill.load(f)
        with open(basis_path, 'rb') as f:
            self.valley_basis = dill.load(f)

        self.setup_standard_model = mem.cache(self.setup_standard_model)

        self.geom = self.setup_standard_model()
        self.e_field = 0

    def setup_standard_model(self):
        """ Setup of classic emtpy donor model with free linear parameters. """
        geom = emtpy.geo()
        geom.addMaterial(None, None, self.material)
        geom.addElement(coulombPotential = (0.0,0.0,0.0), VOC = True,\
                        quadPoints = 100, prefactor = -self.coulombPrefactor)
        geom.addElement(elementBasis = self.central_cell, VOC = True)
        geom.addElement(listOfRoots = [[0],[],[]], prefactor = -1, \
                        xBounds = (None,None), descriptors = ['fieldTerm'])
        geom.addBasis(self.valley_basis, valleyIndex = 4)
        geom.autopopulateBasis()
        geom.initializeProblem()

        emtpy.evalMatrixElements(geom)
        emtpy.form3DMatrices(geom)

        return geom

    @property
    def energies(self):
        """ A getter for the energies at self.e_field electric field strength. """
        kwargs = {'potentialMin': -1e3, 'multDict': {'fieldTerm':self.e_field},\
                  'numEigVals': len(valleys)}
        energies, eig_vects, panic = emtpy.solveOneParticleHam(self.geom, **kwargs)
        return energies

    def valley_basis_coeffs(self, state, valley):
        """ A getter for the energies at self.e_field electric field strength. """
        kwargs = {'potentialMin': -1e3, 'multDict': {'fieldTerm':self.e_field},\
                  'numEigVals': len(valleys)}
        energies, eig_vects, panic = emtpy.solveOneParticleHam(self.geom, **kwargs)

        n_basis_elts = eig_vects.shape[0]/len(valleys)
        mixture = self.valley_mixture(state) # checks eigenvector quality
        idx = states.index(state)
        basis_coeffs = eig_vects[:, idx].reshape((len(valleys), n_basis_elts))

        # NOTE: Angle can vary randomly so modulus is used instead!
        return np.abs(basis_coeffs[valleys.index(valley),:]/mixture[valley])

    def valley_mixture(self, state):
        """ Defined as the mixture of a complete valley basis across valleys """
        kwargs = {'potentialMin': -1e3, 'multDict': {'fieldTerm':self.e_field},\
                  'numEigVals': len(valleys)}
        energies, eig_vects, panic = emtpy.solveOneParticleHam(self.geom, **kwargs)

        n_basis_elts = eig_vects.shape[0]/len(valleys)
        idx = states.index(state)
        basis_coeffs = eig_vects[:, idx].reshape((len(valleys), n_basis_elts))

        val_x = basis_coeffs[0,:]
        real = basis_coeffs.real/val_x.real
        imag = basis_coeffs.imag/val_x.imag

        # check consistency between normalised real and imaginary parts
        if not(np.allclose(real, imag, rtol = 1e-2)):
            raise ValueError('Mismatch between real and imaginary matrices!')

        # check that the normalised basis elements are the same within each valley
        for v in range(len(valleys)):
            real = basis_coeffs[v,:].real/val_x.real
            imag = basis_coeffs[v,:].imag/val_x.imag
            if not(np.allclose(real[:-1], real[1:], rtol = 1e-2)) or \
                    not(np.allclose(imag[:-1], imag[1:], rtol = 1e-2)):
                raise ValueError('Valley basis function composition is not' + \
                                 'homogeneous across valleys.')

        return dict(zip(valleys, (basis_coeffs.real/val_x.real)[:,0]))
