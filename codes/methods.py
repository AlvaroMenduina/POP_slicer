### --------------------------------------- ###
#-#                 methods                 #-#
### --------------------------------------- ###

"""
Date: Jan 2018
Author: Alvaro Menduina Fernandez - University of Oxford
Email: alvaro.menduinafernandez@physics.ox.ac.uk
Description:
    Package which contains methods for the calibration of
    'Non-Common Path Aberrations' (NCPA) in the context of
    HARMONI, the Integral Field Spectrograph of E-ELT

Methods included so far:
    (1) Phase Diversity

    (2) Differential Optical Transfer Function

"""

import numpy as np
from numpy import loadtxt
import zern_core as zern
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
from numpy.random import RandomState
from scipy.optimize import minimize, curve_fit, least_squares
from time import time as timer
from numba import jit
from astropy.io import fits
import warnings

### ------------------------------------------------------------------------- ----------------- ###
#-#                                       Helper Functions                                      #-#
### ------------------------------------------------------------------------- ----------------- ###

def load_fits(file_name):
    with fits.open(file_name) as hdul:
        print(hdul.info())
        image = hdul[1].data
        if np.isnan(image).any():
            warnings.warn("NaN value detected in at least one pixel")
    return image

def read_zernike_list(txt_file):
    list_zernikes = loadtxt(txt_file, dtype=str)
    return list_zernikes

def HARMONI_scale(spaxel_scale=4, wave=1.5e-6):
    """
    Tells you how you need to define your Aperture Radius with respect to
    the Size of the Pupil array so that the PSF at a certain wavelength
    is sampled with Spaxels of a certain scale
    :param spaxel_scale: the intended scale [mas] of your spaxels
    :param wave: wavelegth in [m]
    :return: eps: ratio of the D_ELT / L with L the physical size of your
    pupil array
    """
    ARCSECS_IN_A_RAD = 206265   # Number of arcseconds in a radian
    ELT_DIAMETER = 39.           # ELT diameter [m]
    spaxel_rad = spaxel_scale / 1000. / ARCSECS_IN_A_RAD
    L = wave / spaxel_rad       # Physical Lenght in [m]
    # print(L)
    eps = ELT_DIAMETER / L
    # print(eps)
    return eps

def elt_mask(radius, x, y, angl=np.pi/2.):
    """
    Create a Mask with the shape of the ELT pupil proportional to the given Radius
    It includes the central obscuration and the spiders. The mask is just an approximation
    as it assumes a circular aperture, while the real aperture is composed of hexagonal segments
    :param radius: Outer radius of the mask
    :param x, y: XY grid for the mask
    :param angl: Rotation angle for the mask
    :return:
    """
    r_elt = 36.903 / 2.          # Outer Diameter [m]
    r_obs = 11.208 / 2.          # Central obscuration diameter [m]
    spid_width = 0.53            # Width of the spiders [m]
    r_outer = radius
    r_inner = r_obs * (radius / r_elt)
    d_spid = spid_width * (radius / r_elt)

    mask_obsc = (x**2 + y**2 <= r_outer**2) & (x**2 + y**2 >= r_inner**2)
    mask_spider0 = np.abs(y - x * np.tan(angl - np.pi/3.)) >= d_spid / 2. / np.abs(np.cos(angl - np.pi/3.))
    mask_spider1 = np.abs(y - x * np.tan(angl)) >= d_spid / 2. / np.abs(np.cos(angl))
    mask_spider2 = np.abs(y - x * np.tan(angl + np.pi/3.)) >= d_spid / 2. / np.abs(np.cos(angl + np.pi/3.))

    return mask_obsc * mask_spider0 * mask_spider1 * mask_spider2

def complex_function(aperture, phase, wave):
    """
    Construct a complex pupil function based on a given phase map in [nm]
    P(r) = Aper(r) * exp(1j * 2 * Pi * Phase(r) / Wavelength)
    :param aperture: Pupil aperture map
    :param phase: Phase map in [nm]
    :param wave: Wavelength in [nm]
    """
    phase_rad = 2. * np.pi * phase / wave
    return aperture * np.exp(1j * phase_rad)

def compute_PV_2maps(phase_ref, phase_guess):
    """
    Compute the residual Peak-to-Valley difference between two phase maps
    :param phase_ref: Reference phase map
    :param phase_guess: Estimated phase map
    :return:
    """
    residual = phase_ref - phase_guess
    PV = np.max(residual[np.nonzero(residual)]) - np.min(residual[np.nonzero(residual)])
    # np.nonzero solves the problem of having a MASKED phase map with MAX=a, MIN=b with a,b > 0
    # otherwise, the PV would be MAX - 0, instead of MAX - MIN
    return PV

def compute_rms(phase_map):
    """
    Compute the RMS of a certain phase map (assumed to be already masked)
    """
    values = phase_map
    N = values.shape[0]
    mean_value = np.mean(values)
    rms = np.sqrt(1./N * np.sum((values - mean_value)**2))
    return rms

def compute_SNR(images_clean, images_photon):
    """
    Compute the Signal/Noise ratio for a given set of CLEAN and NOISE images
    """
    mean_SNR = []
    std_SNR = []
    for (clean, noisy) in zip(images_clean, images_photon):
        mask = noisy > 5*np.mean(noisy)
        # plt.figure()
        # plt.imshow(mask)
        #
        # plt.figure()
        # plt.imshow(noisy)
        noise = np.abs(noisy - clean)
        SNR_map = noisy / noise
        mean_SNR.append(np.mean(SNR_map[mask]))
        std_SNR.append(np.std(SNR_map[mask]))
    return mean_SNR, std_SNR



### ------------------------------------------------------------------------- ----------------- ###
#-#                                         Main Methods                                        #-#
### ------------------------------------------------------------------------- ----------------- ###


class Images(object):
    def __init__(self, xx, yy, pupil_mask, N_images=3):
        """
        Object which takes care of generating the Images for a Phase Diversity calculation

        Takes into account detector effects FIXME !
        :param xx:
        :param yy:
        :param pupil_mask:
        :param N_images:
        """
        self.pupil_mask = pupil_mask
        self.N_pix = xx.shape[0]
        self.N_imag = N_images

        # Take of the coordinates and masked arrays
        rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
        self.rho_m, self.theta_m = rho[self.pupil_mask], theta[self.pupil_mask]

    def set_phase_map(self, z_coef, rho_zern=1., PV_rescale=True, PV_goal=50):
        """
        Generates a phase map based on Zernike polynomials
        This usually represents an NCPA map
        Uses the fast methods implemented in zern_core.py
        :param z_coef: coefficient of the Zernike series expansion
        :param PV_goal: desired Peak-To-Valley [nm] of the phase map
        """
        # Construct the base phase map
        self.zern_model = zern.ZernikeNaive(mask=self.pupil_mask)
        phase_map = self.zern_model(coef=z_coef, rho=self.rho_m / rho_zern, theta=self.theta_m,
                      normalize_noll=False, mode='Jacobi', print_option='Silent')
        if PV_rescale:
            # Compute the current PV and rescale the coefficients
            current_pv = np.max(phase_map[np.nonzero(phase_map)]) - np.min(phase_map[np.nonzero(phase_map)])
            # Remember that at this point the phase map is a 1D array (a masked 2D)
            phase_map = zern.rescale_phase_map(phase_map, peak=PV_goal/2)
            self.ncpa_coef = self.zern_model.coef * (PV_goal / current_pv)  # Save the coefficients
            self.phase_map = zern.invert_mask(phase_map, self.pupil_mask)
        else:
            self.phase_map = zern.invert_mask(phase_map, self.pupil_mask)
            self.ncpa_coef = self.zern_model.coef


    def set_phase_diversity(self, n=2, m=0, rho_zern=1., ratio=10):
        """
        Creates the Phase Diversity map which will be used to 'defocus'
        the images. Although the common thing is to use a pure defocus term (n=2, m=0)
        any Zernike polynomial is possible

        The Phase Diversity map is rescaled according to a desired PV which is
        'ratio' times the PV of the NCPA
        """
        diversity = self.zern_model.Z_nm(n=n, m=m, rho=self.rho_m / rho_zern, theta=self.theta_m,
                                         normalize_noll=False, mode='Jacobi')
        pv_diversity = np.max(diversity) - np.min(diversity)
        pv_phase = np.max(self.phase_map) - np.min(self.phase_map)
        phase_diversity = (ratio * pv_phase) * (diversity / pv_diversity)
        # Remember that at this point the phase diversity is a 1D array (a masked 2D)
        self.phase_diversity = zern.invert_mask(phase_diversity, self.pupil_mask)

    def create_images(self, wave, t_exp=1.0):
        """
        :param wave: Wavelength always in [nm]!!
        :param t_exp: fake exposure time used to increase the signal so that we can
        check the influence of different photon noise SNR levels
        """
        norm_pix = 1./(self.N_pix)   # Normalization of the Fourier transform

        # Nominal Image
        pupil_nominal = complex_function(self.pupil_mask, self.phase_map, wave)
        propagated_nominal = norm_pix * fftshift(fft2(pupil_nominal))
        image_nominal = t_exp * (np.abs(propagated_nominal))**2

        # + Phase Diversity
        pupil_plus_defocus = complex_function(self.pupil_mask, (self.phase_map + self.phase_diversity), wave)
        propagated_plus_defocus = norm_pix * fftshift(fft2(pupil_plus_defocus))
        image_plus_defocus = t_exp * (np.abs(propagated_plus_defocus))**2

        if self.N_imag == 2:

            self.image_nominal = np.floor(image_nominal)
            self.image_plus_defocus = np.floor(image_plus_defocus)

        if self.N_imag == 3:
            pupil_minus_defocus = complex_function(self.pupil_mask, (self.phase_map - self.phase_diversity), wave)
            propagated_minus_defocus = norm_pix * fftshift(fft2(pupil_minus_defocus))
            image_minus_defocus = t_exp * (np.abs(propagated_minus_defocus))**2

            self.image_nominal = np.floor(image_nominal)
            self.image_plus_defocus = np.floor(image_plus_defocus)
            self.image_minus_defocus = np.floor(image_minus_defocus)

    def photon_noise(self):
        im_size = (self.N_pix, self.N_pix)
        self.image_nominal = np.floor(RandomState(1234).poisson(lam=self.image_nominal, size=im_size))
        self.image_plus_defocus = np.floor(RandomState(4321).poisson(lam=self.image_plus_defocus, size=im_size))
        if self.N_imag == 3:
            self.image_minus_defocus = np.floor(RandomState(5678).poisson(lam=self.image_minus_defocus, size=im_size))

    def flat_field_effect(self, variance):
        im_size = (self.N_pix, self.N_pix)
        a0 = 1.0 - variance
        b0 = 1.0 + variance
        self.flat_field_map = RandomState(1234).uniform(a0, b0, size=im_size)
        new_nom = self.image_nominal.copy() * self.flat_field_map
        new_plus = self.image_plus_defocus.copy() * self.flat_field_map
        if self.N_imag == 3:
            new_minus = self.image_minus_defocus.copy() * self.flat_field_map
            return [np.floor(new_nom), np.floor(new_plus), np.floor(new_minus)]


class PhaseDiversity(object):

    def __init__(self, xx, yy, pupil_mask, rho_aper, rho_zern, images,
                 phase_diversity, zern_model, wave, t_exp=1.0):

        self.t_exp = t_exp
        self.pupil_mask = pupil_mask
        self.N_images = len(images)
        self.N_pix = xx.shape[0]
        self.image_nominal = images[0]
        self.image_plus_defocus = images[1]
        if self.N_images == 3:
            self.image_minus_defocus = images[2]
        self.wave = wave
        self.phase_diversity = phase_diversity
        self.zern_model = zern_model
        self.rho_aper, self.rho_zern = rho_aper, rho_zern

        # Take of the coordinates and masked arrays
        rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
        self.rho_m, self.theta_m = rho[self.pupil_mask], theta[self.pupil_mask]

        # Reshape the Zern Model Matix H
        H = zern_model.model_matrix
        self.model_matrix = zern.invert_model_matrix(H, pupil_mask)

    def set_true_phase(self, ncpa_coef):
        r = self.rho_m * (self.rho_zern / self.rho_aper)
        nominal_map = self.zern_model(coef=ncpa_coef, rho=r, theta=self.theta_m,
                      normalize_noll=False, mode='Jacobi', print_option='Silent')
        self.true_phase = zern.invert_mask(nominal_map, self.pupil_mask)

    def evaluate_phase(self, zern_coef):
        r = self.rho_m * (self.rho_zern / self.rho_aper)
        nominal_map = self.zern_model(coef=zern_coef, rho=r, theta=self.theta_m,
                      normalize_noll=False, mode='Jacobi', print_option='Silent')
        nominal_map = zern.invert_mask(nominal_map, self.pupil_mask)
        # plt.figure()
        # plt.imshow(nominal_map)
        # plt.colorbar()
        return nominal_map

    def cost(self, zern_coef):
        norm_pix = 1. / (self.N_pix)
        r = self.rho_m * (self.rho_zern / self.rho_aper)
        nominal_map = self.zern_model(coef=zern_coef, rho=r, theta=self.theta_m,
                      normalize_noll=False, mode='Jacobi', print_option='Silent')
        nominal_map = zern.invert_mask(nominal_map, self.pupil_mask)

        # Nominal Image
        pupil_nominal = complex_function(self.pupil_mask, nominal_map, self.wave)
        propagated_nominal = norm_pix * fftshift(fft2(pupil_nominal))
        image_nominal = self.t_exp * (np.abs(propagated_nominal))**2
        J_nominal = (self.image_nominal - image_nominal)**2

        # + Phase Diversity
        pupil_plus_defocus = complex_function(self.pupil_mask, (nominal_map + self.phase_diversity), self.wave)
        propagated_plus_defocus = norm_pix * fftshift(fft2(pupil_plus_defocus))
        image_plus_defocus = self.t_exp * (np.abs(propagated_plus_defocus))**2
        J_plus = (self.image_plus_defocus - image_plus_defocus) ** 2

        if self.N_images == 3:
            # - Phase Diversity
            pupil_minus_defocus = complex_function(self.pupil_mask, (nominal_map - self.phase_diversity), self.wave)
            propagated_minus_defocus = norm_pix * fftshift(fft2(pupil_minus_defocus))
            image_minus_defocus = self.t_exp * (np.abs(propagated_minus_defocus))**2
            J_minus = (self.image_minus_defocus - image_minus_defocus) ** 2
            return np.sum(J_nominal + J_plus + J_minus) / (self.t_exp**2)
        else:
            return np.sum(J_nominal + J_plus) / (self.t_exp**2)

    def helper_grad(self, Z_k, E, Ec, FE, FEc):

        wave_factor = 2*np.pi/self.wave*1j * Z_k
        norm_pix = 1. / (self.N_pix)

        # FE = norm_pix * fftshift(fft2(E))

        f1a = norm_pix * fftshift(fft2(wave_factor * E))
        # f1b = np.conj(FE)

        f2a = norm_pix * fftshift(fft2(-wave_factor * Ec))
        f2a = f2a[::-1, ::-1]
        f2a = np.roll(f2a, shift=(1,1), axis=(1,0))
        fourier_factor = -1*(f1a * FEc + f2a * FE)
        return fourier_factor

    def grad_analytic(self, zern_coef):
        """
        Analytic gradient for speed purposes
        """
        grad_start = timer()

        N_zern = zern_coef.shape[0]

        wave_factor = 2*np.pi/self.wave*1j
        norm_pix = 1. / (self.N_pix)
        r = self.rho_m * (self.rho_zern / self.rho_aper)
        nominal_map = self.zern_model(coef=zern_coef, rho=r, theta=self.theta_m,
                      normalize_noll=False, mode='Jacobi', print_option='Silent')
        nominal_map = zern.invert_mask(nominal_map, self.pupil_mask)

        # Nominal Image
        pupil_nominal = complex_function(self.pupil_mask, nominal_map, self.wave)
        propagated_nominal = norm_pix * fftshift(fft2(pupil_nominal))
        image_nominal = self.t_exp * (np.abs(propagated_nominal))**2

        # + Phase Diversity
        pupil_plus_defocus = complex_function(self.pupil_mask, (nominal_map + self.phase_diversity), self.wave)
        propagated_plus_defocus = norm_pix * fftshift(fft2(pupil_plus_defocus))
        image_plus_defocus = self.t_exp * (np.abs(propagated_plus_defocus))**2

        # - Phase Diversity
        pupil_minus_defocus = complex_function(self.pupil_mask, (nominal_map - self.phase_diversity), self.wave)
        propagated_minus_defocus = norm_pix * fftshift(fft2(pupil_minus_defocus))
        image_minus_defocus = self.t_exp * (np.abs(propagated_minus_defocus)) ** 2

        base_factor_nom = 2*(self.image_nominal - image_nominal)
        base_factor_plus = 2*(self.image_plus_defocus - image_plus_defocus)
        base_factor_minus = 2*(self.image_minus_defocus - image_minus_defocus)

        # Helper stuff
        Ec_nom, Ec_plus, Ec_minus = np.conj(pupil_nominal), np.conj(pupil_plus_defocus), np.conj(pupil_minus_defocus)

        FE_nom = norm_pix * fftshift(fft2(pupil_nominal))
        FE_plus = norm_pix * fftshift(fft2(pupil_plus_defocus))
        FE_minus = norm_pix * fftshift(fft2(pupil_minus_defocus))

        FEc_nom, FEc_plus, FEc_minus = np.conj(FE_nom), np.conj(FE_plus), np.conj(FE_minus)

        # print('Common: %f sec' %(timer() - grad_start))

        g = np.zeros(N_zern)
        for k in range(N_zern):
            Z_k = self.model_matrix[:,:,k]

            fourier_factor_nom = self.helper_grad(Z_k, pupil_nominal, Ec_nom, FE_nom, FEc_nom)
            fourier_factor_plus = self.helper_grad(Z_k, pupil_plus_defocus, Ec_plus, FE_plus, FEc_plus)
            fourier_factor_minus = self.helper_grad(Z_k, pupil_minus_defocus, Ec_minus, FE_minus, FEc_minus)

            grad_nom = np.sum(base_factor_nom * self.t_exp * fourier_factor_nom)
            grad_plus = np.sum(base_factor_plus * self.t_exp * fourier_factor_plus)
            grad_minus = np.sum(base_factor_minus * self.t_exp * fourier_factor_minus)
            g[k] = np.real(grad_nom + grad_plus + grad_minus)/ (self.t_exp**2)

        return g

    def callback_function(self, coef):
        """
        Callback to print intermediate results at each iteration
        """
        cost_now = self.cost(coef)
        self.cost_array.append(cost_now)
        coef_copy = coef
        coef_copy[0] = 0    #remove piston
        print ('\nAt iteration %d :' % self.counter)
        r = self.rho_m * (self.rho_zern / self.rho_aper)
        nominal_map = self.zern_model(coef=coef_copy, rho=r, theta=self.theta_m,
                      normalize_noll=False, mode='Jacobi', print_option='Silent')
        nominal_map = zern.invert_mask(nominal_map, self.pupil_mask)
        try:
            p0 = self.true_phase
            PV = compute_PV_2maps(phase_ref=self.true_phase, phase_guess=nominal_map)
            self.PV_array.append(PV)
            RMS = compute_rms((self.true_phase - nominal_map)[self.pupil_mask])
            self.RMS_array.append(RMS)
            print('Merit Function: %.3E' %cost_now)
            print('PV : %.3f' %PV)
            print('RMS: %.3f' %RMS)
        except AttributeError:
            pass
        # cost_at_iter = self.cost(coef)
        # print 'Merit function = %e' %cost_at_iter
        self.guesses[:,:,self.counter] = nominal_map
        self.counter += 1

    def optimize(self, coef0, N_iter=100):
        """
        Run the optimization based on the Phase Diversity merit function
        """
        # Reinitialize counters and convergence arrays
        self.counter = 0
        self.cost_array = []
        self.PV_array = []
        self.RMS_array = []
        self.guesses = np.zeros((self.N_pix, self.N_pix, N_iter))

        optimization = minimize(self.cost, coef0, method='BFGS', jac=self.grad_analytic,
                                callback=self.callback_function,
                                options={'disp':True, 'maxiter':N_iter})

        self.final_coef = optimization['x']