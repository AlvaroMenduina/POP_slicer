### ----------------------------------------------- ###
#-#                 PHASE DIVERSITY                 #-#
### ----------------------------------------------- ###

"""
Date: Jun 2018
Author: Alvaro Menduina Fernandez - University of Oxford
Email: alvaro.menduinafernandez@physics.ox.ac.uk
Description:
    Phase Diversity experiment using PSFs produced via POP
    with Zemax through the PCS Image Slicer
"""

import os
import sys
import numpy as np
from numpy.fft import fft2, fftshift
import matplotlib.pyplot as plt
import pop_methods as pop
import zern_core as zern


""" PARAMETERS """

# System
wave = 0.8          # um
aper_D = 0.13333    # mm
foc_length = 20     # mm

# ZBF
N_pix = 1024
x_size = 0.60
extends = [-x_size/2, x_size/2, -x_size/2, x_size/2]
slice_start = 23
slice_end = 33
i_central = (slice_end - slice_start)//2
N_slices = slice_end - slice_start + 1

# PSF sampling
n_crop = 1
lu = wave * 1e-3 * N_pix/n_crop * foc_length / (2*x_size)
eps = aper_D / lu

# Phase Diversity
nom_defocus = 0.25  # waves

if __name__ == "__main__":

    if sys.argv[-1] == "load":
        # Over-ride the Zemax process (as it's very slow)
        nom_psf = np.load('nom_psf.npy')
        plus_psf = np.load('plus_psf.npy')
        minus_psf = np.load('minus_psf.npy')

        images = [nom_psf, plus_psf, minus_psf]

    else:

        path_zemax = os.path.join('zemax_files', 'PHASE_DIVERSITY')
        nom_psf, plus_psf, minus_psf = pop.read_phase_diversity_pop(path_zemax, slice_start, slice_end)

        images = [nom_psf, plus_psf, minus_psf]
        np.save('nom_psf', nom_psf)
        np.save('plus_psf', plus_psf)
        np.save('minus_psf', minus_psf)

    # Create the IDEAL PSF (No Slicer)
    x = np.linspace(-1., 1., n_crop*N_pix, endpoint=True)
    xx, yy = np.meshgrid(x, x)
    rho = np.sqrt(xx ** 2 + yy ** 2)
    theta = np.arctan2(yy, xx)
    aper_mask = rho <= eps

    rho, theta = rho[aper_mask], theta[aper_mask]
    rho /= eps

    plt.figure()
    plt.imshow(aper_mask)
    plt.title('Pupil Aperture')

    pupil_complex = aper_mask * np.exp(1j * np.zeros((n_crop*N_pix, n_crop*N_pix)))
    complex_field = fftshift(fft2(pupil_complex))
    image = (np.abs(complex_field))**2
    peak_image = np.max(image)
    image /= peak_image

    # Compare the IDEAL and the TRUE PSFs
    plt.figure()
    plt.subplot(121)
    plt.imshow(nom_psf, extent=extends, cmap='jet')
    plt.xlabel('X [mm]')
    plt.ylabel('Y [mm]')
    plt.title('Nominal PSF (Slicer)')

    plt.subplot(122)
    plt.imshow(pop.crop_array(image, N_pix), extent=extends, cmap='jet')
    plt.xlabel('X [mm]')
    plt.title('Ideal PSF')

    # Show an example of NCPA map
    zern_model = zern.ZernikeNaive(mask=aper_mask)
    zern_coef = np.zeros(10)
    zern_coef[4] = 0.25
    zz = zern_model(coef=zern_coef, rho=rho, theta=theta, mode='Standard', print_option=None)
    phase = zern.invert_mask(zz, aper_mask)

    H = zern_model.model_matrix

    plt.figure()
    plt.imshow(zern.invert_mask(H[:, 4], aper_mask), extent=[-1, 1, -1, 1])
    plt.xlim([-1.1*eps, 1.1*eps])
    plt.ylim([-1.1*eps, 1.1*eps])
    plt.colorbar()

    # Investigate why there's a difference in Cost
    # when you use a slightly different defocus than
    # the one Zemax says we used

    j = []
    focus = np.linspace(0.1, 0.2, 20)
    for foc in focus:
        foc_zern = np.zeros(6)
        foc_zern[4] = foc
        deff = zern_model(coef=foc_zern, rho=rho, theta=theta, mode='Standard', print_option=None)
        deff = zern.invert_mask(deff, aper_mask)
        pd = pop.POP_PhaseDiversity(images=images, eps=eps, aper_mask=aper_mask,
                                    zern_model=zern_model, phase_diversity=deff)
        j.append(pd.cost(np.zeros(6)))

    plt.figure()
    plt.plot(focus, j)
    plt.xlabel('Defocus [waves]')
    plt.title('PD Cost')

    # Run the Phase Diversity estimation
    foc_zern = np.zeros(10)
    foc_zern[4] = nom_defocus
    diversity = zern_model(coef=foc_zern, rho=rho, theta=theta, mode='Standard', print_option=None)
    diversity = zern.invert_mask(diversity, aper_mask)

    pd = pop.POP_PhaseDiversity(images=images, eps=eps, aper_mask=aper_mask,
                                zern_model=zern_model, phase_diversity=diversity)
    pd.grad_analytic(np.zeros(10))
    pd.optimize(np.zeros(10), N_iter=25)

    # Plot the estimated NCPA

    x0 = np.linspace(-1, 1, 1024, endpoint=True)
    xx0, yy0 = np.meshgrid(x0, x0)
    mask = xx0 ** 2 + yy0 ** 2 <= 1
    r0 = np.sqrt(xx0 ** 2 + yy0 ** 2)
    theta0 = np.arctan2(yy0, xx0)
    rho_0 = r0[mask]
    t0 = theta0[mask]
    zern_model1 = zern.ZernikeNaive(mask=mask)
    z = zern_model1(coef=pd.final_coef, rho=rho_0, theta=t_0, mode='Standard', print_option=None)
    final = zern.invert_mask(z, mask)

    mean = np.mean(z)
    n = z.shape[0]
    RMS = np.sqrt(1/n * np.sum((z - mean)**2))

    plt.figure()
    plt.imshow(final * 800)
    plt.colorbar()
    plt.title('NCPA map [nm] guessed by PD (including Slicer Effects)')

    plt.show()