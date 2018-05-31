import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import zern_core as zern
import methods as mt
import slicer_methods as smt
import machine_learning_methods as fz
from time import time as timer
from sklearn.neural_network import MLPRegressor
import os

# PSF Parameters
N_pix = 256
PSF_pix = 15
wave = 1.5                  # Wavelength [microns]
max_N_zern = 6
rho_max = 1.
t_exp = 1.

# HARMONI scale
spaxel_scale = 4      # HARMONI spaxel scale [mas]
D_elt = 39.
eps = mt.HARMONI_scale(spaxel_scale, wave=wave*1e-6)

# Machine Learning Parameters
PV_training = 0.75           # [microns]
PV_test = 0.75               # [microns]
N_images = int(5000)
N_test = 1000
training_epochs = 9
N_layer = (350,)

# IMAGE SLICER Parameters
rand_state = RandomState(1234)
Nx, Ny, N_spax = 4, 30, 25
defocus = 20.0                        # [mm]

# Coordinates
x = np.linspace(-rho_max, rho_max, N_pix)
xx, yy = np.meshgrid(x, x)
rho = np.sqrt(xx ** 2 + yy ** 2)
theta = np.arctan2(xx, yy)
extends = [-rho_max, rho_max, -rho_max, rho_max]



if __name__ == "__main__":

    print('\n====================================================')
    print('                  PSF parameters:                   ')
    print('____________________________________________________')
    print('N_pixels: %d' %N_pix)
    print('D_ELT/L = %.2f' %eps)
    print('Wavelenght: %.1f [microns]' %wave)
    print('T exposure: %.2f' %t_exp)
    print('Max Zernikes: %d' %max_N_zern)
    print('====================================================\n')

    pupil_mask = mt.elt_mask(eps, xx, yy)

    # WATCH OUT because this rho_masked is renormalized to 1
    # so that the zern_model.model_matrix follows the convention of giving
    # Z = 1 at the borders (i.e the typical Zernike map)
    rho_masked, theta_masked = rho[pupil_mask] / eps, theta[pupil_mask]

    # Initialize the Zernike Model
    zern_model = zern.ZernikeNaive(mask=pupil_mask)
    # Use a Random vector to run it once and thus
    # initiliaze the Model Matrix H
    _coef = np.random.normal(size=max_N_zern)
    zern_model(_coef, rho_masked, theta_masked,
               normalize_noll=False, mode='Jacobi', print_option='Silent')
    H_matrix = zern_model.model_matrix

    """ ++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
    """                     MACHINE LEARNING                   """
    """ ++++++++++++++++++++++++++++++++++++++++++++++++++++++ """


    smart_psf = fz.SmartPSF(zern_model=zern_model, rho_max=rho_max, eps=eps,
                            N_pix=N_pix, PSF_pix=PSF_pix, N_images=N_images,
                            wave=wave, t_exp=t_exp)
    smart_psf.create_PSF_maps(PV=PV_training/2, low=False)

    p0 = smart_psf.PSF_maps[0]
    for i in range(max_N_zern):
        # z = smart_psf.zernike_maps[i]
        # z2d = zern.invert_mask(z, pupil_mask)
        # plt.figure()
        # plt.imshow(z2d, extent=extends, cmap='jet')
        # plt.xlim([-eps, eps])
        # plt.ylim([-eps, eps])
        # plt.colorbar()

        p = smart_psf.PSF_maps[i]
        p = p.reshape((PSF_pix, PSF_pix))
        plt.figure()
        plt.imshow(p, cmap='viridis')
        plt.colorbar()
        plt.title(i)

    """ Training the ML model """
    smart_model = MLPRegressor(hidden_layer_sizes=N_layer, activation='logistic',
                               solver='lbfgs', learning_rate='adaptive', max_iter=500,
                               verbose=False, tol=1e-9, warm_start=True,
                               alpha=0.,
                               learning_rate_init=0.01)

    # Train the model
    print('         Training the MACHINE LEARNING model        ')
    print('____________________________________________________')
    print('Training Epoch 0')
    training_set = smart_psf.PSF_maps[:-3, :]  # Exclude the last one
    targets = smart_psf.zern_coef[:-3, 1:]  # Exclude the Piston and Tilt terms
    smart_model.fit(X=training_set, y=targets)
    current_loss = smart_model.loss_
    print('Current loss F=%e\n' % current_loss)

    for i in range(training_epochs):
        print('Training Epoch %d' % (i+1))
        smart_model.fit(X=training_set, y=targets)
        current_loss = smart_model.loss_
        print('Current loss F=%e\n' % current_loss)

    print('\nChecking an example')
    last_map = smart_psf.PSF_maps[-2:-1, :]
    guessed = smart_model.predict(X=last_map)
    print('ML model guesses:')
    print(guessed[0])
    print('True values are :')
    print(smart_psf.zern_coef[-2, 1:])
    print('====================================================\n')

    """ Testing the Model """
    smart_psf.create_test_dataset(N_images=N_test, PV_goal=PV_test)
    pred, true_val, error = smart_psf.predict_test(smart_model)



    """ ++++++++++++++++++++++++++++++++++++++++++++++++++++++ """
    """                       IMAGE SLICER                     """
    """ ++++++++++++++++++++++++++++++++++++++++++++++++++++++ """

    """ Create the NCPA map to feed trough the Slicer """
    zern_coef = rand_state.uniform(low=0., high=1., size=max_N_zern)
    zern_coef[0] = 0.  # remove piston
    _ncpa = np.dot(H_matrix, zern_coef)
    pv = np.max(_ncpa[np.nonzero(_ncpa)]) - np.min(_ncpa[np.nonzero(_ncpa)])
    zern_coef *= (PV_test / pv)

    """ Create the Pupil Masks for the Slicer and so on """
    slicer = smt.ImageSlicer(Nx=Nx, Ny=Ny, N_spax=N_spax)
    N_slices = slicer.N_slices
    slicer.compute_key_stuff(defocus=defocus)
    slicer.compute_cutoffs()
    slicer.create_masks(N_pix, eps)

    """ Define the Fourier Propagator and compute the true PSF"""
    trapezoid = smt.Integrator()
    propagator = smt.FourierPropagator(zern_model, rho_max, eps, N_pix, wave,
                                       slicer, trapezoid)
    propagator.create_wavefront(coef=zern_coef, PV_goal=PV_test)
    plt.figure()
    plt.imshow(propagator.wavefront, extent=extends, cmap='jet')
    plt.xlim([-eps, eps])
    plt.ylim([-eps, eps])
    plt.colorbar()
    plt.title('NCPA map with %.3f [um] PV' % PV_test)

    # propagator.propagate_to_image(low_memory=True)

    i_slice, j_spaxel = -1, -1

    N_min = N_pix//2 - (PSF_pix-1)//2
    N_max = N_pix//2 + (PSF_pix-1)//2 + 1
    # PSF_slicer = propagator.final_intensities[0,1, N_min:N_max, N_min:N_max]
    PSF_slicer = propagator.propagate_single_spaxel(i_slice, j_spaxel)[N_min:N_max, N_min:N_max]
    PSF_nominal = smart_psf.compute_PSF(propagator.wavefront).reshape((PSF_pix, PSF_pix))

    plt.figure()
    plt.imshow(PSF_nominal)
    plt.colorbar()
    plt.title('PSF that ML would expect')

    plt.figure()
    plt.imshow(PSF_slicer)
    plt.colorbar()
    plt.title('PSF (post-SLICER) that ML receives')

    plt.figure()
    plt.imshow(PSF_nominal - PSF_slicer)
    plt.colorbar()
    plt.title('Difference: Expected vs Received')


    # k = N_slices * N_spax
    k = 1
    PSF_slicer = PSF_slicer.reshape(k, PSF_pix*PSF_pix)
    psf = PSF_slicer.flatten()

    true_coef = zern_coef[1:]

    print('\nChecking IMAGE SLICER PSF')
    guessed = smart_model.predict(X=PSF_slicer)
    print('\nML model guesses:')
    print(guessed[0, :])
    print('\nTrue values are :')
    print(true_coef)

    """ Compare the guess and the true NCPA """
    _true_ncpa = np.dot(H_matrix, zern_coef)
    guessed_coef = np.zeros_like(zern_coef)
    guessed_coef[1:] = guessed
    _guessed_ncpa = np.dot(H_matrix, guessed_coef)
    _res_ncpa = _true_ncpa - _guessed_ncpa
    PV = np.max(_res_ncpa) - np.min(_res_ncpa)
    mu = np.mean(_res_ncpa)
    nn = _res_ncpa.shape[0]
    RMS = np.sqrt(1./nn * np.sum((_res_ncpa - mu)**2))

    true_ncpa = 1e3*zern.invert_mask(_true_ncpa, pupil_mask)
    guessed_ncpa = 1e3*zern.invert_mask(_guessed_ncpa, pupil_mask)
    res_ncpa = true_ncpa - guessed_ncpa

    print('\nDefocus = %.1f [mm]' %defocus)
    print('PV residual NPCA = %.1f [nm]' %(1e3 * PV))
    print('RMS residual NPCA = %.1f [nm]\n' % (1e3 * RMS))

    plt.figure()
    plt.imshow(true_ncpa, cmap='jet')
    plt.colorbar()
    plt.title('True NCPA map [nm]')

    plt.figure()
    plt.imshow(guessed_ncpa, cmap='jet')
    plt.colorbar()
    plt.title('Guessed NCPA map [nm] (defocus = %.1f [mm])' %defocus)

    plt.figure()
    plt.imshow(res_ncpa, cmap='jet')
    plt.colorbar()
    plt.title('Residual NCPA map [nm] (defocus = %.1f [mm])' %defocus)

    for i in range(max_N_zern - 1):
        plt.figure()
        guess_10 = np.sort(guessed[:,i])
        # guess_50 = np.sort(guessed_50[:,i])
        # guess_50 = np.sort(guessed_25[:, i])
        plt.plot(guess_10, label='Defocus 10')
        plt.plot(guess_25, label='Defocus 25')
        plt.plot(guess_50, label='Defocus 50')
        plt.axhline(y=true_coef[i], linestyle='--', color='Black', label='True')
        plt.xlabel('Spaxel')
        plt.title(smart_psf.zern_list[i])
        plt.legend()

    plt.show()