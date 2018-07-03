# Physical Optics Propagation (POP) - Machine Learning
#
# This code is for the Machine Learning method of NCPA but including slicer effects
# We are trying to find out whether a MULTI-WAVELENGTH approach might work
# That is, instead of using a defocus to break the degeneracy, we could use
# the PSF at different wavelengths (from the data cube)
#
#

import os
import sys
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import pop_methods as pop
from sklearn.neural_network import MLPRegressor

# Slices
slice_start = 24
slice_end = 32
i_central = (slice_end - slice_start)//2
N_slices = slice_end - slice_start + 1

# POP arrays
x_size = 0.54
N_pix = 64
extends = [-x_size / 2, x_size / 2, -x_size / 2, x_size / 2]
xc = np.linspace(-x_size / 2, x_size / 2, 10)

# Machine Learning parameters
N_layer = (300,)
N_iter = 150
N_epochs = 4
random_train = True
rand_state = RandomState(1234)

# Zernikes
N_zern = 5
sampling = 5
N_train = sampling ** N_zern

z_min = 0.0
z_max = 1.0
delta = (z_max - z_min) / (N_zern - 1)
z_norm = 0.1

# Wavelength
wave_nom = 0.8
wave = 1.0

zern_list = ['Defocus', 'Astig_X', 'Astig_Y', 'Coma_X', 'Coma_Y']

def generate_sampling(sampling, N_zern, delta, start=0.0):
    coefs = np.empty((sampling**N_zern, N_zern))
    for i in range(N_zern):
        n = sampling ** (N_zern - (i + 1))
        a = start * np.ones(n)
        for j in range(sampling - 1):
            b = (start + (j + 1) * delta) * np.ones(n)
            a = np.concatenate((a, b))
        index = np.tile(a, sampling ** i)
        coefs[:, i] = index
    return coefs


if __name__ == "__main__":

   path_zemax = os.path.join('zemax_files', 'ML_DEFOCUSING')
   results_path = os.path.join('results', 'ML_MULTIWAVE')

   slicer_PSFs = np.empty((N_train, 2*N_pix*N_pix))
   pop_slicer_nom = pop.POP_Slicer()

   """ Generate the huge array of coefficients without doing # N_zern loops"""
   zern_coefs = generate_sampling(sampling, N_zern, delta, start=z_min)

   """ Load and process the ZBF files to generate the PSFs """
   for k in range(N_train):
      if k == 2899 or k == 2583:
          slicer_PSFs[k, :] = np.zeros(2*N_pix**2)

      else:
          if k < 10:
              name_nominal = 'SlicerTwisted_006f6a_POP_ML' + '% d_' %k
              name_extrawave = 'SlicerTwisted_006f6a_POP_ML_MULTIWAVE' + '% d_WAVE1_' %k
          else:
              name_nominal = 'SlicerTwisted_006f6a_POP_ML' + '%d_' %k
              name_extrawave = 'SlicerTwisted_006f6a_POP_ML_MULTIWAVE' + '%d_WAVE1_' % k


          pop_slicer_nom.get_zemax_files(path_zemax, name_nominal,
                                         start=slice_start, finish=slice_end, mode='irradiance')
          slicers_nom = np.sum(pop_slicer_nom.beam_data, axis=0).flatten()

          pop_slicer_nom.get_zemax_files(path_zemax, name_extrawave,
                                     start=slice_start, finish=slice_end, mode='irradiance')
          slicers_wave = np.sum(pop_slicer_nom.beam_data, axis=0).flatten()
          slicer_PSFs[k, :] = np.concatenate((slicers_nom, slicers_wave))

   """ Normalize every PSF to the peak of the Nominal (Un-aberrated)"""
   peak_nom = np.max(slicer_PSFs[0, :N_pix**2])
   slicer_PSFs /= peak_nom

   """ Show a couple of examples """
   i_ex = 550
   wave0 = slicer_PSFs[i_ex, :N_pix**2].reshape((N_pix, N_pix))
   wave1 = slicer_PSFs[i_ex, N_pix**2:].reshape((N_pix, N_pix))
   joint = np.concatenate((wave0, wave1), axis=1)

   plt.figure()
   plt.imshow(joint, origin='lower', cmap='jet')
   plt.colorbar(orientation='horizontal')
   plt.xlabel('Pixels')
   plt.ylabel('Pixels')
   plt.title('Slicer PSF at 0.8 um and 1.0 um')
   plt.savefig(os.path.join(results_path, 'Z%d' %i_ex))

   plt.figure()
   plt.imshow(np.log10(joint), origin='lower', cmap='jet')
   plt.colorbar(orientation='horizontal')
   plt.xlabel('Pixels')
   plt.ylabel('Pixels')
   plt.title('Log10 Slicer PSF at 0.8 um and 1.0 um')
   plt.savefig(os.path.join(results_path, 'Log10_Z%d' % i_ex))

   """ Randomize the training """
   n_test = 20
   random_choice = rand_state.choice(N_train, N_train-n_test, replace=False)
   rand_train_set = slicer_PSFs[random_choice,:]
   rand_zern_coef = zern_coefs[random_choice]

   test_choice = np.delete(np.arange(N_train), random_choice)

   if random_train==True:
       training_set = rand_train_set
       targets = rand_zern_coef
       test_set = slicer_PSFs[test_choice, :]
       test_coef = zern_coefs[test_choice, :]

   else:
       training_set = slicer_PSFs[:N_train - n_test, :]
       targets = zern_coefs[:N_train - n_test, :]
       test_set = slicer_PSFs[N_train - n_test:N_train, :]
       test_coef = zern_coefs[N_train - n_test:N_train, :]

   solver = 'lbfgs'
   N_layer = (300, 200, 100,)
   model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                        solver='adam', max_iter=N_iter, verbose=True,
                        batch_size='auto', shuffle=True, tol=1e-7,
                        warm_start=True, alpha=1e-2, random_state=1234)
   model.fit(X=training_set, y=targets)

   """ Test the model """

   guessed = model.predict(X=test_set)
   print("\nML model guesses:")
   print(guessed)
   print("\nTrue Values")
   print(test_coef)

   for k in range(N_zern):

       guess = guessed[:, k]
       coef = test_coef[:, k]
       plt.figure()
       plt.scatter(coef, guess)
       plt.plot(coef, coef, color='black')
       title = zern_list[k] + '  (N_train=%d, N_test=%d)' %(N_train - n_test, n_test)
       plt.title(title)
       plt.xlabel('True Value [waves]')
       plt.ylabel('Predicted Value [waves]')
       plt.savefig(os.path.join(results_path, zern_list[k]))


   plt.show()