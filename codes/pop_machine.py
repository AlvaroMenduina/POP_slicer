# Physical Optics Propagation (POP) - Machine Learning
#
# This code is for the Machine Learning method of NCPA but including slicer effects
#
#
#

import os
import sys
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import pop_methods as pop
from sklearn.neural_network import MLPRegressor
import time

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
extra_focus = 0.05

z_min = 0.0
z_max = 1.0
delta = (z_max - z_min) / (N_zern - 1)
z_norm = 0.1

N = 3**5

zern_list = ['Defocus', 'Astig_X', 'Astig_Y', 'Coma_X', 'Coma_Y']


if __name__ == "__main__":

  path_zemax = os.path.join('zemax_files', 'ML_DEFOCUSING')
  results_path = os.path.join('results', 'ML_DEFOCUSING')

  pop_slicer_nom = pop.POP_Slicer()
  pop_slicer_foc = pop.POP_Slicer()

  slicer_PSFs = np.empty((N_train, 2*N_pix*N_pix))
  zern_coefs = np.empty((N_train, N_zern))

  """ Generate the huge array of coefficients without doing # N_zern loops"""
  for i in range(N_zern):
     n = sampling ** (N_zern - (i+1))
     a = np.zeros(n)
     for j in range(sampling-1):
        b = (j+1) * delta * np.ones(n)
        a = np.concatenate((a, b))
     index = np.tile(a, sampling**i)
     zern_coefs[:, i] = index

  """ Load and process the ZBF files to generate the PSFs """
  for k in range(N_train):
     if k == 2899:
         slicer_PSFs[k, :] = np.zeros(2*N_pix**2)

     else:
         if k < 10:
             name_nominal = 'SlicerTwisted_006f6a_POP_ML' + '% d_' %k
             name_defocus = 'SlicerTwisted_006f6a_POP_ML_FOC_' + ' %d_' % k
         else:
             name_nominal = 'SlicerTwisted_006f6a_POP_ML' + '%d_' %k
             name_defocus = 'SlicerTwisted_006f6a_POP_ML_FOC_' + '%d_' % k


         pop_slicer_nom.get_zemax_files(path_zemax, name_nominal,
                                        start=slice_start, finish=slice_end, mode='irradiance')
         slicers_nom = np.sum(pop_slicer_nom.beam_data, axis=0).flatten()

         pop_slicer_foc.get_zemax_files(path_zemax, name_defocus,
                                    start=slice_start, finish=slice_end, mode='irradiance')
         slicers_foc = np.sum(pop_slicer_foc.beam_data, axis=0).flatten()
         slicer_PSFs[k, :] = np.concatenate((slicers_nom, slicers_foc))

  """ Normalize every PSF to the peak of the Nominal (Un-aberrated)"""
  peak_nom = np.max(slicer_PSFs[0])
  slicer_PSFs /= peak_nom

  peaks = np.max(slicer_PSFs, axis=1)
  mean_peaks = np.mean(peaks)

  plt.figure()
  plt.scatter(range(N_train), peaks, s=2)

  """ Show a couple of examples """
  i_ex = 1942
  nom_ex = slicer_PSFs[i_ex, :N_pix**2].reshape((N_pix, N_pix))
  foc_ex = slicer_PSFs[i_ex, N_pix**2:].reshape((N_pix, N_pix))
  joint = np.concatenate((nom_ex, foc_ex), axis=1)

  plt.figure()
  plt.imshow(joint, origin='lower', cmap='jet')
  plt.colorbar(orientation='horizontal')
  plt.xlabel('Pixels')
  plt.ylabel('Pixels')
  plt.title('Slicer PSF (Zern: ' +str(z_norm*zern_coefs[i_ex]) +')')
  plt.savefig(os.path.join(results_path, 'Z%d' %i_ex))

  plt.figure()
  plt.imshow(np.log10(joint), origin='lower', cmap='jet')
  plt.colorbar(orientation='horizontal')
  plt.xlabel('Pixels')
  plt.ylabel('Pixels')
  plt.title('Log10 Slicer PSF (Zern: ' +str(z_norm*zern_coefs[i_ex]) +')')
  plt.savefig(os.path.join(results_path, 'Log10_Z%d' % i_ex))

  plt.show()

  """ Randomize the training """
  n_test = 50
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
      training_set = slicer_PSFs[:N - n_test, :]
      targets = zern_coefs[:N - n_test, :]
      test_set = slicer_PSFs[N - n_test:N, :]
      test_coef = zern_coefs[N - n_test:N, :]

  """ Train the Machine Learning """
  solver = 'lbfgs'
  N_layer = (300, 200, 100,)
  model = MLPRegressor(hidden_layer_sizes=N_layer, activation='relu',
                       solver='adam', max_iter=N_iter, verbose=True,
                       batch_size='auto', shuffle=True, tol=1e-7,
                       warm_start=True, alpha=1e-2, random_state=1234)
  model.fit(X=training_set, y=targets)
  #
  # for i in range(N_epochs):
  #     print("\nTraining Epoch %d" %(i+1))
  #     model.fit(X=training_set, y=targets)
  #     current_loss = model.loss_
  #     print("Current loss F=%e" %current_loss)

  """ Test the model """

  guessed = model.predict(X=test_set)
  # print("\nML model guesses:")
  # print(guessed)
  # print("\nTrue Values")
  # print(test_coef)
  res = guessed - test_coef
  mean_err = np.mean(np.linalg.norm(res, axis=1))
  print(mean_err)
  # print(zern_coefs[N - 10:N])

  true = zern_coefs[N]
  res = guessed[0] - true
  res_norm = np.linalg.norm(res)
  rel_err = res_norm / np.linalg.norm(true)
  print('\nRelative error: %.3f' %rel_err)

  for k in range(N_zern):

      guess = guessed[:, k]
      coef = test_coef[:, k]
      plt.figure()
      plt.scatter(coef, guess)
      plt.plot(coef, coef, color='black')
      title = zern_list[k] + '  (N_train=%d, N_test=%d)' %(N_train - n_test, n_test)
      plt.title(title)
      plt.savefig(os.path.join(results_path, zern_list[k]))


  test = np.zeros((1, 2*N_pix**2))
  test_true = np.array([[0.023, 0.074, 0.010, 0.069, 0.025],
               [0.023, 0.023, 0.010, 0.025, 0.069],
               [0.015, 0.014, 0.069, 0.069, 0.025]])
  test_id = 2
  pop_slicer_test = pop.POP_Slicer()
  name_convention = 'SlicerTwisted_006f6a_POP_ML_TEST%d_' %test_id
  pop_slicer_test.get_zemax_files(path_zemax, name_convention,
                             start=slice_start, finish=slice_end, mode='irradiance')
  test_nom = np.sum(pop_slicer_test.beam_data, axis=0).flatten()

  name_convention = 'SlicerTwisted_006f6a_POP_ML_FOC_TEST%d_' %test_id
  pop_slicer_test.get_zemax_files(path_zemax, name_convention,
                             start=slice_start, finish=slice_end, mode='irradiance')
  test_foc = np.sum(pop_slicer_test.beam_data, axis=0).flatten()
  test[0,:] = np.concatenate((test_nom, test_foc))
  # FIXME! DO NOT FORGET TO NORMALIZE THE PSFs BY THE NOMINAL PEAK
  test /= peak_nom
  guessed = model.predict(X=test)
  print("\nML model guesses:")
  print(guessed)
  print('\nTrue Values')
  print(test_true[test_id-1] / z_norm)

  """ Compare the guess PSF """
  input = test[0,:N_pix**2].reshape((N_pix, N_pix))
  joint = np.concatenate((input, input), axis=1)

  plt.figure()
  plt.imshow(joint, origin='lower', cmap='jet')
  plt.colorbar(orientation='horizontal')
  plt.xlabel('Pixels')
  plt.ylabel('Pixels')
  plt.title('Input (left) and Guessed PSF (right)')
  # plt.savefig(os.path.join(results_path, 'Z%d' %i_ex))


  plt.show()