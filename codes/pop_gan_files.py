# Physical Optics Propagation (POP) - Machine Learning
#
# Read Zemax Beam Files and create some slicer PSFs to be used in
# the training of Generative Adversarial Networks
#
#

import os
import sys
import numpy as np
from numpy.random import RandomState
import matplotlib.pyplot as plt
import pop_methods as pop

# Slices
slice_start = 24
slice_end = 32
i_central = (slice_end - slice_start)//2
N_slices = slice_end - slice_start + 1

# POP arrays
N_pix = 64
N_zern = 5
sampling = 5
N_train = sampling ** N_zern

bad_file = 2899

if __name__ == "__main__":

   path_zemax = os.path.join('zemax_files', 'ML_DEFOCUSING')
   results_path = os.path.join('results', 'GAN')

   slicer_PSFs = np.empty((N_train, N_pix, N_pix))

   pop_slicer = pop.POP_Slicer()

   for k in range(N_train):

       if k == bad_file:  # FIXME! Somehow the stupid file 2899 is broken
           slicer_PSFs[k, :] = np.zeros((N_pix, N_pix))

       else:
           if k < 10:
               name_nominal = 'SlicerTwisted_006f6a_POP_ML' + '% d_' % k
           else:
               name_nominal = 'SlicerTwisted_006f6a_POP_ML' + '%d_' % k

           pop_slicer.get_zemax_files(path_zemax, name_nominal,
                                      start=slice_start, finish=slice_end, mode='irradiance')
           slicer_PSFs[k, :, :] = np.sum(pop_slicer.beam_data, axis=0)

   PSFs = np.concatenate((slicer_PSFs[:bad_file, :, :], slicer_PSFs[bad_file+1:, :, :]), axis=0)
   peak_nom = np.max(PSFs[0])
   PSFs /= peak_nom

   print('Generated %d PSFs of (%d, %d) pixels' %(PSFs.shape[0], N_pix, N_pix))

   file_name = os.path.join(results_path, 'GAN_PSF')
   np.savez_compressed(file_name, psf=PSFs)

   # check that it worked
   compressed_file = os.path.join(results_path, 'GAN_PSF.npz')
   loaded = np.load(compressed_file)
   print('\nLoaded file of dimensions:')
   print(loaded['psf'].shape)

   plt.show()