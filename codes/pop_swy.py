# Physical Optics Propagation (POP) - effect of SemiWidth Y (SWY)
#
# This code runs the analysis of the effect of oversizing the Pupil mirrors on the
# propagation of light through the PCS Image Slicer.
#
#
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pop_methods as pop

""" PARAMETERS """

# SemiWidth Y values [MICRONS]
MIN_SWY = 15
MAX_SWY = 825
N_SWY = 7
DELTA_SWY = (MAX_SWY - MIN_SWY)//(N_SWY - 1)

# Slices
N_slices = 55
i_central = 27
beam_rad = 145.           #[microns]

# ZBF
N_pix = 1024

def crop_array(array, X_size, N_pix):

    N = array.shape[0]
    pix_min = N // 2 - N_pix // 2
    pix_max = N // 2 + N_pix // 2
    new_size = X_size * (N_pix/N)
    return array[pix_min:pix_max, pix_min:pix_max], new_size


if __name__ == "__main__":

   # Total power per SLICE (55) per SEMI_WIDTH case (N_SWY)
   powers = np.empty((N_SWY, N_slices))
   # Ratio between Beam Radius @Mirror / Mirror SemiWidth_Y
   beam_ratios = np.empty(N_SWY)
   irradiance = np.empty((N_SWY, N_slices, N_pix, N_pix))
   phase = np.empty((N_SWY, N_slices, N_pix, N_pix))

   results_path = os.path.join('results', 'SWY')

   pop_slicer = pop.POP_Slicer()

   # Loop over all SEMIWIDTHS Y
   widths = np.linspace(MIN_SWY, MAX_SWY, N_SWY, endpoint=True, dtype=int)
   for (k, swy) in enumerate(widths):
       print("___________________________________________________")
       print("\nSEMI_WIDTH Y: %d [microns]" %swy)

       beam_ratios[k] = swy / beam_rad

       path_zemax = os.path.join('zemax_files', 'SWY' + str(swy))
       name_convention = 'SlicerTwisted_006f6a_POP_SMY' + str(swy) + '_'

       pop_slicer.get_zemax_files(path_zemax, name_convention,
                                  start=1, finish=55, mode='both')
       powers[k, :] = pop_slicer.powers.copy()
       irradiance[k, :, :, :] = pop_slicer.beam_data.copy()
       phase[k, :, :, :] = pop_slicer.phases.copy()

   powers = np.load(os.path.join(results_path, 'powers_SWY.npy'))
   irradiance = np.load(os.path.join(results_path, 'irradiance_SWY.npy'))
   phase = np.load(os.path.join(results_path, 'phase_SWY.npy'))

   np.save(os.path.join(results_path, 'powers_SWY'), powers)
   np.save(os.path.join(results_path, 'irradiance_SWY'), irradiance)
   np.save(os.path.join(results_path, 'phase_SWY'), phase)

   """ Total Power [Watts] """
   """ Central Slice Only"""
   plt.figure()
   plt.plot(beam_ratios, powers[:, i_central], label='Central')
   plt.plot(beam_ratios, powers[:, i_central + 1], label='Central + 1')
   plt.legend(title='Slice')
   # plt.yscale('log')
   plt.xlabel('Ratio [SemiWidth_Y / BeamRadius]')
   plt.ylabel('Power ratio [ ]')
   plt.yticks(np.linspace(0, 1., 11))
   plt.title('Evolution of transmitted irradiance vs Mirror/Beam ratio')

   """ Central Slice Only"""
   plt.figure()
   plt.plot(beam_ratios, powers[:, i_central], label='Central')
   plt.scatter(beam_ratios, powers[:, i_central])
   plt.xlabel('Ratio [SemiWidth_Y / BeamRadius]')
   plt.ylabel('Power ratio [ ]')
   plt.yticks(np.linspace(0, 1., 11))

   """ All Slices"""
   tot_pow = np.sum(powers, axis=1)
   plt.figure()
   plt.plot(beam_ratios, tot_pow, label='All Slices')
   plt.plot(beam_ratios, powers[:, i_central], label='Central Only')
   plt.legend(title='Slices')
   plt.ylim([0,1])
   plt.xlabel('Ratio [SemiWidth_Y / BeamRadius]')
   plt.ylabel('Power ratio [ ]')
   # plt.yscale('log')
   plt.yticks(np.linspace(0, 1., 11))

   """ 2D Irradiance Plots"""
   """ Central Slice Only"""
   n_pix = 64
   for (k, swy) in enumerate(widths):
       ratio = beam_ratios[k]
       irrad = irradiance[k, i_central, :, :]
       croped, x_size = crop_array(irrad, X_size=10., N_pix=n_pix)
       extends = [-x_size/2, x_size/2, -x_size/2, x_size/2]

       xc = np.linspace(-x_size/2, x_size/2, 10)

       plt.figure()
       plt.plot(xc, 0.03*np.ones(10), color='white', linestyle='--', alpha=0.5)
       plt.plot(xc, -0.03 * np.ones(10), color='white', linestyle='--', alpha=0.5)
       plt.plot(xc, 0.09*np.ones(10), color='white', linestyle='--', alpha=0.5)
       plt.plot(xc, -0.09 * np.ones(10), color='white', linestyle='--', alpha=0.5)
       plt.imshow(np.log10(croped), extent=extends)
       plt.colorbar()
       plt.xlabel('X [mm]')
       plt.ylabel('Y [mm]')
       plt.title('Log10(Power [W]) Central Slice (SemiWidth Y: %d [um])' %swy)
       plt.savefig('Log10Power_CentralSlice_SWY_' + str(swy) + 'um_Npix_' +str(n_pix))

   for (k, swy) in enumerate(widths):
       ratio = beam_ratios[k]
       irrad = np.sum(irradiance, axis=1)[k]
       croped, x_size = crop_array(irrad, X_size=10., N_pix=n_pix)
       extends = [-x_size / 2, x_size / 2, -x_size / 2, x_size / 2]

       xc = np.linspace(-x_size/2, x_size/2, 10)

       plt.figure()
       plt.plot(xc, 0.03*np.ones(10), color='white', linestyle='--', alpha=0.5)
       plt.plot(xc, -0.03 * np.ones(10), color='white', linestyle='--', alpha=0.5)
       plt.plot(xc, 0.09*np.ones(10), color='white', linestyle='--', alpha=0.5)
       plt.plot(xc, -0.09 * np.ones(10), color='white', linestyle='--', alpha=0.5)
       plt.imshow(np.log10(croped), extent=extends)
       plt.colorbar()
       plt.xlabel('X [mm]')
       plt.ylabel('Y [mm]')
       plt.title('Log10(Power [W]) (SemiWidth Y: %d [um])' % swy)
       plt.savefig('Log10(Total_Power)_SWY_' + str(swy) + 'um_Npix_' + str(n_pix))

   """ 2D Phase Plots"""
   for (k, swy) in enumerate(widths):
       ratio = beam_ratios[k]
       irr = np.sum(irradiance, axis=1)[k]
       pha = np.sum(phase, axis=1)[k]
       # pha = phase[k, i_central, :, :]
       # irr = irradiance[k, i_central, :, :]
       mask = np.argwhere(irr > 1e-6)
       masked_phase = np.empty(pha.shape)
       masked_phase.fill(np.nan)
       for ij in mask:
           i, j = ij[0], ij[1]
           masked_phase[i, j] = pha[i, j]

       croped, x_size = crop_array(masked_phase, X_size=10., N_pix=64)
       extends = [-x_size / 2, x_size / 2, -x_size / 2, x_size / 2]

       plt.figure()
       plt.imshow(croped, extent=extends)
       plt.colorbar()
       plt.xlabel('X [mm]')
       plt.ylabel('Y [mm]')
       plt.title('Phase (SemiWidth Y: %d [um])' % swy)
       if k==0:
           break

   plt.show()