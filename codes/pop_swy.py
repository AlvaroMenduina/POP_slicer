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
beam_rad = 145.           #[microns]

# ZBF
N_pix = 1024

x_size = 0.35
slice_start = 25
slice_end = 31
i_central = (slice_end - slice_start)//2
N_slices = slice_end - slice_start + 1

thres = 0.05

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
  phases = np.empty((N_SWY, N_pix, N_pix))
  phase_masks = np.empty((N_SWY, N_pix, N_pix))
  E_field = np.empty((N_SWY, N_slices, 4,  N_pix, N_pix))
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
                                 start=25, finish=31, mode='both')
      powers[k, :] = pop_slicer.powers.copy()
      irradiance[k, :, :, :] = pop_slicer.beam_data.copy()
      E_field[k, :, :] = pop_slicer.phases.copy()
      if k == 0:
          thres = 0.1
      else:
          thres = 0.05
      phases[k, :, :], phase_masks[k, :, :] = pop.compute_phase(E_field[k], threshold=thres, mode='all_slices')

  # powers = np.load(os.path.join(results_path, 'powers_SWY.npy'))
  # irradiance = np.load(os.path.join(results_path, 'irradiance_SWY.npy'))
  # phase = np.load(os.path.join(results_path, 'phase_SWY.npy'))
  #
  # np.save(os.path.join(results_path, 'powers_SWY'), powers)
  # np.save(os.path.join(results_path, 'irradiance_SWY'), irradiance)
  # np.save(os.path.join(results_path, 'phase_SWY'), phase)

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
      # croped, x_size = crop_array(irrad, X_size=x_size, N_pix=n_pix)
      extends = [-x_size/2, x_size/2, -x_size/2, x_size/2]

      xc = np.linspace(-x_size/2, x_size/2, 10)

      plt.figure()
      plt.plot(xc, 0.03*np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.plot(xc, -0.03 * np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.plot(xc, 0.09*np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.plot(xc, -0.09 * np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.imshow(np.log10(irrad), extent=extends)
      plt.colorbar()
      plt.xlabel('X [mm]')
      plt.ylabel('Y [mm]')
      plt.title('Log10(Power [W]) Central Slice (SemiWidth Y: %d [um])' %swy)
      plt.savefig('Log10Power_CentralSlice_SWY_' + str(swy) + 'um_Npix_' +str(n_pix))

  for (k, swy) in enumerate(widths):
      ratio = beam_ratios[k]
      irrad = np.sum(irradiance, axis=1)[k]
      # croped, x_size = crop_array(irrad, X_size=x_size, N_pix=n_pix)
      extends = [-x_size / 2, x_size / 2, -x_size / 2, x_size / 2]

      xc = np.linspace(-x_size/2, x_size/2, 10)

      plt.figure()
      plt.plot(xc, 0.03*np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.plot(xc, -0.03 * np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.plot(xc, 0.09*np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.plot(xc, -0.09 * np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.imshow(irrad, extent=extends)
      plt.colorbar()
      plt.xlabel('X [mm]')
      plt.ylabel('Y [mm]')
      plt.title('Log10(Power [W]) (SemiWidth Y: %d [um])' % swy)
      plt.savefig('Log10(Total_Power)_SWY_' + str(swy) + 'um_Npix_' + str(n_pix))

  """ 2D Phase Plots"""
  RMS = []
  RMS_nm = []
  for (k, swy) in enumerate(widths):
      ratio = beam_ratios[k]
      phase = 800 * phases[k] * phase_masks[k]
      clean_phase = phase[np.nonzero(phase)]
      vmin, vmax = np.min(clean_phase), np.max(clean_phase)
      n = clean_phase.shape[0]
      phase_mean = np.mean(clean_phase)
      rms = np.sqrt(1./n * np.sum((clean_phase - phase_mean)**2))
      RMS.append(rms/800)
      RMS_nm.append(rms)
      extends = [-x_size / 2, x_size / 2, -x_size / 2, x_size / 2]

      plt.figure()
      plt.plot(xc, 0.03*np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.plot(xc, -0.03 * np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.plot(xc, 0.09*np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.plot(xc, -0.09 * np.ones(10), color='white', linestyle='--', alpha=0.5)
      plt.imshow(phase, vmax=vmax, vmin=vmin, extent=extends)
      cbar = plt.colorbar()
      cbar.ax.set_ylabel('nm')
      plt.xlabel('X [mm]')
      plt.ylabel('Y [mm]')
      plt.xlim([-0.1, 0.1])
      plt.ylim([-0.1, 0.1])
      plt.title('Phase (SemiWidth Y: %d [um]) RMS: %.1f[nm]' %(swy, rms))
      plt.savefig('Phase_SWY_' + str(swy) + '_nm')


  plt.figure()
  plt.plot(beam_ratios, RMS_nm, label='Diffraction')
  plt.plot(2.5*np.ones(len(beam_ratios)), color='black', linestyle='--', label='Geometrical')
  plt.legend()
  # plt.ylim([0, np.max(RMS_nm)])
  plt.xlabel('Ratio [SemiWidth_Y / BeamRadius]')
  plt.ylabel('RMS [nm]')
  plt.title('RMS wavefront error [nm]')
  plt.savefig('Wavefront')

  RMS = np.array(RMS)
  strehl = np.exp(-RMS**2)


  FWHM_slice = []
  maxmax = np.max(np.sum(irradiance, axis=1))
  for (k, swy) in enumerate(widths):
      ratio = beam_ratios[k]
      irrad = np.sum(irradiance, axis=1)[k]
      extends = [-x_size / 2, x_size / 2, -x_size / 2, x_size / 2]
      x_proj = irrad.copy()[N_pix//2, :]
      y_proj = irrad.copy()[:, N_pix//2]
      peak = np.max(x_proj)

      x_proj /= maxmax
      y_proj /= maxmax
      i_FWHM = np.argsort((np.abs(x_proj - 0.5*peak/maxmax)))[:2]
      j_FWHM = np.argsort((np.abs(y_proj - 0.5*peak/maxmax)))[:2]

      x = np.linspace(-x_size / 2, x_size / 2, N_pix, endpoint=True)
      x_min = x[i_FWHM[0]]
      x_max = x[i_FWHM[1]]
      FWHM_X = np.abs(x_min - x_max)
      y_min = x[j_FWHM[0]]
      y_max = x[j_FWHM[1]]
      FWHM_Y = np.abs(y_min - y_max)
      FWHM_slice.append(FWHM_Y)
      x_minus = min([x_min, x_max, y_min, y_max])
      x_plus = max([x_min, x_max, y_min, y_max])
      plt.figure()
      plt.plot(x, x_proj, label='Along Slice (X)')
      plt.plot(x, y_proj, label='Across Slice (Y)')
      plt.axhline(y=0.5*peak/maxmax, xmin=x_minus, color='black', linestyle='--')
      plt.axvline(x=x[i_FWHM[0]], ymax=0.5*peak/maxmax, color='black', linestyle='--')
      plt.axvline(x=x[i_FWHM[1]], ymax=0.5*peak/maxmax, color='black', linestyle='--')
      plt.axvline(x=x[j_FWHM[0]], ymax=0.5*peak/maxmax, color='black', linestyle='--')
      plt.axvline(x=x[j_FWHM[1]], ymax=0.5*peak/maxmax, color='black', linestyle='--')
      plt.legend()
      plt.xlabel('X [mm]')
      plt.ylim([0, 1])
      plt.title('FWHM_X = %.1f um, FWHM_Y = %.1f um (SemiWidth Y: %d [um])' %(FWHM_X*1000, FWHM_Y*1000, swy))
      plt.savefig('FWHM SWY %d' %swy)

  fwhm = np.array(FWHM_slice)

  plt.show()
