# Physical Optics Propagation (POP) - effect of Aperture
#
# This code runs the analysis of Light Loss effect due to the aperture
#
#
#

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pop_methods as pop

""" PARAMETERS """

# Slices
N_slices = 55
i_central = 27
beam_rad = 0.1434           #[mm]

# Mirror aperture
semi_aper = 0.825           #[mm]

# POP arrays
x_size = 4.
N_pix = 1024
extends = [-x_size / 2, x_size / 2, -x_size / 2, x_size / 2]
xc = np.linspace(-x_size / 2, x_size / 2, 10)

def crop_array(array, X_size, N_pix):

    N = array.shape[0]
    pix_min = N // 2 - N_pix // 2
    pix_max = N // 2 + N_pix // 2
    new_size = X_size * (N_pix/N)
    return array[pix_min:pix_max, pix_min:pix_max], new_size

if __name__ == "__main__":

   # Total power per SLICE (55) per SEMI_WIDTH case (N_SWY)
   powers = np.zeros(N_slices)
   irradiance = np.empty((N_slices, N_pix, N_pix))

   results_path = os.path.join('results', 'MIRROR')
   pop_slicer = pop.POP_Slicer()
   path_zemax = os.path.join('zemax_files', 'MIRROR')
   # FIXME This convention is Hard-Coded in pop_methods.py "_0018" to follow the stupid Zemax
   # because it doesn't follow the .config file when you propagate
   # to a surface different from the IMAGE ...
   name_convention = 'SlicerTwisted_006f6a_POP_MIRROR_'
   pop_slicer.get_zemax_files(path_zemax, name_convention,
                              start=1, finish=55, mode='both')
   powers[:] = pop_slicer.powers.copy()
   irradiance= pop_slicer.beam_data.copy()

   tot_irradiance = np.sum(irradiance, axis=0)
   tot_irradiance_norm = tot_irradiance / tot_irradiance.max()

   theta = np.linspace(0, 2*np.pi, 100)
   x_r, y_r = beam_rad * np.cos(theta), beam_rad * np.sin(theta)

   """ Total Irradiance (Linear Scale)"""
   plt.figure()
   plt.plot(xc, semi_aper * np.ones(10), color='white', linestyle='--', label='Mirror Aperture')
   plt.plot(xc, -semi_aper * np.ones(10), color='white', linestyle='--')
   plt.plot(x_r, y_r, color='black', label='Geometric Size')
   plt.imshow(tot_irradiance_norm, extent=extends)
   plt.legend()
   plt.colorbar()
   plt.xlabel('X [mm]')
   plt.ylabel('Y [mm]')
   plt.title('Power [W] (Normalized)')
   file_name = 'Power Normalized'
   plt.savefig(os.path.join(results_path, file_name))

   plt.figure()
   plt.plot(xc, semi_aper * np.ones(10), color='white', linestyle='--', label='Mirror Aperture')
   plt.plot(xc, -semi_aper * np.ones(10), color='white', linestyle='--')
   plt.plot(x_r, y_r, color='black', label='Geometric Size')
   plt.imshow(tot_irradiance, extent=extends)
   plt.legend()
   plt.colorbar()
   plt.xlabel('X [mm]')
   plt.ylabel('Y [mm]')
   plt.title('Power [W] | Total Power = %.6f' %(np.sum(tot_irradiance)))
   file_name = 'Power'
   plt.savefig(os.path.join(results_path, file_name))

   """ Total Irradiance (Log Scale)"""
   plt.figure()
   plt.plot(xc, semi_aper * np.ones(10), color='white', linestyle='--', label='Mirror Aperture')
   plt.plot(xc, -semi_aper * np.ones(10), color='white', linestyle='--')
   plt.plot(x_r, y_r, color='black', label='Geometric Size')
   plt.imshow(np.log10(tot_irradiance), extent=extends)
   plt.legend()
   plt.colorbar()
   plt.xlabel('X [mm]')
   plt.ylabel('Y [mm]')
   plt.title('Log10(Power [W]) | Total Power = %.6f' %(np.sum(tot_irradiance)))
   file_name = 'Log10 Power'
   plt.savefig(os.path.join(results_path, file_name))

   # Crop the areas outside the geometric radius
   x = np.linspace(-x_size / 2, x_size / 2, N_pix, endpoint=True)
   xx, yy = np.meshgrid(x, x)
   mask_geom = xx**2 + yy**2 <= beam_rad**2
   inside_geom = tot_irradiance * mask_geom
   power_geom = np.sum(inside_geom)
   print('Power inside the Geometrical Radius: %.4f [W]' %power_geom)

   # Crop the areas outside the mirror aperture
   mask_aper = np.abs(yy) <= semi_aper
   inside_aper = tot_irradiance * mask_aper
   power_aper = np.sum(inside_aper)
   print('Power inside the Mirror Aperture: %.4f [W]' %power_aper)

   max_masked = np.max(inside_aper[np.nonzero(inside_aper)])

   """ Total Irradiance (Log Scale)"""
   plt.figure()
   plt.plot(xc, semi_aper * np.ones(10), color='black', linestyle='--', label='Mirror Aperture')
   plt.plot(xc, -semi_aper * np.ones(10), color='black', linestyle='--')
   plt.plot(x_r, y_r, color='black', label='Geometric Size')
   plt.imshow(mask_aper*np.log10(tot_irradiance), vmax=np.log10(max_masked), extent=extends)
   plt.legend()
   plt.colorbar()
   plt.xlabel('X [mm]')
   plt.ylabel('Y [mm]')
   plt.title('Log10(Power [W]) | Power inside aperture = %.4f' %power_aper)
   file_name = 'Aperture Log10 Power'
   plt.savefig(os.path.join(results_path, file_name))

   plt.show()