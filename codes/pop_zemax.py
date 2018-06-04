import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pop_methods as pop

# Beam Parameters
wave = 0.8                          # Wavelength [microns]

if __name__ == "__main__":

    name_convention = 'SlicerTwisted_006f6a_POP_'
    path_zemax = os.path.join('zemax_files', 'SlicerTwisted_006f6a_POP')
    info, data = pop.read_all_zemax_files(path_zemax, name_convention,
                                          start=1, finish=55,
                                          mode='irradiance')

    new_beam_info, new_data = pop.crop_arrays(info, data, N_pix=256)

    no_resampling = np.sum(new_data, axis=0)

    grids, resampled_data = pop.resample_grids(new_beam_info, new_data)

    resampled = np.sum(resampled_data, axis=0)

    v_max = max(no_resampling.max(), resampled.max())
    v_min = min(no_resampling[np.nonzero(no_resampling)].min(), resampled[np.nonzero(resampled)].min())

    """ Quick Way: Disregard any resampling """
    plt.figure()
    plt.imshow(no_resampling, origin='lower', vmin=v_min, vmax=v_max, cmap='jet')
    plt.colorbar()
    plt.title('Without Resampling')

    plt.figure()
    plt.imshow(np.log10(no_resampling), origin='lower', vmin=np.log10(v_min), vmax=np.log10(v_max), cmap='jet')
    plt.colorbar()
    plt.title('Without Resampling (Log10 scale)')

    """ Slow Painful Way: Do the proper resampling """
    plt.figure()
    plt.imshow(resampled, origin='lower', vmin=v_min, vmax=v_max, cmap='jet')
    plt.colorbar()
    plt.title('With Resampling')

    plt.figure()
    plt.imshow(np.log10(resampled), origin='lower', vmin=np.log10(v_min), vmax=np.log10(v_max), cmap='jet')
    plt.colorbar()
    plt.title('With Resampling (Log10 scale)')

    """ Comparison difference """
    plt.figure()
    plt.imshow(np.log10(np.abs(resampled - no_resampling)/no_resampling), origin='lower', cmap='jet')
    plt.colorbar()
    # plt.title('Residual')
    plt.title('Relative difference (With - Without)/Without [log10 scale]')


    plt.show()
