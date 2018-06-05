import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import pop_methods as pop

if __name__ == "__main__":

    name_convention = 'SlicerTwisted_006f6a_POP_'
    path_zemax = os.path.join('zemax_files', 'SlicerTwisted_006f6a_POP')

    pop_slicer = pop.POP_Slicer()

    if sys.argv[-1] == 'load':
        raw_data = np.load('raw_data.npy')
        raw_info = np.load('raw_info.npy')
        pop_slicer.beam_data = raw_data
        pop_slicer.beam_info = raw_info
    else:
        pop_slicer.get_zemax_files(path_zemax, name_convention,
                                   start=1, finish=55, mode='irradiance')
        # Keep a copy after loading in case there is a bug in pop_slicer
        # Loading the files takes so long...!
        raw_data = pop_slicer.beam_data.copy()
        raw_info = pop_slicer.beam_info.copy()

    pop_slicer.crop_arrays(N_pix=128)
    grids, resampled_data = pop_slicer.resample_grids(mode='pyresample')

    no_resampling = np.sum(pop_slicer.cropped_beam_data, axis=0)

    resampled = np.sum(resampled_data, axis=0)

    v_max = max(no_resampling.max(), resampled.max())
    v_min = min(no_resampling[np.nonzero(no_resampling)].min(), resampled[np.nonzero(resampled)].min())

    # Add collapsed result to the Beam array
    final_data = np.concatenate((resampled[np.newaxis,:,:], resampled_data))
    pop.save_to_fits(os.path.join('fits_files','SlicerTwisted'), final_data)

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
    plt.imshow((np.abs(no_resampling - resampled)), origin='lower', cmap='jet')
    plt.colorbar()
    # plt.title('Residual')
    plt.title('Relative difference (With - Without)/Without [log10 scale]')


    plt.show()
