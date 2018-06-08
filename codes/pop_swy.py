# Physical Optics Propagation (POP) - effect of SemiWidth Y (SWY)
#
# This code analysis the effect of oversizing the Pupil mirrors on the
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
beam_rad = 150.           #[microns]

if __name__ == "__main__":

    # Total power per SLICE (55) per SEMI_WIDTH case (N_SWY)
    powers = np.empty((N_SWY, N_slices))
    # Ratio between Beam Radius @Mirror / Mirror SemiWidth_Y
    beam_ratios = np.empty(N_SWY)

    path_zemax = os.path.join('zemax_files', 'SWY')
    pop_slicer = pop.POP_Slicer()

    # Loop over all SEMIWIDTHS Y
    widths = np.linspace(MIN_SWY, MAX_SWY, N_SWY, endpoint=True, dtype=int)
    for (k, swy) in enumerate(widths):
        print("___________________________________________________")
        print("\nSEMI_WIDTH Y: %d [microns]" %swy)

        beam_ratios[k] = swy / beam_rad

        # #FIXME check whether there are whitespaces in the name
        # name_convention = 'SlicerTwisted_006f6a_SWY' + str(swy) + "_POP"
        #
        # pop_slicer.get_zemax_files(path_zemax, name_convention,
        #                            start=1, finish=55, mode='irradiance')
        # powers[k, :] = pop_slicer.powers.copy()


    plt.figure()
    plt.plot(beam_ratios, powers[:, i_central], label='Central')
    plt.plot(beam_ratios, powers[:, i_central + 1], label='Central + 1')
    plt.yscale('log')
    plt.xlabel('Ratio [SemiWidth_Y / BeamRadius]')
    plt.ylabel('Power ratio [ ]')
    plt.title('Evolution of transmitted irradiance vs Mirror/Beam ratio')

    plt.show()