import os
import re
import numpy as np
import matplotlib.pyplot as plt
import pop_methods as pop

# Beam Parameters
wave = 0.8                          # Wavelength [microns]
N, M = 2048, 2048                   # Grid size

def read_zemax_header(file_name):
    """
    Reads the textfile containing the POP results
    and extracts the Beam information:
        - Slicer_ID: int
        - Display Width X and Height Y: float, float
        - Peak Irradiance, Total power: float, float
    :param file_name: path to the .txt file
    :return: [Slicer_ID, Width X, Height Y, Total Power]
    """
    with open(file_name, "r", encoding="utf-8") as file:
        for i in range(15):
            # print(i)
            current_line = file.readline()
            if i == 7:
                regex = r"\d+"
                numbers = re.findall(regex, current_line)
                slicer_id = int(numbers[-1])
            if i == 11:
                regex = r"[-+]?[0-9]+\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
                numbers = re.findall(regex, current_line)
                X_size, Y_size = float(numbers[0]), float(numbers[1])
            if i == 12:
                regex = r"[-+]?[0-9]+\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
                numbers = re.findall(regex, current_line)
                peak_irr, power = float(numbers[0]), float(numbers[1])

    return [slicer_id, X_size, Y_size, peak_irr, power]

def read_all_zemax_files(start=1, finish=55):
    """
    Goes through the .txt files of all Slices and
    extracts the beam information (X_size, Y_size) etc
    as well as the Irradiance distribution
    """

    beam_info = []
    irradiance_values = []
    path_zemax = 'zemax_files'

    for k in np.arange(start, finish+1):
        file_id = 'SlicerTwisted_006f6a_POP_' + str(k) + '_POP.txt'
        file_name = os.path.join(path_zemax, file_id)

        # Read only the Beam info (Slicer ID, X_size, Y_size, ...)
        beam_data = read_zemax_header(file_name)
        beam_info.append(beam_data)

        # Read the arrays
        data = np.loadtxt(file_name, skiprows=15)
        irradiance_values.append(data)

    beam_info = np.array(beam_info)
    irradiance_values = np.array(irradiance_values)

    return beam_info, irradiance_values

def crop_arrays(beam_info, irradiance_values, N_pix=512):
    N_slices = beam_info.shape[0]
    N = irradiance_values.shape[1]
    pix_min = N//2 - N_pix//2
    pix_max = N//2 + N_pix//2
    new_beam_info = beam_info.copy()
    new_irradiance = np.zeros((N_slices, N_pix, N_pix))
    for k in range(N_slices):
        new_beam_info[k, 1:3] *= (N_pix/N)
        data = irradiance_values[k]
        new_irradiance[k] = data[pix_min:pix_max, pix_min:pix_max]
    return new_beam_info, new_irradiance

def resample_grids(beam_info, irradiance_values):

    N_slices = beam_info.shape[0]
    N = irradiance_values.shape[1]
    M = irradiance_values.shape[2]

    # Find the widest grid
    i_max = beam_info[:, 1].argmax()
    j_max = beam_info[:, 2].argmax()

    # In case (i_max != j_max)
    if i_max != j_max:
        x_max = max(beam_info[i_max, 1], beam_info[j_max, 1])
        y_max = max(beam_info[j_max, 2], beam_info[i_max, 2])
        # No slicer is "exactly" the reference grid
        max_ID = 9999
        print('\nNo particular slice has both Max X and Max Y')
        print('Resampling all grids to (X_max, Y_max): ', (x_max, y_max))
        print('\n')
    else:
        x_max = beam_info[i_max, 1]
        y_max = beam_info[j_max, 2]
        max_ID = beam_info[i_max, 0]
        print('\nSlice #%d is the largest with (X_max, Y_max): (%f, %f)' %(max_ID, x_max, y_max))
        print('Resampling the rest of grids to that reference')
        print('\n')

    # Create the Reference Grid
    x = np.linspace(-x_max/2., x_max/2, num=N, endpoint=True)
    y = np.linspace(-y_max/2., y_max/2, num=M, endpoint=True)

    xx_ref, yy_ref = np.meshgrid(x, y, indexing='ij')
    ref_grid = np.array([xx_ref, yy_ref])
    print('Reference Grid Shape')
    print(ref_grid.shape)
    resampler = pop.ResampleGrid2D(ref_grid)

    grids = []
    new_values = []

    # Iterate over all slices
    for k in range(N_slices):
        print('\nSlice #%d' %(beam_info[k,0]))
        D_x, D_y = beam_info[k, 1], beam_info[k, 2]
        x = np.linspace(-D_x / 2., D_x / 2, num=N, endpoint=True)
        y = np.linspace(-D_y / 2., D_y / 2, num=M, endpoint=True)
        xx, yy = np.meshgrid(x, y, indexing='ij')
        print('Grid shape')
        grid = np.array([xx, yy])
        print(grid.shape)

        if beam_info[k,0] == max_ID: # No need to resample
            print('No need to resample')
            grids.append(grid)
            new_values.append(irradiance_values[k])

        else:
            new_grid, new_value = resampler.resample_grid(grid, irradiance_values[k])
            grids.append(new_grid)
            new_values.append(new_value)

    grids = np.array(grids)
    new_values = np.array(new_values)

    return grids, new_values

def collapse_data(new_values, beam_info):

    # Get the info about the different slices
    ids, X, Y = beam_info[:, 0], beam_info[:, 1], beam_info[:, 2]
    peak_irr, tot_power = beam_info[:, -2], beam_info[:, -1]

    plt.figure()
    plt.plot(ids, X, label='X_size')
    plt.plot(ids, Y, label='Y_size')
    plt.legend()
    plt.xlabel('Slice ID')
    plt.ylabel('Size [mm]')
    plt.title('Evolution of display size for each slice')

    plt.figure()
    plt.plot(ids, peak_irr)
    plt.yscale('log')
    plt.xlabel('Slice ID')
    plt.title('Peak Irradiance [W/mm^2]')

    plt.figure()
    plt.plot(ids, tot_power)
    plt.yscale('log')
    plt.xlabel('Slice ID')
    plt.title('Power contribution [W]')

    # Collapse the Irradiance data cube to get the total
    final_data = np.sum(new_values, axis=0)

    plt.figure()
    plt.imshow(final_data, origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Combined intensity (Scale=Linear)')

    plt.figure()
    plt.imshow(np.log10(final_data), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Combined intensity (Scale=Log10)')

if __name__ == "__main__":

    path_zemax = 'zemax_files'
    # file_id = 'SlicerTwisted_006f6a_POP_27_POP.txt'
    file_id = '27.txt'



    # with open(file_name, "r") as file:
    #     for i in range(15):
    #         current_line = file.readline()
    #         if i == 11:
    #             regex = r"[-+]?[0-9]+\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
    #             numbers = re.findall(regex, current_line)
    #             print(current_line)
    #             X = numbers[0]

    beam_info = []
    datas = []
    for k in [26, 27, 28, 29, 30]:
        file_id = str(k) + '.txt'

        file_name = os.path.join(path_zemax, file_id)

        numbers = read_zemax_header(file_name)
        data = np.loadtxt(file_name, delimiter='\t', skiprows=15)
        print(numbers)
        datas.append(data)
        beam_info.append(numbers)

        # plt.figure()
        # plt.imshow(data, origin='lower', cmap='jet')
        # plt.colorbar()

    datas = np.array(datas)
    beam_info = np.array(beam_info)

    new_beam_info, new_data = crop_arrays(beam_info, datas, N_pix=256)

    no_resampling = np.sum(new_data, axis=0)

    plt.figure()
    plt.imshow(no_resampling, origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Without Resampling')

    plt.figure()
    plt.imshow(np.log10(no_resampling), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Without Resampling (Log10 scale)')


    # resample
    grids, resampled_data = resample_grids(new_beam_info, new_data)

    resampled = np.sum(resampled_data, axis=0)

    plt.figure()
    plt.imshow(resampled, origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('With Resampling')

    plt.figure()
    plt.imshow(np.log10(resampled), origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('With Resampling (Log10 scale)')

    # Difference
    plt.figure()
    plt.imshow(resampled - no_resampling, origin='lower', cmap='jet')
    plt.colorbar()
    plt.title('Residual')


    plt.show()
