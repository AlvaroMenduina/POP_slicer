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
    with open(file_name, "r") as file:
        for i in range(15):
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

def read_all_zemax_files(start, finish):

    beam_info = []
    beam_values = []
    path_zemax = 'zemax_files'

    for k in np.arange(start, finish):
        file_id = 'SlicerTwisted_006f6a_POP_' + str(k) + '_POP.txt'
        file_name = os.path.join(path_zemax, file_id)

        # Read only the Beam info (Slicer ID, X_size, Y_size, ...)
        beam_data = read_zemax_header(file_name)
        beam_info.append(beam_data)

        # Read the arrays
        data = np.loadtxt(file_name, skiprows=15)
        beam_values.append(data)

    beam_info = np.array(beam_info)
    beam_values = np.array(beam_values)

    return


if __name__ == "__main__":

    path_zemax = 'zemax_files'
    # file_id = 'SlicerTwisted_006f6a_POP_27_POP.txt'
    file_id = 'test.txt'

    file_name = os.path.join(path_zemax, file_id)

    numbers = read_zemax_header(file_name)

    # data = np.loadtxt(file_name, skiprows=15)

    # with open(file_name, "r") as file:
    #     for i in range(15):
    #         current_line = file.readline()
    #         if i == 11:
    #             regex = r"[-+]?[0-9]+\.?[0-9]+(?:[eE][-+]?[0-9]+)?"
    #             numbers = re.findall(regex, current_line)
    #             print(current_line)
    #             X = numbers[0]


    plt.show()
