import os
import numpy as np
import matplotlib.pyplot as plt
from pyzdde.zdde import readBeamFile
from astropy.io import fits
from time import time as tm
from pyresample import utils, image, geometry

# ============================================================================== #
#                                .FITS FILES                                     #
# ============================================================================== #

def save_to_fits(file_name, beam_data):

    hdul = fits.PrimaryHDU()
    hdul.data = beam_data

    fits_name = file_name + '.fits'
    hdul.writeto(fits_name, overwrite=True)
    print('\nSaving file: ', fits_name)

# ============================================================================== #
#                                ZEMAX INTERFACE                                 #
# ============================================================================== #

def read_beam_file(file_name, mode='irradiance'):
    """
    Reads a Zemax Beam File and returns either the Irradiance or the Phase
    of the Magnetic field E
    """
    beamData = readBeamFile(file_name)
    (version, (nx, ny), ispol, units, (dx, dy), (zposition_x, zposition_y),
     (rayleigh_x, rayleigh_y), (waist_x, waist_y), lamda, index, re, se,
     (x_matrix, y_matrix), (Ex_real, Ex_imag, Ey_real, Ey_imag)) = beamData

    area = (1e-3*nx*dx) * (1e-3*ny*dy)

    E_real = np.array([Ex_real, Ey_real])
    E_imag = np.array([Ex_imag, Ey_imag])

    re = np.linalg.norm(E_real, axis=0)
    im = np.linalg.norm(E_imag, axis=0)

    #FIXME why are the arrays flipped?

    if mode=='irradiance':
        irradiance = (re ** 2 + im ** 2).T
        power = np.sum(irradiance)
        print('Total Power: ', power)
        p = np.max(irradiance)
        print('Peak Irradiance: %f [W/m^2]' %(p/area))
        return (nx, ny), (dx, dy), irradiance, power

    if mode=='phase':
        phase = np.arctan2(im, re).T
        return (nx, ny), (dx, dy), phase

def read_all_zemax_files(path_zemax, name_convention, start=1, finish=55, mode='irradiance'):
    """
    Goes through the ZBF Zemax Beam Files of all Slices and
    extracts the beam information (X_size, Y_size) etc
    as well as the Irradiance distribution
    """
    info = []
    data = []
    powers = []

    for k in np.arange(start, finish+1):
        print('\n======================================')

        if k < 10:
            # file_id = 'SlicerTwisted_006f6a_POP_ ' + str(k) + '_POP.ZBF'
            file_id = name_convention + ' ' + str(k) + '_POP.ZBF'
        else:
            # file_id = 'SlicerTwisted_006f6a_POP_' + str(k) + '_POP.ZBF'
            file_id = name_convention + str(k) + '_POP.ZBF'
        file_name = os.path.join(path_zemax, file_id)

        print('Reading Beam File: ', file_id)

        NM, deltas, beam_data, _power = read_beam_file(file_name, mode=mode)
        Dx, Dy = NM[0] * deltas[0], NM[1] * deltas[1]
        info.append([k, Dx, Dy])
        data.append(beam_data)
        powers.append(powers)

    beam_info = np.array(info)
    irradiance_values = np.array(data)
    powers = np.array(powers)

    # print('\nTime spent LOADING files: %.1f [sec]' %(tm() - start))

    return beam_info, irradiance_values, powers

# ============================================================================== #
#                                GRID RESAMPLING                                 #
# ============================================================================== #

class POP_Slicer(object):
    """
    Physical Optics Propagation (POP) analysis of an Image Slicer


    """

    def __init__(self):
        pass

    def get_zemax_files(self, zemax_path, name_convention, start, finish, mode):
        _info, _data, _powers = read_all_zemax_files(zemax_path, name_convention, start, finish, mode)
        self.beam_info = _info
        self.beam_data = _data
        self.powers = _powers

    def crop_arrays(self, N_pix=512):

        N_slices = self.beam_info.shape[0]
        N = self.beam_data.shape[1]
        pix_min = N // 2 - N_pix // 2
        pix_max = N // 2 + N_pix // 2
        new_beam_info = self.beam_info.copy()
        new_irradiance = np.zeros((N_slices, N_pix, N_pix))
        for k in range(N_slices):
            new_beam_info[k, 1:3] *= (N_pix / N)
            data = self.beam_data[k]
            new_irradiance[k] = data[pix_min:pix_max, pix_min:pix_max]

        self.cropped_beam_info = new_beam_info
        self.cropped_beam_data = new_irradiance

    def pyresample_method(self, ref_grid_lim, grid_lim, data):
        """
        Method which uses the library "pyresample" which is must faster
        """
        # FIXME: Understand how Proj_Dict influences the result

        X_MAX, Y_MAX = ref_grid_lim
        x_max, y_max = grid_lim

        N, M = data.shape

        proj_dict = {'a': '6371228.0', 'units': 'm', 'lon_0': '0',
                     'proj': 'laea', 'lat_0': '-90'}
        ref_area = geometry.AreaDefinition(area_id='REF', name='reference', proj_id='A',
                                           proj_dict=proj_dict, x_size=N, y_size=M,
                                           area_extent=[-X_MAX, -Y_MAX, X_MAX, Y_MAX])

        init_area = geometry.AreaDefinition(area_id='INIT', name='initial', proj_id='A',
                                            proj_dict=proj_dict, x_size=N, y_size=M,
                                            area_extent=[-x_max, -y_max, x_max, y_max])

        base = image.ImageContainer(data, init_area)
        row_indices, col_indices = utils.generate_quick_linesample_arrays(init_area, ref_area)
        result = base.get_array_from_linesample(row_indices, col_indices)
        return result

    def resample_grids(self, mode='pyresample'):
        """
        Takes care of resampling each of the Slice Grids to a
        Reference Grid which is the widest of all.

        The aim is for all Irradiance arrays to have the same
        physical dimensions! Although they initially have the same
        number of pixels, they represent slightly different X, Y
        dimensions.

        That information is stored in the Beam Info arrays
        and comes from the Zemax Beam Files

        modes:
            - 'homemade': hard-coded slow method but it works
            - 'pyresample': fast library but not 100% sure
        """

        try:
            beam_info = self.cropped_beam_info
            irradiance_values = self.cropped_beam_data
        except AttributeError:
            beam_info = self.beam_info
            irradiance_values = self.beam_data

        N_slices = beam_info.shape[0]
        N = irradiance_values.shape[1]
        M = irradiance_values.shape[2]

        print('Grids to resample: (N, M) = (%d, %d)' %(N, M))
        start = tm()

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
        self.x_array = x
        self.y_array = y

        xx_ref, yy_ref = np.meshgrid(x, y, indexing='ij')
        ref_grid = np.array([xx_ref, yy_ref])
        print('Reference Grid Shape')
        print(ref_grid.shape)
        resampler = ResampleGrid2D(ref_grid)

        grids = []
        new_values = []

        # Iterate over all slices
        for k in range(N_slices):
            print('\nSlice #%d' %(beam_info[k,0]))
            D_x, D_y = beam_info[k, 1], beam_info[k, 2]
            x = np.linspace(-D_x / 2., D_x / 2, num=N, endpoint=True)
            y = np.linspace(-D_y / 2., D_y / 2, num=M, endpoint=True)
            xx, yy = np.meshgrid(x, y, indexing='ij')
            grid = np.array([xx, yy])

            # Check whether we resample or pass
            if beam_info[k,0] == max_ID:
                print('No need to resample')
                grids.append(grid)
                new_values.append(irradiance_values[k])

            else:
                if mode=='homemade':
                    new_grid, new_value = resampler.resample_grid(grid, irradiance_values[k])
                    grids.append(new_grid)
                    new_values.append(new_value)
                if mode=='pyresample':
                    new_value = self.pyresample_method([x_max/2, y_max/2], [D_x/2, D_y/2], irradiance_values[k])
                    grids.append(ref_grid)
                    new_values.append(new_value)

        grids = np.array(grids)
        new_values = np.array(new_values)

        speed = (tm() - start)
        print('\nTotal time to resample: %.1f [sec]' %speed)
        print('Average time per slice: %.1f [sec]' %(speed/N_slices))
        print('Average time per slice per grid node: %.1f [sec]' %(speed/N_slices/N/M))

        return grids, new_values

class ResampleGrid1D(object):
    """
    Takes a 1D function defined by (Grid, Values)
    and resamples it to a Reference Grid
    """
    def __init__(self, ref_grid):
        self.ref_grid = ref_grid

    def find_nearest(self, array, value):
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    def interpolate(self, i_left, i_ref, i_right, data):
        x, f = data[0], data[1]
        x_left, x_right = x[i_left], x[i_right]
        f_left, f_right = f[i_left], f[i_right]
        x_ref = self.ref_grid[i_ref]
        f_ref = f_left + (x_ref - x_left)/(x_right - x_left)*(f_right - f_left)
        return f_ref

    def resample_grid(self, x_grid, f_values):
        new_grid = self.ref_grid.copy()
        new_values = np.zeros_like(x_grid)
        N = x_grid.shape[0]

        for (i, x) in enumerate(x_grid):
            # Find nearest grid point in Reference Grid
            i_ref, x_ref = self.find_nearest(self.ref_grid, x)
            print(i, i_ref)
            print(x, x_ref)
            if (x_ref > x) and (i < (N-1)): # forward interpolation
                print('fw')
                f_new = self.interpolate(i_left=i, i_ref=i_ref, i_right=i+1, data=[x_grid, f_values])
                new_grid[i_ref] = self.ref_grid[i_ref]
                new_values[i_ref] = f_new
            if (x_ref < x) and (i > 0): # backward interpolation
                print('bw')
                f_new = self.interpolate(i_left=i-1, i_ref=i_ref, i_right=i, data=[x_grid, f_values])
                new_grid[i_ref] = self.ref_grid[i_ref]
                new_values[i_ref] = f_new
            if (x_ref > x) and (i >= (N - 1)):  # forward interpolation
                print('bw')
                f_new = self.interpolate(i_left=i-1, i_ref=i_ref, i_right=i, data=[x_grid, f_values])
                new_grid[i_ref] = self.ref_grid[i_ref]
                new_values[i_ref] = f_new

        return new_grid, new_values

class ResampleGrid2D(object):
    """
    Takes a 2D function defined by (Grid, Values)
    and resamples it to a Reference Grid

    Grids are stored as [[XX], [YY]] with dimensions [2, N, M]
    """
    def __init__(self, ref_grid):
        self.ref_grid = ref_grid

    def find_nearest(self, XY):
        """
        For a given position XY (X_i, Y_j) in the Grid
        Find the nearest node on the Reference Grid
        onto which the interpolation will be done
        :param XY: a list containing [X_i, Y_j]
        :return: [i_ref, j_ref], [X_ref, Y_ref]
        """
        xx, yy = self.ref_grid[0,:,:], self.ref_grid[1,:,:]
        r2 = (xx - XY[0])**2 + (yy - XY[1])**2
        idx = np.unravel_index(np.argmin(r2, axis=None), r2.shape)
        return idx, (xx[idx], yy[idx])

    def interpolate_bilinear(self, x, y, F):
        """
        Formula for the Bilinear interpolation
        :param x: list containing (x1, x_ref, x2)
        :param y: list containing (y1, x_ref, y2)
        :param F: matrix containing the values of f at [[11, 12],
                                                        21, 22]]
        :return: value of the interpolator at (x_ref, y_ref)

        (x1, y2) --------------------- (x2, y2)
           |                              |
           |                              |
           |            (X_ref, Y_ref)    |
           |                   *          |
           |                              |
        (x1, y1) --------------------- (x2, y1)

        """
        x1, x_ref, x2 = x
        y1, y_ref, y2 = y
        delta_x = (x2 - x1)
        delta_y = (y2 - y1)
        deltas = 1. / delta_x / delta_y
        x_vec = np.array([x2 - x_ref, x_ref - x1])
        y_vec = np.array([y2 - y_ref, y_ref - y1])
        f_ref = deltas * np.dot(x_vec, np.dot(F, y_vec))
        return f_ref

    def interpolate(self, ij, ij_ref, data, mode='fu'):
        """
        Takes care of the interpolation for a given (x_i, y_j)
        Depending on the relative position of (x_i, y_j) and the
        interpolation point (x_ref, y_ref) it takes points
        located "forward/backward" or "up/down" of (x_i, y_j)
        :param ij: list containing the indices of the point in the Grid
        :param ij_ref: list containing the indices of reference point in the  Reference Grid
        :param data: list containing [Grid, Values]
        :param mode: how to select the other points of the interpolation
                        - "fu" Forward - Up (i+1, j+1)
                        - "fd" Forward - Down (i+1, j-1)
                        - "bd" Backward - Down (i-1, j-1)
                        - "bu" Backward - Up (i-1, j+1)
        :return: value of the interpolation at (x_ref, y_ref)
        """
        xy, ff = data[0], data[1]
        i, j = ij[0], ij[1]
        i_ref, j_ref = ij_ref[0], ij_ref[1]
        xi, yi = xy[0, i, j], xy[1, i, j]
        x_ref, y_ref = self.ref_grid[0, i_ref, j_ref], self.ref_grid[1, i_ref, j_ref]

        if mode=="fu":
            x = (xi, x_ref, xy[0, i+1, j])
            y = (yi, y_ref, xy[1, i, j+1])
            F = np.array([[ff[i, j], ff[i, j+1]],
                      [ff[i+1, j], ff[i+1, j+1]]])
            f_ref = self.interpolate_bilinear(x, y, F)

        if mode=="fd":
            x = (xi, x_ref, xy[0, i+1, j])
            y = (yi, y_ref, xy[1, i, j-1])
            F = np.array([[ff[i, j], ff[i, j-1]],
                      [ff[i+1, j], ff[i+1, j-1]]])
            f_ref = self.interpolate_bilinear(x, y, F)

        if mode=="bd":
            x = (xi, x_ref, xy[0, i-1, j])
            y = (yi, y_ref, xy[1, i, j-1])
            F = np.array([[ff[i, j], ff[i, j-1]],
                      [ff[i-1, j], ff[i-1, j-1]]])
            f_ref = self.interpolate_bilinear(x, y, F)

        if mode=="bu":
            x = (xi, x_ref, xy[0, i-1, j])
            y = (yi, y_ref, xy[1, i, j+1])
            F = np.array([[ff[i, j], ff[i, j+1]],
                      [ff[i-1, j], ff[i-1, j+1]]])
            f_ref = self.interpolate_bilinear(x, y, F)

        # Make sure your interpolation doesn't give you a NEGATIVE value
        if f_ref < 0.0:
            f_ref = ff[i, j]

        return f_ref

    def resample_grid(self, xy_grid, f_values):
        """
        Main function which receives the Grid and the Values
        and takes care of resampling the Grid onto the Reference Grid
        :param xy_grid: Grid size:[2, N, M]
        :param f_values: Values size: [N, M]
        :return: (resampled) Grid and (interpolated) Values
        """
        new_grid = self.ref_grid.copy()
        new_values = np.zeros_like(xy_grid[0])
        N, M = xy_grid[0].shape

        for i in np.arange(1,N-1):
            # print('Row #%d' %i)
            for j in np.arange(1,M-1):

                xy = xy_grid[:,i,j]
                x, y = xy[0], xy[1]

                # start = tm()
                ij_ref, xy_ref = self.find_nearest(xy)
                # print('Find Nearest: %f [sec]' %(tm() - start))
                i_ref, j_ref = ij_ref[0], ij_ref[1]
                x_ref, y_ref = xy_ref[0], xy_ref[1]

                if (x_ref > x) and (y_ref > y):  # forward & up
                    f_new = self.interpolate([i,j], ij_ref, [xy_grid, f_values], mode='fu')
                    new_grid[:,i_ref, j_ref] = self.ref_grid[:,i_ref,j_ref]
                    new_values[i_ref, j_ref] = f_new

                if (x_ref > x) and (y_ref < y):  # forward & down
                    f_new = self.interpolate([i,j], ij_ref, [xy_grid, f_values], mode='fd')
                    new_grid[:,i_ref, j_ref] = self.ref_grid[:,i_ref,j_ref]
                    new_values[i_ref, j_ref] = f_new

                if (x_ref < x) and (y_ref < y):  # backward & down
                    f_new = self.interpolate([i,j], ij_ref, [xy_grid, f_values], mode='fd')
                    new_grid[:,i_ref, j_ref] = self.ref_grid[:,i_ref,j_ref]
                    new_values[i_ref, j_ref] = f_new

                if (x_ref < x) and (y_ref > y):  # backward & up
                    f_new = self.interpolate([i,j], ij_ref, [xy_grid, f_values], mode='fu')
                    new_grid[:,i_ref, j_ref] = self.ref_grid[:,i_ref,j_ref]
                    new_values[i_ref, j_ref] = f_new

        return new_grid, new_values

plt.show()
