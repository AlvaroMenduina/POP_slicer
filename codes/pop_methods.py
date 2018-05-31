import numpy as np
import matplotlib.pyplot as plt


N = 1024
x_ref = np.linspace(-10, 10, N)
f = x_ref ** 2

x = np.linspace(-9, 9, N)
g = x

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
            for j in np.arange(1,M-1):

                xy = xy_grid[:,i,j]
                x, y = xy[0], xy[1]

                ij_ref, xy_ref = self.find_nearest(xy)
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

N = 128
x_ref = np.linspace(-5.1, 5.1, N)
xx_ref, yy_ref = np.meshgrid(x_ref, x_ref, indexing='ij')
ff = xx_ref**2 - yy_ref**2
ref_grid = np.array([xx_ref, yy_ref])

x = np.linspace(-5, 5, N)
xx, yy = np.meshgrid(x, x, indexing='ij')
grid = np.array([xx, yy])
gg = xx + yy


resample = ResampleGrid2D(ref_grid)
new_grid, new_values = resample.resample_grid(grid, gg)

plt.figure()
plt.imshow(new_values)
plt.colorbar()

plt.figure()
plt.imshow(gg)
plt.colorbar()

plt.show()
