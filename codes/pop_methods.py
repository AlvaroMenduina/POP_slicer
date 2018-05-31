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

    XX, YY Grids are stored as arrays of [N, M, 2]
    """
    def __init__(self, ref_grid):
        self.ref_grid = ref_grid

    def find_nearest(self, XY_ref, XY):
        xx, yy = XY_ref[0,:,:], XY_ref[1,:,:]
        r2 = (xx - XY[0])**2 + (yy - XY[1])**2
        idx = np.unravel_index(np.argmin(r2, axis=None), r2.shape)
        print(idx)
        return idx, (xx[idx], yy[idx])

    def interpolate(self, i_left, j_down, ij_ref, i_right, j_up, data):
        xy, ff = data[0], data[1]
        x_ld, y_ld = xy[0, i_left, j_down], xy[1, i_left, j_down]
        x_lu, y_lu = xy[0, i_left, j_up], xy[1, i_left, j_up]
        x_ru, y_ru = xy[0, i_right, j_up], xy[1, i_right, j_up]
        x_rd, y_rd = xy[0, i_right, j_down], xy[1, i_right, j_down]

        i_ref, j_ref = ij_ref[0], ij_ref[1]
        x_ref, y_ref = self.ref_grid[0, i_ref, j_ref], self.ref_grid[1, i_ref, j_ref]
        deltas = 1. / ((x_ru - x_lu)*(y_ru - y_rd))
        x_vec = np.array([x_ru - x_ref, x_ref - x_lu])
        y_vec = np.array([y_ru - y_ref, y_ref - y_rd])
        F = np.array([[ff[i_left, j_down], ff[i_left, j_up]],
                      [ff[i_right, j_down], ff[i_right, j_up]]])
        f_ref = deltas * np.dot(x_vec, np.dot(F, y_vec))
        return f_ref

    def resample_grid(self, xy_grid, f_values):
        new_grid = self.ref_grid.copy()
        new_values = np.zeros_like(xy_grid)
        N, M = xy_grid.shape[0], xy_grid.shape[1]

        for i in range(N):
            for j in range(M):
                # print('\nIndex')
                # print(i, j)
                xy = xy_grid[:,i,j]
                x, y = xy[0], xy[1]
                # print('Position')
                # print(xy)
                ij_ref, xy_ref = self.find_nearest(self.ref_grid, xy)
                i_ref, j_ref = ij_ref[0], ij_ref[1]
                x_ref, y_ref = xy_ref[0], xy_ref[1]
                if (x_ref > x) and (y_ref > y): # forward & up
                    f_new = self.interpolate(i_left=i, j_down=j, ij_ref=ij_ref, i_right=i+1, j_up=j+1, data=[xy_grid, f_values])
                    print(f_new)
                    # new_grid[i_ref] = self.ref_grid[i_ref]
                    # new_values[i_ref] = f_new


N = 128
x_ref = np.linspace(-5, 5, N)
xx_ref, yy_ref = np.meshgrid(x_ref, x_ref)
ff = xx_ref**2 - yy_ref**2
ref_grid = np.array([xx_ref, yy_ref])

x = np.linspace(-4, 4, N)
xx, yy = np.meshgrid(x, x)
grid = np.array([xx, yy])
gg = xx + yy


resample = ResampleGrid2D(ref_grid)
x_new, g_new = resample.resample_grid(grid, gg)

# resample = ResampleGrid1D(x_ref)
# x_new, g_new = resample.resample_grid(x, g)
#
# plt.figure()
# plt.plot(x_ref, f, label='base')
# plt.plot(x, g, label='add')
# plt.plot(x_new, g_new, label='resampled')
# plt.legend()
# plt.show()
