import numpy as np
from numpy.fft import fft2, fftshift
from scipy.optimize import fsolve
import matplotlib.pyplot as plt
import methods as mt
import zern_core as zern
from time import time as timer


class ImageSlicer(object):

    def __init__(self, Nx=5, Ny=30, N_spax=100):
        """
        Object which models the effects of the HARMONI Image Slicer
        Parameters:
            Nx: Number of field points per Spaxel along the X-axis
            Ny: Number of field points per Spaxel along the Y-axis
            N_spax: Number of Spaxels along a single Slice
        """
        # Physical Parameters
        # This is for the Coarsest 60mas scale NA = 0.00549
        # The value for 4 mas should be NA = 0.00147
        self.NA = 0.00549                                           # Numerical aperture [rad] telecentric beam
        self.N_slices = 38                                          # Total number of slices in the stack
        self.width = 1.                                             # Width of each slice [mm]
        self.semi_length = 51./2.                                   # Half the lenth of the slice [mm]
        self.max_theta = np.deg2rad(14)                             # Max incidence angle on slicer stack [rad]
        self.delta_theta = (2*self.max_theta)/(self.N_slices-1)     # Angle between adjacent slices [rad]

        # Sampling parameters
        self.Nx, self.Ny = Nx, Ny                               # Amount of points per Spaxel
        self.N_spax = N_spax                                    # Number of Spaxels along 1 Slice

        # Timing and Housekeeping
        self.N_masks = self.N_slices * self.N_spax * self.Nx * self.Ny
        self.t_start = timer()

        print('\n====================================================')
        print('      HARMONI Image Slicer - Parameters')
        print('____________________________________________________')
        print('Numerical Aperture NA = %.4f' %self.NA)
        print('Number of Slices: %d' %self.N_slices)
        print('Slice width w = %.1f [mm]' %self.width)
        print('Slice lenght l = %.1f [mm]' %(2*self.semi_length))
        print('Max incidence angle = +- %.1f [deg]' %(np.rad2deg(self.max_theta)))
        print('====================================================')
        print('\n      Spaxel Sampling - Parameters')
        print('____________________________________________________')
        print('Number of SPAXELS N_spax = %d' %self.N_spax)
        print('SPAXEL subsampling = (%d, %d) field points' %(self.Nx, self.Ny))
        print('====================================================')
        print('\nTotal number of Aperture Masks N = %d' %self.N_masks)


        # FIXME deprecated right?!
        self.light_losses = np.zeros((Ny, Nx, self.N_slices//2))
        self.size_footprint = np.zeros((Ny, Nx, self.N_slices//2))

        # FIXME: remember the Discrepancy of sometimes using N_slices and sometimes N_slices//2

    @property
    def colors(self):
        """ Create a list of colors to plot the footprints for each slice """
        import pylab
        colors = []
        cm = pylab.get_cmap('gist_rainbow')
        for i in range(self.N_slices):
            colors.append(cm(1. * i / self.N_slices))
        return colors

    def compute_footprint(self, x_focal, theta_slice, h_defocus):
        """ Compute the intersection between a given slice and the beam """
        h = np.abs(x_focal * np.tan(theta_slice) + h_defocus)
        delta = h * np.tan(self.NA)         # Size along the local Y axis
        # Sizes along the local X axis
        delta_minus = h * np.sin(self.NA) / np.cos(self.NA - theta_slice)
        delta_plus = h * np.sin(self.NA) / np.cos(self.NA + theta_slice)
        return delta, delta_minus, delta_plus

    def elliptic_footprint(self, x_slice, delta, delta_minus, delta_plus):
        """
        Takes the sizes of the footprint (the Deltas) and translates them into
        more familiar ellipse parameters
        """
        a = (delta_minus + delta_plus) / 2.     # Semi-major axis
        p = delta                               # Semilatus rectum
        b = np.sqrt(p * a)                      # Semi-minor axis
        focal_length = a - delta_minus          # Distance from ellipse center to focus
        x_center = x_slice + focal_length       # Center of the ellipse on the slicer plane
        mean_radius = np.sqrt(a * b)            # Mean radius of ellipse (same area as circle)
        eccentricity = focal_length/a
        return x_center, a, b, eccentricity, mean_radius

    def shadowing_estimate(self, max_radius):
        """ Computes a first order estimate of the % of light loss induced by shadowing """
        f = max_radius / np.tan(self.NA)
        delta_h = self.semi_length * np.tan(self.delta_theta)
        fh = f / delta_h
        eps_delta = fh / (1 + fh)
        delta0 = max_radius / 2.
        delta_area = 2*np.sqrt(max_radius**2 - delta0**2)*delta0*(1-eps_delta)
        area0 = np.pi*max_radius**2
        shadowed = delta_area/area0
        print('Shadowing contribution ~= %.1f per cent' %(100*shadowed))
        return shadowed

    def compute_light_loss_footprint(self, x_center, y_center, radius, y_max, y_min):
        """
        For a given spaxel, computes whether the footprint (assumed to be circular) falls
        outside the boundary (i.e. onto an adjacent slice) and computes the fraction of
        footprint area that is lost
        :param y_max: max y for current slice (i.e. y_slice + width)
        :param y_min: min y for current slice
        """
        if radius == 0.:        # Perfectly focused beam
            return 0.
        total_area = np.pi * radius**2
        upper_loss, lower_loss = 0., 0.

        if (y_center + radius) > y_max: # Check whether the footprint falls outside the current slice boundaries
            # Compute distance between the two intersections delta_x = X2 - X1
            delta_x = 2*np.sqrt(radius**2 - (y_max - y_center)**2)
            if np.abs(delta_x - 2*radius) < 1e-6:  # intersection is essentially diametral
                alfa, beta = np.pi, 0.      # alfa is angle of the circular sector, # beta is angle of base triangle
            else:
                alfa = np.arccos(1. - delta_x**2 / (2*radius**2))
                beta = 0.5*(np.pi - alfa)
            area_sector = 0.5 * radius**2 * alfa
            area_triangle = 0.5 * delta_x * radius * np.sin(beta)
            upper_loss = (area_sector - area_triangle) / total_area

        if (y_center - radius) < y_min:
            delta_x = 2*np.sqrt(radius**2 - (y_min - y_center)**2)
            if np.abs(delta_x - 2*radius) < 1e-6:
                alfa, beta = np.pi, 0.
            else:
                alfa = np.arccos(1. - delta_x**2 / (2*radius**2))
                beta = 0.5*(np.pi - alfa)
            area_sector = 0.5 * radius**2 * alfa
            area_triangle = 0.5 * delta_x * radius * np.sin(beta)
            lower_loss = (area_sector - area_triangle) / total_area

        total_loss = upper_loss + lower_loss
        return total_loss

    def compute_key_stuff(self, defocus):
        """
        Coordinates contain [N_slice, N_spaxel, Ny, Nx, (x, y, R)]
        So coordinates_slice[1, 0, :, 0, 1] represents the Y-coordinates for the
        2nd slice, 1st spaxel of that slice at a given x0
        """
        self.coordinates_slice = np.zeros((self.N_slices, self.N_spax, self.Ny, self.Nx, 3))

        # X coordinates along a Slice accounting for N_spax * N_x per spax
        x_focal_whole = np.linspace(-self.semi_length, self.semi_length, self.Nx*self.N_spax)

        # For each SLICE
        for i_slice in range(self.N_slices):
            theta_slice = -self.max_theta + i_slice * self.delta_theta  # Angle of current slice
            # print('Slice #%d, theta=%.2f deg' % (i_slice, np.rad2deg(theta_slice)))
            y_min, y_max = i_slice * self.width, (i_slice + 1) * self.width
            y_focal = np.linspace(y_min, y_max, self.Ny)

            # For each SPAXEL
            for i_spax in range(self.N_spax):
                x_focal = x_focal_whole[i_spax*self.Nx : (i_spax+1)*self.Nx]
                x_slice, y_slice = x_focal / np.cos(theta_slice), y_focal
                xx_focal, yy_focal = np.meshgrid(x_focal, y_focal)
                xx_slice, yy_slice = np.meshgrid(x_slice, y_slice)

                # Compute intersection for each beam
                delta, delta_minus, delta_plus = self.compute_footprint(x_focal=xx_focal, theta_slice=theta_slice, h_defocus=defocus)
                # Compute elliptic footprint parameters
                centers, widths, heights, eccentricity, mean_radius = self.elliptic_footprint(x_slice=xx_slice, delta=delta,
                                                                                              delta_minus=delta_minus,delta_plus=delta_plus)
                # Combine the XY meshes and Radii into a single array
                xyr = np.concatenate((xx_slice[:, :, np.newaxis],
                                      yy_slice[:, :, np.newaxis],
                                      mean_radius[:, :, np.newaxis]), axis=-1)
                # Save that array for each SLICE and each SPAXEL
                self.coordinates_slice[i_slice, i_spax, :, :, :] = xyr

    def compute_cutoffs(self):
        """
        For each point on the image slicer, evaluate whether the Radius of the footprint
        is larger than the width of the current Slice. If so, save the Epsilon [ ] for the
        upper and lower part so that we can construct the proper pupil masks

        Epsilon is adimensional and it refers to the cutoff distance measured in
        radius of the footprint: d_cutoff = eps * r_footprint
        """
        cutoffs = np.ones((self.N_slices, self.N_spax, self.Ny, self.Nx, 2))
        cutoffs[:,:,:,:,1] *= -1.
        # For each SLICE
        for k_slice in range(self.N_slices):
            # For each SPAXEL
            for k_spax in range(self.N_spax):
                upper_y = np.max(self.coordinates_slice[k_slice,k_spax,:,0,1])
                lower_y = np.min(self.coordinates_slice[k_slice,k_spax,:,0,1])

                for j in range(self.Ny):
                    for i in range(self.Nx):
                        y_center = self.coordinates_slice[k_slice,k_spax,j,i,1]
                        radius = self.coordinates_slice[k_slice,k_spax,j,i,2]
                        if (y_center + radius) > upper_y:
                            eps_up = (upper_y - y_center) / radius
                            cutoffs[k_slice, k_spax, j, i, 0] = eps_up
                        if (y_center - radius) < lower_y:
                            eps_down = (lower_y - y_center) / radius
                            cutoffs[k_slice, k_spax, j, i, 1] = eps_down
        self.cut_offs = cutoffs

    def create_masks(self, N_pix, eps):
        """
        Computes an aperture mask for each: slice, spaxel, x-position, y-position
        The masks are always centered to simplify the masking of the NCPA map
        Once propagated into Intensities at the detector plane they are shifted
        according to their true centers (X_c, Y_c)
        """
        x = np.linspace(-1, 1, N_pix)
        y = np.linspace(-1, 1, N_pix)
        xx, yy = np.meshgrid(x, y)
        circular_mask = mt.elt_mask(eps, xx, yy)
        shape_mask = (self.N_slices, self.N_spax, self.Ny, self.Nx, N_pix, N_pix)
        self.masks = np.empty(shape_mask, dtype=bool)

        # For each SLICE
        for k_slice in range(self.N_slices):
            # For each SPAXEL
            for k_spax in range(self.N_spax):
                for j in range(self.Ny):
                    for i in range(self.Nx):
                        eps_up = eps * self.cut_offs[k_slice, k_spax, j, i, 0]
                        eps_down = eps * self.cut_offs[k_slice, k_spax, j, i, 1]
                        mask_up = yy <= eps_up
                        mask_down = yy >= eps_down
                        self.masks[k_slice, k_spax, j, i, :, :] = mask_up * mask_down * circular_mask
        self.t_final = timer()
        total = self.t_final - self.t_start
        print('Time spend in Slicer Operations: %.5f [sec]' %total)
        print('Average time per Aperture Mask: %.5f [sec]' %(total/self.N_masks))
        print('====================================================')


class FourierPropagator(object):
    def __init__(self, zern_model, rho_max, eps, N_pix, wave, image_slicer, integrator):
        """
        Object that takes care of propagating each Pupil Function to the Image plane
        It receives the Masks created by the ImageSlicer, creates the NCPA wavefront map
        and for each Slice, X-position and Y-position, performs Fourier propagation.

        Once propagated, the intensities are shifted along Y, to account for the fact that
        all Slicer Masks were centered at (0,0) while in reality they correspond to different positions
        in the Slice Plane (X, Y)
        :param zern_model: Zernike model that computes the NCPA maps
        :param rho_max: basically eps*Radius
        :param eps: ELT pupil aperture ratio
        :param N_pix: Number of pixels used to sample the NCPA maps and Pupil
        :param wave: Wavelength in [nm]
        """
        self.zern_model = zern_model
        self.N_pix = N_pix
        self.rho_zern = 1. * eps * rho_max
        self.wave = wave
        self.eps = eps
        self.slice_width = 1e-3   # meters
        self.PSF_pix = 10         # How many pixels to include in the sum(intensities) at each side of the PSF

        # Slicer
        self.slicer = image_slicer
        self.N_slices, self.N_spax = self.slicer.N_slices, self.slicer.N_spax,
        self.Ny, self.Nx = self.slicer.Ny, self.slicer.Nx

        # Integrator
        self.integrator = integrator

        # Dummy mask
        x = np.linspace(-rho_max, rho_max, self.N_pix)
        xx, yy = np.meshgrid(x, x)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(xx, yy)
        aperture_mask = mt.elt_mask(eps * rho_max, xx, yy)
        self.rho, self.theta = rho[aperture_mask], theta[aperture_mask]
        self.rho /= self.rho_zern
        self.dummy_mask = aperture_mask

    def create_wavefront(self, coef, PV_goal=100):
        """ Compute NCPA in [nm]"""
        # Remove Piston
        coef[0] = 0
        _ncpa = self.zern_model(coef=coef, rho=self.rho, theta=self.theta,
                                normalize_noll=False, mode='Jacobi', print_option='Silent')
        current_pv = np.max(_ncpa[np.nonzero(_ncpa)]) - np.min(_ncpa[np.nonzero(_ncpa)])
        self.new_coef = coef * PV_goal / current_pv
        rescaled_ncpa = self.zern_model(coef=self.new_coef, rho=self.rho, theta=self.theta,
                                normalize_noll=False, mode='Jacobi', print_option='Silent')
        ncpa2D = zern.invert_mask(rescaled_ncpa, self.dummy_mask)
        self.wavefront = ncpa2D

    def set_phase_diversity(self, n=2, m=0, ratio=10.):
        """
        Create an Defocus map to be used in Phase Diversity
        """
        diversity = self.zern_model.Z_nm(n=n, m=m, rho=self.rho, theta=self.theta,
                                         normalize_noll=False, mode='Jacobi')
        pv_diversity = np.max(diversity) - np.min(diversity)
        pv_phase = np.max(self.wavefront) - np.min(self.wavefront)
        phase_diversity = (ratio * pv_phase) * (diversity / pv_diversity)
        # Remember that at this point the phase diversity is a 1D array (a masked 2D)
        self.phase_diversity = zern.invert_mask(phase_diversity, self.dummy_mask)

    def propagate_single_spaxel(self, i_slice, j_spaxel):
        print('\n Propagating a single SPAXEL')
        print('Slice #%d, Spaxel #%d' %(i_slice, j_spaxel))
        print('____________________________________________________')

        norm_pix = 1. / self.N_pix
        wave_factor = 2 * np.pi / self.wave

        # Add N_y, N_x dimensions to Wavefront
        wavefront_matrix = self.wavefront[np.newaxis, np.newaxis, :, :]
        aperture = self.slicer.masks[i_slice, j_spaxel]
        pupil_functions = aperture * np.exp(1j * wave_factor * wavefront_matrix)
        electric_field = norm_pix * fftshift(fft2(pupil_functions), axes=(-2, -1))
        intensities = (np.abs(electric_field))**2

        integrated_intensity_map = self.integrator.integrate_2D(intensities)
        return integrated_intensity_map

    def propagate_to_image(self, low_memory=False):
        norm_pix = 1. / self.N_pix
        wave_factor = 2 * np.pi / self.wave
        # Add N_y, N_x dimensions to Wavefront
        wavefront_matrix = self.wavefront[np.newaxis, np.newaxis, :, :]

        if low_memory == False:
            self.intensities = np.zeros((self.N_slices, self.N_spax, self.Ny, self.Nx, self.N_pix, self.N_pix))
            print('\n     Fourier propagation')
            print('____________________________________________________')
            for i_slice in range(self.N_slices):
                # print('Slice #%d' %i_slice)
                for i_spaxel in range(self.N_spax):
                    aperture = self.slicer.masks[i_slice, i_spaxel]
                    pupil_functions = aperture * np.exp(1j * wave_factor * wavefront_matrix)
                    electric_field = norm_pix * fftshift(fft2(pupil_functions), axes=(-2,-1))
                    self.intensities[i_slice, i_spaxel] = (np.abs(electric_field))**2
            self.average_spaxel_intensity()

        if low_memory == True:
            t_start = timer()
            N_min = self.N_pix // 2 - (self.PSF_pix - 1) // 2
            N_max = self.N_pix // 2 + (self.PSF_pix - 1) // 2 + 1
            self.final_intensities = np.zeros((self.N_slices, self.N_spax, self.N_pix, self.N_pix))
            self.spaxel_map = np.zeros((self.N_slices, self.N_spax))
            print('\n   Fourier propagation (This takes a long time!)')
            print('____________________________________________________')
            print('Low Memory Approach')

            for i_slice in range(self.N_slices):
                # print('Slice #%d' %i_slice)
                for i_spaxel in range(self.N_spax):
                    aperture = self.slicer.masks[i_slice, i_spaxel]
                    pupil_functions = aperture * np.exp(1j * wave_factor * wavefront_matrix)
                    electric_field = norm_pix * fftshift(fft2(pupil_functions), axes=(-2,-1))
                    intensities = (np.abs(electric_field))**2
                    integrated_intensity_map = self.integrator.integrate_2D(intensities)
                    self.final_intensities[i_slice, i_spaxel] = integrated_intensity_map
                    chopped_intensity = integrated_intensity_map[N_min:N_max, N_min:N_max]
                    if (i_slice==0) and (i_spaxel==0):
                        plt.figure()
                        plt.imshow(integrated_intensity_map)
                        plt.colorbar()

                    self.spaxel_map[i_slice, i_spaxel] = np.sum(chopped_intensity)
            t_final = timer()
            speed = t_final - t_start
            print('Time spent in Fourier propagation: %.3f [sec]' %speed)
            print('Average time per SPAXEL: %.3f [sec]' %(speed/(self.N_slices*self.N_spax)))
            print('____________________________________________________')
            print('Min counts = %f' %np.min(self.spaxel_map))
            print('Max counts = %f' %np.max(self.spaxel_map))
            print('====================================================')

    def average_spaxel_intensity(self):
        """
        Compute the average photon count for each spaxel by summing the
        integrated intensities after Fourier propagation
        As the arrays are quite large (N_pix, N_pix), we only sum over a close-up
        of the PSF which spans self.PSF_pix at each side of the peak
        """
        self.spaxel_map = np.zeros((self.N_slices, self.N_spax))
        print('Spaxel Integration')
        pix_min, pix_max = self.N_pix // 2 - self.PSF_pix, self.N_pix // 2 + self.PSF_pix
        for i_slice in range(self.N_slices):
            print('Slice #%d' %i_slice)
            for i_spaxel in range(self.N_spax):
                intensities = self.intensities[i_slice, i_spaxel]
                integrated_intensity_map = self.integrator.integrate_2D(intensities)
                chopped_intensity = integrated_intensity_map[pix_min:pix_max, pix_min:pix_max]
                self.spaxel_map[i_slice, i_spaxel] = np.sum(chopped_intensity)

    def shift_intensities(self, intensity_to_shift):
        """
        Account for the changes in the center positions of the pupil masks
        Otherwise all intensities will be centered at the same position
        """
        # Compute how many pixels (n_shift) does a Slice Width cover
        footprint_radii = self.slicer.coordinates_slice[:, :, -1]
        # Don't forget the Radii come in [mm]
        r = 1e-3*np.mean(footprint_radii)
        n_shift = int(self.slice_width * self.N_pix * self.eps / (2*r))
        print(n_shift)

        shifted_intensities = np.zeros_like(intensity_to_shift)
        nhy, nhx = intensity_to_shift.shape[0], intensity_to_shift.shape[1]
        if np.mod(n_shift, 2) == 0:
            self.shifts_pixels = np.linspace(-n_shift//2, n_shift//2, nhy).astype(int)
        if np.mod(n_shift, 2) != 0:
            self.shifts_pixels = np.linspace(-n_shift//2, n_shift//2 + 1, nhy).astype(int)

        for j in range(nhy):
            for i in range(nhx):
                shifted_intensities[j, i, :, :] = np.roll(intensity_to_shift[j,i,:,:], self.shifts_pixels[j], axis=0)
        return shifted_intensities

    def reference_result(self):
        """
        Create Reference intensity maps assuming NO Slicer Effects
        i.e. using a standard Aperture Mask
        Useful for quantifying the effects of the Slicer
        """
        norm_pix = 1. / self.N_pix
        wave_factor = 2 * np.pi / self.wave
        pupil_function = self.dummy_mask * np.exp(1j * wave_factor * self.wavefront)
        electric_field = norm_pix * fftshift(fft2(pupil_function))
        reference_intensity = (np.abs(electric_field))**2
        nhy, nhx = self.intensities.shape[0], self.intensities.shape[1]
        self.reference_intensities = np.zeros_like(self.intensities)
        for j in range(nhy):
            for i in range(nhx):
                self.reference_intensities[j,i,:,:] = reference_intensity

class Integrator(object):

    def __init__(self):
        """
        Standard Trapezoid rule integration used to combine all the
        Intensity maps within one Spaxel
        """
        pass

    def create_weights_matrix(self, nhx, nhy):
        """
        Creates a Matrix of Weights for integration according to the
        2D Trapezoidal rule
        """
        if nhx == 1:        # For integration only along Y
            W = 2. * np.ones(nhy)
            W[0], W[-1] = 1., 1.
            return W
        if nhx != 1:        # For full 2D integration
            W = 4. * np.ones((nhx, nhy))
            top_edge = self.create_weights_matrix(nhx=1, nhy=nhy)
            side_edge = self.create_weights_matrix(nhx=1, nhy=nhx)
            W[0, :], W[-1, :] = top_edge, top_edge
            W[:, 0], W[:, -1] = side_edge, side_edge
            return W

    def integrate_2D(self, intensity_matrix):
        nhy, nhx = intensity_matrix.shape[0], intensity_matrix.shape[1]
        weights = self.create_weights_matrix(nhy, nhx)
        factor = np.sum(weights)
        final_intensity = 1./factor*np.sum((weights[:, :, np.newaxis, np.newaxis] * intensity_matrix), axis=(0,1))
        return final_intensity

    def integrate_1D(self, intensity_matrix):
        """ Integrates only along hy' the cross section of the Slice """
        # REMEMBER THAT THE MASK MATRIX IS [Ny, Nx, Npix, Npix]
        nhy = intensity_matrix.shape[0]
        weights = self.create_weights_matrix(nhx=1, nhy=nhy)
        factor = np.sum(weights)
        final_intensity = 1./factor*np.sum((weights[:, np.newaxis,  np.newaxis, np.newaxis] * intensity_matrix), axis=0)
        return final_intensity

class Images(object):
    def __init__(self, xx, yy, pupil_mask, N_images=3):
        """ nhx means #points along the X axis of a slice
            nhy means #points along the Y axis of a slice
        """
        self.pupil_mask = pupil_mask
        self.N_pix = xx.shape[0]
        self.N_imag = N_images
        rho, theta = np.sqrt(xx ** 2 + yy ** 2), np.arctan2(xx, yy)
        self.rho_m, self.theta_m = rho[self.pupil_mask], theta[self.pupil_mask]

    def set_slicer_and_propagator(self, nhx, nhy, zern_model, rho_max, eps, wave):
        self.eps = eps
        self.slicer = ImageSlicer(Nx=nhx, Ny=nhy)
        self.propagator = FourierPropagator(zern_model, rho_max, eps, self.N_pix, wave)
        self.trapezoid = Integrator()
        # FIXME
        self.k_slice = 10
        self.i_x = -1
        # x = self.slicer.coordinates_slice[self.k_slice, 0, self.i_x, 0]

    def update_masks_and_propagate(self, defocus):

        self.slicer.compute_key_stuff(defocus)
        self.slicer.compute_cutoffs()
        self.slicer.create_masks(self.N_pix, rho_max=1, eps=self.eps)
        self.nominal_masks = self.slicer.masks[self.k_slice, :, :, :, :]
        self.propagator.set_slicer_masks(self.nominal_masks)
        self.propagator.propagate_to_image()
        intensities = self.trapezoid.integrate_1D(self.propagator.intensities)
        return intensities[self.i_x]

    def generate_images(self, zern_coef, PV_goal, ratio, method='mm_defocus'):
        """ 2 methods:
        (1) mm_defocus: Ratio here indicates [mm] of displacement of Slicer
        (2) zern_defocus: Ratio here indicates [xTimes the PV of the NCPA]
            in Zernike defocus
        """

        if method == 'mm_defocus':
            self.method_used = method
            # Initialize NCPA wavefront
            self.propagator.create_wavefront(coef=zern_coef, PV_goal=PV_goal)
            self.nominal_image = self.update_masks_and_propagate(defocus=0.)
            self.plus_defocus_image = self.update_masks_and_propagate(defocus=ratio)
            self.minus_defocus_image = self.update_masks_and_propagate(defocus=-ratio)

        if method == 'zern_defocus':
            self.method_used = method
            self.propagator.create_wavefront(coef=zern_coef, PV_goal=PV_goal)
            self.propagator.set_phase_diversity(ratio=ratio)
            nominal_phase = self.propagator.wavefront.copy()
            defocus = self.propagator.phase_diversity.copy()
            # Nominal
            self.nominal_image = self.update_masks_and_propagate(defocus=0.)
            # Plus defocus
            self.propagator.wavefront = nominal_phase + defocus
            self.plus_defocus_image = self.update_masks_and_propagate(defocus=0.)
            # Minus defocus
            self.propagator.wavefront = nominal_phase - defocus
            self.minus_defocus_image = self.update_masks_and_propagate(defocus=0.)

    def plot_images(self, defocus, pix_lim=15):
        pix_min = self.N_pix // 2 - pix_lim
        pix_max = self.N_pix // 2 + pix_lim
        name_list = ['nominal', 'plus ' + str(defocus) + ' defocus', 'minus ' + str(defocus) + ' defocus ']
        image_list = [self.nominal_image, self.plus_defocus_image, self.minus_defocus_image]
        for (image, name) in zip(image_list, name_list):
            image_zoom = image[pix_min:pix_max, pix_min:pix_max]
            plt.figure()
            plt.imshow(image_zoom)
            plt.colorbar()
            plt.title(name + ' (' + self.method_used + ')')



















