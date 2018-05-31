import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import RandomState
import zern_core as zern
import methods as mt
from time import time as timer
from numpy.fft import fft2, fftshift

class FizeauPSF_non_degen(object):

    def __init__(self, zern_model, rho_max, eps, N_pix=1024//2, N_images=1000,
                 max_N_zern=10, wave=10, t_exp=1.0, pix=15, defocus=2.):

        self.zern_model = zern_model
        self.N_pix = N_pix
        self.pix = pix
        self.N_imag = N_images
        self.max_N_zern = max_N_zern
        self.wave = wave
        self.t_exp = t_exp
        self.rho_zern = 1.*eps*rho_max
        self.defocus = defocus

        # Construct the coordinates
        x = np.linspace(-rho_max, rho_max, self.N_pix)
        xx, yy = np.meshgrid(x, x)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(xx, yy)
        aperture_mask = rho <= eps * rho_max
        self.rho, self.theta = rho[aperture_mask], theta[aperture_mask]
        self.rho /= self.rho_zern
        self.extends = [-rho_max, rho_max, -rho_max, rho_max]
        self.aperture_mask = aperture_mask

        self.NN = self.rho.shape[0]
        self.photon_count = []

        self.zern_list = ['Tilt X', 'Tilt Y', 'Astig X', 'Defocus', 'Astig Y',
                          'Trefoil X', 'Coma X', 'Coma Y', 'Trefoil Y',
                          '2nd Trefoil X', '2nd Coma X', 'Spherical Aberr.', '2nd Coma Y', '2nd Trefoil Y',
                          'Z55-', 'Z53-', 'Z51-', 'Z51', 'Z53', 'Z55']

    def extract_subPSF(self, PSF_map):
        pixels = self.pix
        PSF_closeup = np.zeros((pixels, pixels))
        N_min = self.N_pix//2 - (pixels-1)//2
        N_max = self.N_pix//2 + (pixels-1)//2 + 1
        PSF_closeup[:, :] = PSF_map[N_min:N_max, N_min:N_max]
        return PSF_closeup

    def compute_PSF(self, NCPA_map, PSF_normalized=False):

        norm_pix = 1. / (self.N_pix)
        pupil_nominal = mt.complex_function(self.aperture_mask, NCPA_map, self.wave)
        propagated_nominal = norm_pix * fftshift(fft2(pupil_nominal))
        PSF = self.t_exp * (np.abs(propagated_nominal))**2
        self.photon_count.append(PSF.max())

        if PSF_normalized:
            PSF /= np.max(PSF)

        mini_PSF = self.extract_subPSF(PSF)

        return mini_PSF.flatten()

    def compute_defocus_phase(self):
        coef = np.zeros(5)
        coef[-1] = self.defocus
        defocus_map = self.zern_model(coef=coef, rho=self.rho, theta=self.theta,
                                normalize_noll=False, mode='Jacobi', print_option='Silent')
        self.defocus_phase = zern.invert_mask(defocus_map, self.aperture_mask)

    def compute_important_stuff(self, k, coef):
        _ncpa = self.zern_model(coef=coef, rho=self.rho, theta=self.theta,
                                normalize_noll=False, mode='Jacobi', print_option='Silent')
        self.zern_coef[k, :] = coef
        self.zernike_maps[k, :] = _ncpa
        ncpa2D = zern.invert_mask(_ncpa, self.aperture_mask)
        # Defocused version
        ncpa2D_defoc = ncpa2D + self.defocus_phase
        PSF_nom = self.compute_PSF(ncpa2D)
        PSF_foc = self.compute_PSF(ncpa2D_defoc)
        self.PSF_maps[k, :] = np.concatenate((PSF_nom, PSF_foc))

    def concatenate_two_maps(self, raw_map, print_options={'Title':''}):
        map1 = raw_map[:self.pix**2]
        map2 = raw_map[self.pix**2:]
        map1 = map1.reshape((self.pix, self.pix))
        map2 = map2.reshape((self.pix, self.pix))
        map = np.concatenate((map1, map2), axis=1)

        plt.figure()
        plt.imshow(map)
        plt.colorbar(orientation='horizontal')
        plt.title(print_options['Title'])

    def create_PSF_maps(self, PV=1, low=False, low_Z=10):
        """
        Generate the random Zernike coefficients, create the Zernike maps
        and create the PSF maps to be used as 'training' data
        """
        self.seeds = np.zeros(self.N_imag)      # random seeds for the coefficients
        self.zern_coef = np.zeros((self.N_imag, self.max_N_zern))
        self.zernike_maps = np.zeros((self.N_imag, self.NN))
        # This time we use 2 images per map (1 nominal + 1 defocused)
        self.PSF_maps = np.zeros((self.N_imag, 2*self.pix**2))

        print('Generating %d PSF maps' %self.N_imag)
        start = timer()

        for k in range(self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k] = 1.*PV
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)

        for k in np.arange(self.max_N_zern, 2*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - self.max_N_zern] = 0.25*PV
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)

        for k in np.arange(2*self.max_N_zern, 3*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 2*self.max_N_zern] = 0.5*PV
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)

        for k in np.arange(3*self.max_N_zern, 4*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 3*self.max_N_zern] = 0.1*PV
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)

        # Negative
        for k in np.arange(4*self.max_N_zern, 5*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 4*self.max_N_zern] = -1.*PV
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)

        for k in np.arange(5*self.max_N_zern, 6* self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 5*self.max_N_zern] = -0.25*PV
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)

        for k in np.arange(6 * self.max_N_zern, 7 * self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 6 * self.max_N_zern] = -0.5*PV
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)

        for k in np.arange(7 * self.max_N_zern, 8 * self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 7 * self.max_N_zern] = -0.1*PV
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)

        # Series expansion
        for k in np.arange(8*self.max_N_zern, self.N_imag//4):
            self.seeds[k] = k
            _coef = RandomState(k).uniform(low=-1, high=1, size=self.max_N_zern) * PV
            _coef[0] = 0.
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)

        j = 8*self.max_N_zern
        for k in np.arange(self.N_imag//4, self.N_imag//2):
            self.seeds[k] = j
            _coef = 0.5*RandomState(k).uniform(low=-1, high=1, size=self.max_N_zern) * PV
            _coef[0] = 0.
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)
            j += 1

        j = 8*self.max_N_zern
        for k in np.arange(self.N_imag//2, 3*self.N_imag//4):
            self.seeds[k] = j
            _coef = -0.5*RandomState(k).uniform(low=-1, high=1, size=self.max_N_zern) * PV
            _coef[0] = 0.
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)
            j += 1

        j = 8*self.max_N_zern
        for k in np.arange(3*self.N_imag//4, self.N_imag):
            self.seeds[k] = j
            _coef = -1.*RandomState(k).uniform(low=-1, high=1, size=self.max_N_zern) * PV
            _coef[0] = 0.
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            self.compute_important_stuff(k, _coef)
            j += 1

    def create_test_dataset(self, N_images=1000, normalize=False, PV_goal=2, seed_factor=20):
        """
        Create the dataset used to evaluate the fitness of the train model
        """
        self.zern_coef_test = np.zeros((N_images, self.max_N_zern - 1))
        self.zernike_maps_test = np.zeros((N_images, self.NN))
        self.PSF_maps_test = np.zeros((N_images, 2*self.pix**2))

        for k in range(N_images):
            seed = seed_factor*self.N_imag + k
            _coef = RandomState(seed).uniform(low=-1, high=1, size=self.max_N_zern)
            _coef[0] = 0.       # Get rid of the Piston term
            _phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            if normalize:
                max_p, min_p = np.max(_phase[np.nonzero(_phase)]), np.min(_phase[np.nonzero(_phase)])
                PV = max_p - min_p
                _coef *= PV_goal / PV   # Rescale the coefficients to get -1 to 1 map
                _phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                              normalize_noll=False, mode='Jacobi', print_option='Silent')
            # Discard the Piston term
            self.zern_coef_test[k, :] = _coef[1:]
            self.zernike_maps_test[k, :] = _phase
            phase = zern.invert_mask(_phase, self.aperture_mask)
            phase_foc = phase + self.defocus_phase
            PSF_nom = self.compute_PSF(phase)
            PSF_foc = self.compute_PSF(phase_foc)
            self.PSF_maps_test[k, :] = np.concatenate((PSF_nom, PSF_foc))

    def predict_test(self, ML_model):
        N_imag = self.zern_coef_test.shape[0]
        N_zern = self.zern_coef_test.shape[1]
        print('\nTesting the model on a %d-image dataset' %N_imag)
        predictions = ML_model.predict(self.PSF_maps_test)
        true_values = self.zern_coef_test
        error = np.linalg.norm((predictions - true_values), axis=1)
        print('ML Mean error: %.3f' %np.mean(error))
        print('ML Std error: %.3f' %np.std(error))

        for i in range(N_zern):
            pred, true_val = predictions[:, i], true_values[:, i]
            zern_id = self.zern_list[i]
            plt.figure()
            plt.scatter(true_val, pred, s=4)
            plt.plot(true_val, true_val, 'b')
            plt.xlabel('True value')
            plt.ylabel('Predicted')
            # plt.xlim([-1,1])
            # plt.ylim([-1, 1])
            plt.title(zern_id)
        return predictions, true_values, error

class SmartPSF(object):
    # Updated version of FizeauPSF
    def __init__(self, zern_model, rho_max, eps, N_pix,
                 PSF_pix, N_images, wave, t_exp):

        self.rho_max, self.eps = rho_max, eps
        self.N_pix, self.PSF_pix = N_pix, PSF_pix
        self.N_images = N_images
        # Wave should be in [nm]
        self.wave, self.t_exp = wave, t_exp

        # Coordinates and Mask
        x = np.linspace(-rho_max, rho_max, N_pix)
        xx, yy = np.meshgrid(x, x)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(xx, yy)
        pupil_mask = mt.elt_mask(eps, xx, yy)
        self.rho, self.theta = rho[pupil_mask], theta[pupil_mask]
        # Normalize by the maximum radius so that you have Z = 1 at the border
        self.rho_zern = rho_max * eps
        self.rho /= self.rho_zern
        self.pupil_mask = pupil_mask

        # Zernike stuff
        self.zern_model, self.H_matrix = zern_model, zern_model.model_matrix
        self.max_N_zern = self.H_matrix.shape[1]
        self.NN = self.rho.shape[0]     # Amount of points within the mask (flattened)
        self.zern_list = mt.read_zernike_list('zernikes.txt')

        # Check you have enough Images
        if (self.N_images < 32 * self.max_N_zern + 100):
            min_images = 32 * self.max_N_zern + 100
            raise Exception('Not enough Images. Try raising N_images to at least: %d' %min_images)


        print('\n====================================================')
        print('                      SMART PSF                      ')
        print('____________________________________________________')
        print('Number of Zernikes: %d' %self.max_N_zern)

    def extract_subPSF(self, PSF_array):
        """
        Receives a whole (N_pix, N_pix) array of the PSF and crops it to fit
        in a smaller (self.PSF_pix, self.PSF_pix) window
        """
        PSF_crop = np.zeros((self.PSF_pix, self.PSF_pix))
        N_min = self.N_pix//2 - (self.PSF_pix-1)//2
        N_max = self.N_pix//2 + (self.PSF_pix-1)//2 + 1
        PSF_crop[:, :] = PSF_array[N_min:N_max, N_min:N_max]
        return PSF_crop

    def compute_PSF(self, NCPA_map, PSF_normalized=False):
        """
        Receives an NCPA phase map, constructs the Complex Pupil Function
        and computes the PSF by Fourier propagation

        Afterwards, it extracts a smaller crop of the PSF centered around the peak
        :param NCPA_map: [nm]
        :param PSF_normalized: whether to normalize the PSF to MAX = 1
        """
        norm_pix = 1. / (self.N_pix)
        pupil_nominal = mt.complex_function(self.pupil_mask, NCPA_map, self.wave)
        propagated_nominal = norm_pix * fftshift(fft2(pupil_nominal))
        PSF = self.t_exp * (np.abs(propagated_nominal))**2
        # self.photon_count.append(PSF.max())

        if PSF_normalized:
            PSF /= np.max(PSF)

        mini_PSF = self.extract_subPSF(PSF)

        return mini_PSF.flatten()

    def compute_important_stuff(self, k, coef):
        """
        Takes care of saving the Zernike coefficients and maps
        as well as computing the PSF images for the training
        """
        _ncpa = np.dot(self.H_matrix, coef)
        self.zern_coef[k, :] = coef
        self.zernike_maps[k, :] = _ncpa
        ncpa2D = zern.invert_mask(_ncpa, self.pupil_mask)
        self.PSF_maps[k, :] = self.compute_PSF(ncpa2D, False)

    def create_PSF_maps(self, PV, low=False, low_Z=10):
        """
        Generate the random Zernike coefficients, create the Zernike maps
        and create the PSF maps to be used as 'training' data
        """
        self.seeds = np.zeros(self.N_images)      # random seeds for the coefficients
        self.zern_coef = np.zeros((self.N_images, self.max_N_zern))
        self.zernike_maps = np.zeros((self.N_images, self.NN))
        self.PSF_maps = np.zeros((self.N_images, self.PSF_pix**2))

        print('Generating TRAINING SET of %d PSF maps' %self.N_images)
        start = timer()

        # SINGLE ZERNIKE POLYNOMIALS
        for k in range(self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k] = 1.*PV
            self.compute_important_stuff(k, _coef)

        # RESCALED TO DIFFERENT PV
        for k in np.arange(self.max_N_zern, 2*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - self.max_N_zern] = 0.25*PV
            self.compute_important_stuff(k, _coef)

        for k in np.arange(2*self.max_N_zern, 3*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 2*self.max_N_zern] = 0.5*PV
            self.compute_important_stuff(k, _coef)

        for k in np.arange(3*self.max_N_zern, 4*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 3*self.max_N_zern] = 0.1*PV
            self.compute_important_stuff(k, _coef)

        # NEGATIVE & RESCALED TO DIFFERENT PV
        for k in np.arange(4*self.max_N_zern, 5*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 4*self.max_N_zern] = 1.25*PV
            # _coef[k - 4*self.max_N_zern] = -1.*PV
            self.compute_important_stuff(k, _coef)

        for k in np.arange(5*self.max_N_zern, 6* self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 5*self.max_N_zern] = 1.5*PV
            # _coef[k - 5*self.max_N_zern] = -0.25*PV
            self.compute_important_stuff(k, _coef)

        for k in np.arange(6 * self.max_N_zern, 7 * self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 6 * self.max_N_zern] = 2.*PV
            # _coef[k - 6 * self.max_N_zern] = -0.5*PV
            self.compute_important_stuff(k, _coef)

        for k in np.arange(7 * self.max_N_zern, 8 * self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 7 * self.max_N_zern] = 2.25*PV
            # _coef[k - 7 * self.max_N_zern] = -0.1*PV
            self.compute_important_stuff(k, _coef)

        # SERIES EXPANSION OF RANDOM ZERNIKES
        for k in np.arange(8*self.max_N_zern, self.N_images//4):
            self.seeds[k] = k
            _coef = RandomState(k).uniform(low=0, high=1, size=self.max_N_zern)
            # _coef = RandomState(k).uniform(low=-1, high=1, size=self.max_N_zern)
            _coef *= (PV)
            _coef[0] = 0.
            self.compute_important_stuff(k, _coef)

        j = 8*self.max_N_zern
        for k in np.arange(self.N_images//4, self.N_images//2):
            self.seeds[k] = j
            _coef = 0.5*RandomState(k).uniform(low=0, high=1, size=self.max_N_zern)
            # _coef = 0.5*RandomState(k).uniform(low=-1, high=1, size=self.max_N_zern)
            _coef *= (PV )
            _coef[0] = 0.
            self.compute_important_stuff(k, _coef)
            j += 1

        j = 8*self.max_N_zern
        for k in np.arange(self.N_images//2, 3*self.N_images//4):
            self.seeds[k] = j
            # _coef = -0.5*RandomState(k).uniform(low=-1, high=1, size=self.max_N_zern)
            _coef = 0.25*RandomState(k).uniform(low=0, high=1, size=self.max_N_zern)
            _coef *= (PV)
            _coef[0] = 0.
            self.compute_important_stuff(k, _coef)
            j += 1

        j = 8*self.max_N_zern
        for k in np.arange(3*self.N_images//4, self.N_images):
            self.seeds[k] = j
            # _coef = -1.*RandomState(k).uniform(low=-1, high=1, size=self.max_N_zern)
            _coef = 1.25*RandomState(k).uniform(low=0, high=1, size=self.max_N_zern)
            _coef *= (PV )
            _coef[0] = 0.
            self.compute_important_stuff(k, _coef)
            j += 1

        speed = timer() - start
        print('Ellapsed time: %.1f [sec]' %speed)
        print('Average t per image: %f [sec]' %(speed/self.N_images))
        print('____________________________________________________')
        print('PV goal for TRAINING SET: %.3f [um]' %(2*PV))
        PV_ratio, RMS_ratio = self.compute_PV_RMS_ratios(maps=self.zernike_maps)
        print('\nMean PV for TRAINING set %.3f [um]' %np.mean(PV_ratio))
        print('Mean RMS for TRAINING set %.3f [um]' % np.mean(RMS_ratio))
        plt.figure()
        plt.scatter(np.arange(self.N_images), PV_ratio, s=3, color='Black')
        plt.xlabel('Image #')
        plt.ylabel('PV [waves]')
        plt.title('PV of the TRAINING PSF images')
        print('====================================================\n')

    def create_test_dataset(self, N_images, PV_goal, seed_factor=20):
        """
        Create the dataset used to evaluate the fitness of the train model
        """
        self.zern_coef_test = np.zeros((N_images, self.max_N_zern - 1))
        self.zernike_maps_test = np.zeros((N_images, self.NN))
        self.PSF_maps_test = np.zeros((N_images, self.PSF_pix**2))

        print('Generating TEST SET of %d PSF maps' %N_images)
        print('PV of TEST SET: %.3f [nm]' %PV_goal)

        for k in range(N_images):
            seed = seed_factor*self.N_images + k
            _coef = RandomState(seed).uniform(low=0, high=1, size=self.max_N_zern)
            _coef[0] = 0.       # Get rid of the Piston term
            # Normalize by the N_zern to keep the PV at bay
            _coef *= (2*PV_goal / self.max_N_zern)
            _ncpa = np.dot(self.H_matrix, _coef)

            # Discard the Piston term so that the ML does not guess for it
            self.zern_coef_test[k, :] = _coef[1:]
            self.zernike_maps_test[k, :] = _ncpa
            phase = zern.invert_mask(_ncpa, self.pupil_mask)
            self.PSF_maps_test[k, :] = self.compute_PSF(phase, False)


        PV_ratio, RMS_ratio = self.compute_PV_RMS_ratios(maps=self.zernike_maps_test)
        print('\nMean PV for TEST set %.3f [nm]' %np.mean(PV_ratio))
        print('Mean RMS for TEST set %.3f [nm]' % np.mean(RMS_ratio))
        plt.figure()
        plt.scatter(np.arange(N_images), PV_ratio, s=3)
        plt.xlabel('Image #')
        plt.ylabel('PV [waves]')
        plt.title('PV of the testing PSF images')

    def compute_PV_RMS_ratios(self, maps):
        N = maps.shape[0]
        PV_ratios = np.zeros(N)
        RMS_ratios = np.zeros(N)
        for k in range(N):
            phase = maps[k, :]
            PV_ratios[k] = mt.compute_PV_2maps(phase, np.zeros_like(phase))
            RMS_ratios[k] = mt.compute_rms(phase)
        return PV_ratios, RMS_ratios

    def predict_test(self, ML_model):
        N_imag = self.zern_coef_test.shape[0]
        N_zern = self.zern_coef_test.shape[1]
        print('\nTesting the model on a %d-image dataset' %N_imag)
        predictions = ML_model.predict(self.PSF_maps_test)
        true_values = self.zern_coef_test
        error = np.linalg.norm((predictions - true_values), axis=1)
        print('ML Mean error: %.3f' %np.mean(error))
        print('ML Std error: %.3f' %np.std(error))

        for i in range(N_zern):
            pred, true_val = predictions[:, i], true_values[:, i]
            zern_id = self.zern_list[i]
            plt.figure()
            plt.scatter(true_val, pred, s=4)
            plt.plot(true_val, true_val, color='Black', linestyle='--')
            plt.xlabel('True value [um]')
            plt.ylabel('Predicted [um]')
            # plt.xlim([0,1])
            # plt.ylim([0, 1])
            plt.title(zern_id)
        return predictions, true_values, error

class FizeauPSF(object):

    def __init__(self, zern_model, rho_max, eps, N_pix=1024//2, N_images=1000,
                 max_N_zern=10, wave=10, t_exp=1.0, pix=15):

        self.zern_model = zern_model
        self.N_pix = N_pix
        self.pix = pix
        self.N_imag = N_images
        self.max_N_zern = max_N_zern
        self.wave = wave
        self.t_exp = t_exp
        self.rho_zern = 1.*eps*rho_max

        # Construct the coordinates
        x = np.linspace(-rho_max, rho_max, self.N_pix)
        xx, yy = np.meshgrid(x, x)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(xx, yy)
        aperture_mask = rho <= eps * rho_max
        self.rho, self.theta = rho[aperture_mask], theta[aperture_mask]
        self.rho /= self.rho_zern
        self.extends = [-rho_max, rho_max, -rho_max, rho_max]
        self.aperture_mask = aperture_mask

        self.NN = self.rho.shape[0]
        self.photon_count = []

        self.zern_list = ['Tilt X', 'Tilt Y', 'Astig X', 'Defocus', 'Astig Y',
                          'Trefoil X', 'Coma X', 'Coma Y', 'Trefoil Y',
                          '2nd Trefoil X', '2nd Coma X', 'Spherical Aberr.', '2nd Coma Y', '2nd Trefoil Y',
                          'Z55-', 'Z53-', 'Z51-', 'Z51', 'Z53', 'Z55']

    def extract_subPSF(self, PSF_map):
        pixels = self.pix
        PSF_closeup = np.zeros((pixels, pixels))
        N_min = self.N_pix//2 - (pixels-1)//2
        N_max = self.N_pix//2 + (pixels-1)//2 + 1
        PSF_closeup[:, :] = PSF_map[N_min:N_max, N_min:N_max]
        return PSF_closeup

    def compute_PSF(self, NCPA_map, PSF_normalized=False):

        norm_pix = 1. / (self.N_pix)
        pupil_nominal = mt.complex_function(self.aperture_mask, NCPA_map, self.wave)
        propagated_nominal = norm_pix * fftshift(fft2(pupil_nominal))
        PSF = self.t_exp * (np.abs(propagated_nominal))**2
        self.photon_count.append(PSF.max())

        if PSF_normalized:
            PSF /= np.max(PSF)

        mini_PSF = self.extract_subPSF(PSF)

        return mini_PSF.flatten()

    def flat_field_effect(self, maps, flat=0.05):
        im_size = (self.pix, self.pix)
        N = maps.shape[0]
        a0 = 1.0 - flat
        b0 = 1.0 + flat
        self.flat_field_map = (RandomState(5678).uniform(a0, b0, size=im_size)).flatten()
        for k in range(N):
            maps[k, :] *= self.flat_field_map

    def photon_noise(self, maps, apply_noise):
        im_size = self.pix**2
        N = maps.shape[0]
        for k in range(N):
            if apply_noise:
                maps[k, :] = (RandomState(k).poisson(lam=maps[k, :], size=im_size))
            maps[k, :] /= self.t_exp

    def compute_PV_RMS_ratios(self, maps):
        N = maps.shape[0]
        PV_ratios = np.zeros(N)
        RMS_ratios = np.zeros(N)
        for k in range(N):
            phase = maps[k, :]
            PV_ratios[k] = mt.compute_PV_2maps(phase, np.zeros_like(phase)) / self.wave
            RMS_ratios[k] = mt.compute_rms(phase) / self.wave
        return PV_ratios, RMS_ratios

    def create_PSF_maps(self, PV_normalized=False, low=False, low_Z=10):
        """
        Generate the random Zernike coefficients, create the Zernike maps
        and create the PSF maps to be used as 'training' data
        """
        self.seeds = np.zeros(self.N_imag)      # random seeds for the coefficients
        self.zern_coef = np.zeros((self.N_imag, self.max_N_zern))
        self.zernike_maps = np.zeros((self.N_imag, self.NN))
        self.PSF_maps = np.zeros((self.N_imag, self.pix**2))

        print('Generating %d PSF maps' %self.N_imag)
        start = timer()

        for k in range(self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k] = 1.
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            _ncpa = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = _ncpa
            ncpa2D = zern.invert_mask(_ncpa, self.aperture_mask)
            self.PSF_maps[k, :] = self.compute_PSF(ncpa2D)

        for k in np.arange(self.max_N_zern, 2*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - self.max_N_zern] = 0.25
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            # _coef[k - self.max_N_zern - 4] = 0.5
            _ncpa = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = _ncpa
            ncpa2D = zern.invert_mask(_ncpa, self.aperture_mask)
            self.PSF_maps[k, :] = self.compute_PSF(ncpa2D)

        for k in np.arange(2*self.max_N_zern, 3*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 2*self.max_N_zern] = 0.5
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            # _coef[k - 2*self.max_N_zern - 5] = 1.
            _ncpa = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = _ncpa
            ncpa2D = zern.invert_mask(_ncpa, self.aperture_mask)
            self.PSF_maps[k, :] = self.compute_PSF(ncpa2D)

        for k in np.arange(3*self.max_N_zern, 4*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 3*self.max_N_zern] = 0.1
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            _ncpa = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = _ncpa
            ncpa2D = zern.invert_mask(_ncpa, self.aperture_mask)
            self.PSF_maps[k, :] = self.compute_PSF(ncpa2D)

        for k in np.arange(4*self.max_N_zern, self.N_imag//4):
            self.seeds[k] = k
            _coef = np.abs(RandomState(k).uniform(low=-1, high=1, size=self.max_N_zern))
            _coef[0] = 0.
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            # _coef[:3] = np.zeros(3)       # Get rid of the Piston and Tilts
            _ncpa = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            if PV_normalized:
                max_p, min_p = np.max(_ncpa[np.nonzero(_ncpa)]), np.min(_ncpa[np.nonzero(_ncpa)])
                PV = max_p - min_p
                _coef *= 2. / PV   # Rescale the coefficients to get -1 to 1 PV
                _ncpa = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                              normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = _ncpa
            ncpa2D = zern.invert_mask(_ncpa, self.aperture_mask)
            self.PSF_maps[k, :] = self.compute_PSF(ncpa2D)

        j = 4 * self.max_N_zern
        for k in np.arange(self.N_imag//4, self.N_imag//2):
            self.seeds[k] = j
            _coef = 0.25*np.abs(RandomState(j).uniform(low=-1, high=1, size=self.max_N_zern))
            j += 1
            _coef[0] = 0.
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            # _coef[:3] = np.zeros(3)       # Get rid of the Piston and Tilts
            _ncpa = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            if PV_normalized:
                max_p, min_p = np.max(_ncpa[np.nonzero(_ncpa)]), np.min(_ncpa[np.nonzero(_ncpa)])
                PV = max_p - min_p
                _coef *= 2. / PV   # Rescale the coefficients to get -1 to 1 PV
                _ncpa = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                              normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = _ncpa
            ncpa2D = zern.invert_mask(_ncpa, self.aperture_mask)
            self.PSF_maps[k, :] = self.compute_PSF(ncpa2D)

        j = 4 * self.max_N_zern
        for k in np.arange(self.N_imag//2, self.N_imag):
            self.seeds[k] = j
            _coef = 0.5*np.abs(RandomState(j).uniform(low=-1, high=1, size=self.max_N_zern))
            j += 1
            _coef[0] = 0.
            if low:
                _coef[:low_Z] =np.zeros(low_Z)
            # _coef[:3] = np.zeros(3)       # Get rid of the Piston and Tilts
            _ncpa = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            if PV_normalized:
                max_p, min_p = np.max(_ncpa[np.nonzero(_ncpa)]), np.min(_ncpa[np.nonzero(_ncpa)])
                PV = max_p - min_p
                _coef *= 2. / PV   # Rescale the coefficients to get -1 to 1 PV
                _ncpa = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                              normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = _ncpa
            ncpa2D = zern.invert_mask(_ncpa, self.aperture_mask)
            self.PSF_maps[k, :] = self.compute_PSF(ncpa2D)
        #
        # # Add the Astig several times because it's stupid and doesn't recognize it...
        # c = np.linspace(0.1, 1., extra)
        # j = 0
        # for k in np.arange(self.N_imag-extra, self.N_imag):
        #     _coef = np.zeros(self.max_N_zern)
        #     _coef[4] = c[j] * (-1) ** j
        #     j += 1
        #     _ncpa = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
        #                   normalize_noll=False, mode='Jacobi', print_option='Silent')
        #     self.zern_coef[k, :] = _coef
        #     self.zernike_maps[k, :] = _ncpa
        #     ncpa2D = zern.invert_mask(_ncpa, self.aperture_mask)
        #     self.PSF_maps[k, :] = self.compute_PSF(ncpa2D)



        print('Ellapsed time: %.1f sec' % (timer() - start))

    def create_test_dataset(self, N_images=1000, normalize=False, PV_goal=2, seed_factor=20):
        """
        Create the dataset used to evaluate the fitness of the train model
        """
        self.zern_coef_test = np.zeros((N_images, self.max_N_zern - 1))
        self.zernike_maps_test = np.zeros((N_images, self.NN))
        self.PSF_maps_test = np.zeros((N_images, self.pix**2))

        for k in range(N_images):
            seed = seed_factor*self.N_imag + k
            _coef = RandomState(seed).uniform(low=0, high=1, size=self.max_N_zern)
            _coef[0] = 0.       # Get rid of the Piston term
            _phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            if normalize:
                max_p, min_p = np.max(_phase[np.nonzero(_phase)]), np.min(_phase[np.nonzero(_phase)])
                PV = max_p - min_p
                _coef *= PV_goal / PV   # Rescale the coefficients to get -1 to 1 map
                _phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                              normalize_noll=False, mode='Jacobi', print_option='Silent')
            # Discard the Piston term
            self.zern_coef_test[k, :] = _coef[1:]
            self.zernike_maps_test[k, :] = _phase
            phase = zern.invert_mask(_phase, self.aperture_mask)
            self.PSF_maps_test[k, :] = self.compute_PSF(phase)
        PV_ratio, RMS_ratio = self.compute_PV_RMS_ratios(maps=self.zernike_maps_test)
        print('\nMean PV for test set %.2f waves' %np.mean(PV_ratio))
        print('Mean RMS for test set %.2f waves' % np.mean(RMS_ratio))
        plt.figure()
        plt.scatter(np.arange(N_images), PV_ratio, s=3)
        plt.xlabel('Image #')
        plt.ylabel('PV [waves]')
        plt.title('PV of the testing PSF images')

    def predict_test(self, ML_model):
        N_imag = self.zern_coef_test.shape[0]
        N_zern = self.zern_coef_test.shape[1]
        print('\nTesting the model on a %d-image dataset' %N_imag)
        predictions = ML_model.predict(self.PSF_maps_test)
        true_values = self.zern_coef_test
        error = np.linalg.norm((predictions - true_values), axis=1)
        print('ML Mean error: %.3f' %np.mean(error))
        print('ML Std error: %.3f' %np.std(error))

        for i in range(N_zern):
            pred, true_val = predictions[:, i], true_values[:, i]
            zern_id = self.zern_list[i]
            plt.figure()
            plt.scatter(true_val, pred, s=4)
            plt.plot(true_val, true_val, 'b')
            plt.xlabel('True value')
            plt.ylabel('Predicted')
            plt.xlim([0,1])
            plt.ylim([0, 1])
            plt.title(zern_id)
        return predictions, true_values, error

    def compare_wavefront(self, true_coef, pred_coef):
        PVs, RMSs, strehl = [], [], []
        N_imag = true_coef.shape[0]
        piston = np.zeros((N_imag, 1))

        true_coef = np.concatenate((piston, true_coef), axis=1)
        pred_coef = np.concatenate((piston, pred_coef), axis=1)

        i = np.random.randint(low=0, high=N_imag)

        for k in range(N_imag):
            true_phase = self.zern_model(coef=true_coef[k], rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            pred_phase = self.zern_model(coef=pred_coef[k], rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')

            # Rescale to waves
            true_phase /= self.wave
            pred_phase /= self.wave
            PVs.append(np.max(np.abs(true_phase - pred_phase)))
            r = mt.compute_rms(phase_map=true_phase - pred_phase)
            RMSs.append(r)
            strehl.append(np.exp(-r**2))

            if k == i:
                true_phase = zern.invert_mask(true_phase, self.aperture_mask)
                pred_phase = zern.invert_mask(pred_phase, self.aperture_mask)

                plt.figure()
                plt.imshow(true_phase, cmap='jet')
                plt.colorbar()
                plt.title('True Phase')

                plt.figure()
                plt.imshow(pred_phase, cmap='jet')
                plt.colorbar()
                plt.title('Guessed Phase')

                plt.figure()
                plt.imshow(pred_phase - true_phase, cmap='jet')
                plt.colorbar()
                plt.title('Residual Phase')

        print('\nMean PV error [waves]: %.3f' %np.mean(PVs))
        print('Std PV error [waves]: %.3f' %np.std(PVs))
        print('\nMean RMS error [waves]: %.3f' %np.mean(RMSs))
        print('Std RMS error [waves]: %.3f' %np.std(RMSs))

        print('\nMean PV error: %.3f nm @ 550 nm' % (550.*np.mean(PVs)))
        print('\nMean RMS error: %.3f nm @ 550 nm' %(550.*np.mean(RMSs)))

        print('\nMean Strehl ratio: %.2f per cent' %(100*np.mean(strehl)))

        return PVs, RMSs, strehl


    def check_R2(self, predictions, true_values, display=False):
        u = (true_values - predictions)**2
        v = (true_values - np.mean(true_values, axis=1)[:, np.newaxis])**2
        R2 = 1 - u/v

        if display:
            for i in range(R2.shape[-1]):
                zern_id = self.zern_list[i]
                R2_cleaned = R2[:, i][np.where(R2[:, i] >= -1)]

                # plt.figure()
                # plt.scatter(np.arange(R2_cleaned.shape[0]), R2_cleaned, s=5)
                # plt.ylim([-1, 1])
                # plt.title('R2 score:' + zern_id)

                plt.figure()
                plt.hist(R2_cleaned, bins=50)
                plt.xlim([-1, 1])
                plt.xlabel('R2 score:' + zern_id)
                plt.ylabel('Frequency')

        return R2

class IterativeLearning(object):

    def __init__(self, trained_models, zern_model, PSF_model, Z, pix, degen=False):
        self.modelZ1 = trained_models[0]
        self.modelZ2 = trained_models[1]
        self.zern_model = zern_model
        self.PSF_model = PSF_model
        self.Z1, self.Z2 = Z[0], Z[1]
        self.pix = pix
        self.degen = degen
        if degen:
            self.double = 2
        else:
            self.double = 1

    def initial_dataset(self, test_set, zern_coef_test):
        self.N_test = test_set.shape[0]
        self.initial_test_set = test_set
        self.initial_zern_coef = zern_coef_test

    def compute_metrics(self, true_coef, pred_coef):
        f = self.PSF_model
        N_imag = true_coef.shape[0]
        metrics = np.zeros((N_imag, 3))
        piston = np.zeros((N_imag, 1))
        true_coef = np.concatenate((piston, true_coef), axis=1)
        pred_coef = np.concatenate((piston, pred_coef), axis=1)
        for k in range(N_imag):
            true_phase = self.zern_model(coef=true_coef[k], rho=f.rho, theta=f.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            pred_phase = self.zern_model(coef=pred_coef[k], rho=f.rho, theta=f.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            # Rescale to waves
            true_phase /= f.wave
            pred_phase /= f.wave
            metrics[k, 0] = np.max(np.abs(true_phase - pred_phase))
            r = mt.compute_rms(phase_map=true_phase - pred_phase)
            metrics[k, 1], metrics[k, 2] = r, np.exp(-r**2)
        return metrics

    def learn(self, max_iter):
        """
        Iterative learning strategy using 2 separate Neural Networks
        The first one N1 takes care of predicting Z1 LOW order Zernike polynomials
        while the second one N2 takes care of predicting the remaining
        (Z2 - Z1) HIGH order Zernike polynomials.
        """

        # Initialize useful stuff
        self.max_iter = max_iter
        piston = np.zeros((self.N_test, 1))
        self.errors_low = []
        self.errors_high = []
        self.PSF_maps = np.zeros((self.N_test, self.double*self.pix**2, 2*max_iter))
        # Metrics: PV, RMS, Strehl [waves]
        self.metrics = np.zeros((self.N_test, 3, 2*max_iter + 1))
        self.pred_low = np.zeros((self.N_test, self.Z1-1, max_iter))
        self.true_low = np.zeros((self.N_test, self.Z1-1, max_iter))
        self.pred_high = np.zeros((self.N_test, self.Z2 - self.Z1, max_iter))
        self.true_high = np.zeros((self.N_test, self.Z2 - self.Z1, max_iter))
        i = np.random.randint(low=0, high=self.N_test)

        current_dataset = self.initial_test_set
        current_zern_coef = self.initial_zern_coef

        if self.degen:
            # self.PSF_model.concatenate_two_maps(self.initial_test_set[i])
            map = self.show_degen_PSF(self.initial_test_set[i], order='LOW', k=-1)

        if not self.degen:
            initial_PSF = self.initial_test_set[i].reshape((self.pix, self.pix))
            plt.figure()
            plt.imshow(initial_PSF, vmin=initial_PSF.min(), vmax=initial_PSF.max())
            plt.colorbar()
            plt.title('Initial PSF')

        metrics = self.compute_metrics(true_coef=current_zern_coef, pred_coef=np.zeros_like(current_zern_coef))
        self.metrics[:,:,0] = metrics

        for k in range(max_iter):
            # Predict on LOW model
            pred_Z1 = self.modelZ1.predict(current_dataset)
            self.pred_low[:, :, k] = pred_Z1
            self.true_low[:, :, k] = current_zern_coef[:, :(self.Z1 - 1)]
            error_low = current_zern_coef[:, :(self.Z1 - 1)] - pred_Z1 #
            self.errors_low.append(np.linalg.norm(error_low, axis=1))

            # Compute remaining NCPA (after removing LOW)
            remaining_high = current_zern_coef[:, (self.Z1 - 1):]
            remaining = np.concatenate((error_low, remaining_high), axis=1)
            self.metrics[:, :, 2*k + 1] = self.compute_metrics(true_coef=remaining, pred_coef=np.zeros_like(remaining))
            remaining = np.concatenate((piston, remaining), axis=1) #

            # Update the PSF maps
            for j in range(self.N_test):
                coef = remaining[j, :]
                f = self.PSF_model
                _phase = self.zern_model(coef=coef, rho=f.rho, theta=f.theta,
                                      normalize_noll=False, mode='Jacobi', print_option='Silent')
                phase = zern.invert_mask(_phase, f.aperture_mask)
                if self.degen:
                    PSF_nom = f.compute_PSF(phase)
                    PSF_foc = f.compute_PSF(phase + f.defocus_phase)
                    self.PSF_maps[j, :, 2*k] = np.concatenate((PSF_nom, PSF_foc))
                if not self.degen:
                    self.PSF_maps[j, :, 2*k] = f.compute_PSF(phase)
            current_dataset = self.PSF_maps[:, :, 2*k]

            if self.degen:
                # self.PSF_model.concatenate_two_maps(current_dataset[i])
                map = self.show_degen_PSF(current_dataset[i], order='LOW', k=k)
            if not self.degen:
                low_PSF = current_dataset[i].reshape((self.pix, self.pix))
                plt.figure()
                plt.imshow(low_PSF, vmin=low_PSF.min(), vmax=low_PSF.max())
                plt.colorbar()
                plt.title('PSF after removing LOW order (iteration=%d)' %(k+1))

            # Predict on the HIGH model
            pred_Z2 = self.modelZ2.predict(current_dataset)
            error_high = current_zern_coef[:, (self.Z1 - 1):] - pred_Z2 #
            self.errors_high.append(np.linalg.norm(error_high, axis=1))

            self.pred_high[:, :, k] = pred_Z2
            self.true_high[:, :, k] = current_zern_coef[:, (self.Z1 - 1):]

            # Compute remaining NCPA (after removing HIGH)
            remaining = np.concatenate((error_low, error_high), axis=1)
            self.metrics[:, :, 2*k + 2] = self.compute_metrics(true_coef=remaining, pred_coef=np.zeros_like(remaining))
            remaining = np.concatenate((piston, remaining), axis=1) #

            # Update the PSF maps
            for j in range(self.N_test):
                coef = remaining[j, :]
                f = self.PSF_model
                _phase = self.zern_model(coef=coef, rho=f.rho, theta=f.theta,
                                      normalize_noll=False, mode='Jacobi', print_option='Silent')
                phase = zern.invert_mask(_phase, f.aperture_mask)
                if self.degen:
                    PSF_nom = f.compute_PSF(phase)
                    PSF_foc = f.compute_PSF(phase + f.defocus_phase)
                    self.PSF_maps[j, :, 2*k+1] = np.concatenate((PSF_nom, PSF_foc))
                if not self.degen:
                    self.PSF_maps[j, :, 2*k+1] = f.compute_PSF(phase)

            current_dataset = self.PSF_maps[:, :, 2*k + 1]
            current_zern_coef = remaining[:, 1:]

            if self.degen:
                # self.PSF_model.concatenate_two_maps(current_dataset[i])
                map = self.show_degen_PSF(current_dataset[i], order='HIGH', k=k)

            if not self.degen:
                high_PSF = current_dataset[i].reshape((self.pix, self.pix))
                plt.figure()
                plt.imshow(high_PSF, vmin=high_PSF.min(), vmax=high_PSF.max())
                plt.colorbar()
                plt.title('PSF after removing HIGH order (iteration=%d)' %(k+1))

        self.final_dataset = current_dataset

    def show_degen_PSF(self, raw_map, order, k):
        map1 = raw_map[:self.pix**2]
        map2 = raw_map[self.pix**2:]
        map1 = map1.reshape((self.pix, self.pix))
        map2 = map2.reshape((self.pix, self.pix))
        map = np.concatenate((map1, map2), axis=1)

        plt.figure()
        plt.imshow(map, vmin=map.min(), vmax=map.max())
        plt.colorbar(orientation='horizontal')
        if k==-1:
            plt.title('Initial PSF')
        else:
            plt.title('PSF after removing ' + order + ' order (iteration=%d)' % (k + 1))
        return map

    def show_results(self):
        k = 1
        lows, highs = [], []
        for (err_low, err_high) in zip(self.errors_low, self.errors_high):
            mean_low = err_low.mean()
            mean_high = err_high.mean()
            lows.append(mean_low)
            highs.append(mean_high)
            print('\nLOW error at iteration %d: %.5f' %(k,mean_low))
            print('HIGH error at iteration %d: %.5f' %(k, mean_high))
            k += 1
        it = np.arange(len(lows))
        plt.figure()
        plt.plot(it, lows, label='LOW')
        plt.plot(it, highs, label='HIGH')
        plt.legend()
        plt.xlabel('Iteration')
        plt.ylabel('Norm NCPA')

        i = np.random.randint(low=0, high=self.N_test)
        fig = plt.figure()
        ims = []
        for k in range(self.max_iter):
            initial_map = self.PSF_maps[i,:,k]
            map1 = initial_map[:self.pix ** 2]
            map2 = initial_map[self.pix ** 2:]
            map1 = map1.reshape((self.pix, self.pix))
            map2 = map2.reshape((self.pix, self.pix))
            map = np.concatenate((map1, map2), axis=1)
            im = plt.imshow(map, animated=True)
            # plt.colorbar(im)
            ims.append([im])

        ani = animation.ArtistAnimation(fig, ims, interval=2000, blit=True,
                                repeat_delay=1000, repeat=True)
        ani.save('dynamic_psf.mp4')

        fig, ax = plt.subplots()
        points = np.arange(self.pix)
        ax.set_ylim([0, np.round(1.1*np.max(self.PSF_maps[i, :, self.max_iter]))])
        initial_map = self.PSF_maps[i, :, 0]
        map1 = initial_map[:self.pix ** 2]
        map1 = map1.reshape((self.pix, self.pix))
        line, = ax.plot(points, map1[self.pix//2, :])

        def animate(k):
            initial_map = self.PSF_maps[i, :, k]
            map1 = initial_map[:self.pix ** 2]
            map1 = map1.reshape((self.pix, self.pix))
            line.set_ydata(map1[self.pix//2, :])
            return line,

        def init():
            line.set_ydata(np.ma.array(points, mask=True))
            return line,

        ani = animation.FuncAnimation(fig, animate, np.arange(1, self.max_iter), init_func=init,
                                      interval=1000, blit=True)
        plt.show()


        # # Select a random map
        # i = np.random.randint(low=0, high=self.N_test)
        # if self.degen:
        #     initial_PSF = self.PSF_model.concatenate_two_maps(self.initial_test_set[i])
        #     final_PSF = self.PSF_model.concatenate_two_maps(self.final_dataset[i])
        # if not self.degen:
        #     initial_PSF = self.initial_test_set[i].reshape((self.pix, self.pix))
        #     final_PSF = self.final_dataset[i].reshape((self.pix, self.pix))
        #
        #     plt.figure()
        #     plt.imshow(initial_PSF)
        #     plt.colorbar()
        #     plt.title('Initial PSF')
        #
        #     plt.figure()
        #     plt.imshow(final_PSF)
        #     plt.colorbar()
        #     plt.title('Final PSF')
        #
        #     plt.figure()
        #     plt.imshow(final_PSF - initial_PSF)
        #     plt.colorbar()
        #     plt.title('Residual PSF')

    def show_hist(self, max_iter=3):
        # Create Error histograms
        for z in range(self.Z1 - 1):
            mins, maxs = [], []
            plt.figure()
            for i in range(max_iter):
                error = self.true_low[:,z,i] - self.pred_low[:,z,i]
                mins.append(error.min())
                maxs.append(error.max())
                plt.hist(error, bins=20, histtype='step', label='%d' %i)
            xmax, xmin = max(maxs), min(mins)
            delta = max((xmax, np.abs(xmin)))
            plt.xlim([-delta, delta])
            plt.legend(title='Iteration')
            plt.xlabel('Error')
            plt.ylabel('Frequency')
            plt.title(self.PSF_model.zern_list[z])

        for z in range(self.Z2 - self.Z1):
            mins, maxs = [], []
            plt.figure()
            for i in range(max_iter):
                error = self.true_high[:,z,i] - self.pred_high[:,z,i]
                mins.append(error.min())
                maxs.append(error.max())
                plt.hist(error, bins=20, histtype='step', label='%d' %i)
            xmax, xmin = max(maxs), min(mins)
            delta = max((xmax, np.abs(xmin)))
            plt.xlim([-delta, delta])
            plt.legend(title='Iteration')
            plt.xlabel('Error')
            plt.ylabel('Frequency')
            plt.title(self.PSF_model.zern_list[self.Z1 - 1 + z])

class FizeauImages(object):
    """
    Object which takes care of generating the Zernike maps and their associated
    Interference maps to use as either Training or Test Dataset for a Machine Learning
    algorithm.
    """

    def __init__(self, zern_model, rho_max, N_pix=1024//2, N_images=1000, max_N_zern=10, wave_fact=10):
        self.zern_model = zern_model
        self.N_pix = N_pix
        self.N_imag = N_images
        self.max_N_zern = max_N_zern
        self.wave_fact = wave_fact

        # Construct the coordinates
        x = np.linspace(-rho_max, rho_max, self.N_pix)
        xx, yy = np.meshgrid(x, x)
        rho = np.sqrt(xx ** 2 + yy ** 2)
        theta = np.arctan2(xx, yy)
        aperture_mask = rho <= rho_max
        self.rho, self.theta = rho[aperture_mask], theta[aperture_mask]
        self.extends = [-rho_max, rho_max, -rho_max, rho_max]
        self.aperture_mask = aperture_mask

        self.NN = self.rho.shape[0]

    def interference_pattern(self, phase):
        """
        Compute the interference pattern for a given wavefront map
        """
        ref_wave = self.wave_fact * np.ones_like(phase)
        wave = self.wave_fact * phase
        interference = 2 * np.ones_like(phase) ** 2 * (1 + np.cos(wave - ref_wave))
        return interference

    def create_maps(self, PV_goal=2):
        """
        Generate the random Zernike coefficients, create the Zernike maps
        and create the intereference maps to be used as 'training' data
        """
        self.seeds = np.zeros(self.N_imag)      # random seeds for the coefficients
        self.zern_coef = np.zeros((self.N_imag, self.max_N_zern))
        self.zernike_maps = np.zeros((self.N_imag, self.NN))
        self.interf_maps = np.zeros((self.N_imag, self.NN))

        print('Generating %d interference maps' %self.N_imag)
        start = timer()

        # Begin by adding individual Zernikes
        for k in range(self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k] = 1.
            phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = phase
            self.interf_maps[k, :] = self.interference_pattern(phase)

        # Add the same maps but rescaled by -2
        for k in np.arange(self.max_N_zern, 2*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - self.max_N_zern] = -2.
            phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = phase
            self.interf_maps[k, :] = self.interference_pattern(phase)

        # Add the same maps but rescaled by 0.5
        for k in np.arange(2*self.max_N_zern, 3*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 2*self.max_N_zern] = 0.5
            phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = phase
            self.interf_maps[k, :] = self.interference_pattern(phase)

        # Add the same maps but rescaled by -0.25
        for k in np.arange(3*self.max_N_zern, 4*self.max_N_zern):
            _coef = np.zeros(self.max_N_zern)
            _coef[k - 3*self.max_N_zern] = -0.25
            phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = phase
            self.interf_maps[k, :] = self.interference_pattern(phase)

        # Fill in the rest with Random maps
        for k in np.arange(4*self.max_N_zern, self.N_imag//2):
            self.seeds[k] = k
            _coef = RandomState(k).normal(size=self.max_N_zern)
            _coef[0] = 0.       # Get rid of the Piston term
            _phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            max_p, min_p = np.max(_phase[np.nonzero(_phase)]), np.min(_phase[np.nonzero(_phase)])
            PV = max_p - min_p
            _coef *= PV_goal / PV   # Rescale the coefficients to get -1 to 1 map
            phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = phase
            self.interf_maps[k, :] = self.interference_pattern(phase)

        # The same maps but rescaled
        j = 4*self.max_N_zern
        for k in np.arange(self.N_imag//2, self.N_imag):
            self.seeds[k] = j
            _coef = -1*RandomState(j).normal(size=self.max_N_zern)
            j += 1
            _coef[0] = 0.       # Get rid of the Piston term
            _phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            max_p, min_p = np.max(_phase[np.nonzero(_phase)]), np.min(_phase[np.nonzero(_phase)])
            PV = max_p - min_p
            _coef *= PV_goal / PV   # Rescale the coefficients to get -1 to 1 map
            phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            self.zern_coef[k, :] = _coef
            self.zernike_maps[k, :] = phase
            self.interf_maps[k, :] = self.interference_pattern(phase)

        print('Ellapsed time: %.1f sec' %(timer() - start))

    def create_test_dataset(self, N_images = 1000, PV_goal=2, seed_factor=7):
        """
        Create the dataset used to evaluate the fitness of the train model
        """
        self.zern_coef_test = np.zeros((N_images, self.max_N_zern - 1))
        self.zernike_maps_test = np.zeros((N_images, self.NN))
        self.interf_maps_test = np.zeros((N_images, self.NN))

        for k in range(N_images):
            seed = seed_factor*self.N_imag + k
            _coef = RandomState(seed).normal(size=self.max_N_zern)
            _coef[0] = 0.       # Get rid of the Piston term
            _phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            max_p, min_p = np.max(_phase[np.nonzero(_phase)]), np.min(_phase[np.nonzero(_phase)])
            PV = max_p - min_p
            _coef *= PV_goal / PV   # Rescale the coefficients to get -1 to 1 map
            phase = self.zern_model(coef=_coef, rho=self.rho, theta=self.theta,
                          normalize_noll=False, mode='Jacobi', print_option='Silent')
            # Discard the Piston term
            self.zern_coef_test[k, :] = _coef[1:]
            self.zernike_maps_test[k, :] = phase
            self.interf_maps_test[k, :] = self.interference_pattern(phase)

    def train(self, ML_model, new_training_set, new_targets):
        ML_model.fit(X=new_training_set, y=new_targets)
        return ML_model

    def predict_test(self, ML_model):
        print('\nTesting the model on a %d-image dataset' %(self.zern_coef_test.shape[0]))
        predictions = ML_model.predict(self.interf_maps_test)
        true_values = self.zern_coef_test
        error = np.linalg.norm((predictions - true_values), axis=1)
        print('ML Mean error in the Zernike coefficients:', np.mean(error))
        print('ML Std error in the Zernike coefficients:', np.std(error))

        # try with totally random values
        rand_values = np.random.uniform(low=-1, high=1, size=predictions.shape)
        rand_error = np.linalg.norm((predictions - rand_values), axis=1)
        # print('Random Mean error', np.mean(rand_error))
        # print('Random Std error', np.std(rand_error))
        # print('\n')

        return predictions, error

    def check_R2(self, predictions, true_values):
        # R2 = np.zeros(predictions.shape[0])
        u = np.sum((true_values - predictions)**2, axis=1)
        v = np.sum((true_values - np.mean(true_values, axis=1)[:, np.newaxis])**2, axis=1)
        R2 = 1 - u/v
        return R2

