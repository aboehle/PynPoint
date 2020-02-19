"""
Functions for calculating detection limits.
"""

import math

from typing import Tuple

import numpy as np

from photutils import aperture_photometry, CircularAperture
from typeguard import typechecked

from pynpoint.util.analysis import student_t, fake_planet, false_alarm
from pynpoint.util.image import polar_to_cartesian, center_subpixel
from pynpoint.util.psf import pca_psf_subtraction
from pynpoint.util.residuals import combine_residuals

@typechecked
def contrast_limit(path_images: str,
                   path_psf: str,
                   noise: np.ndarray,
                   mask: np.ndarray,
                   parang: np.ndarray,
                   psf_scaling: float,
                   extra_rot: float,
                   pca_number: int,
                   threshold: Tuple[str, float],
                   aperture: float,
                   residuals: str,
                   snr_inject: float,
                   num_iter: int,
                   posang_ignore: Tuple[float, float],
                   position: Tuple[float, float]) -> Tuple[float, float, float, float, float, np.ndarray, np.ndarray]:

    """
    Function for calculating the contrast limit at a specified position for a given sigma level or
    false positive fraction, both corrected for small sample statistics.

    Parameters
    ----------
    path_images : str
        System location of the stack of images (3D).
    path_psf : str
        System location of the PSF template for the fake planet (3D). Either a single image or a
        stack of images equal in size to science data.
    noise : numpy.ndarray
        Residuals of the PSF subtraction (3D) without injection of fake planets. Used to measure
        the noise level with a correction for small sample statistics.
    mask : numpy.ndarray
        Mask (2D).
    parang : numpy.ndarray
        Derotation angles (deg).
    psf_scaling : float
        Additional scaling factor of the planet flux (e.g., to correct for a neutral density
        filter). Should have a positive value.
    extra_rot : float
        Additional rotation angle of the images in clockwise direction (deg).
    pca_number : int
        Number of principal components used for the PSF subtraction.
    threshold : tuple(str, float)
        Detection threshold for the contrast curve, either in terms of 'sigma' or the false
        positive fraction (FPF). The value is a tuple, for example provided as ('sigma', 5.) or
        ('fpf', 1e-6). Note that when sigma is fixed, the false positive fraction will change with
        separation. Also, sigma only corresponds to the standard deviation of a normal distribution
        at large separations (i.e., large number of samples).
    aperture : float
        Aperture radius (pix) for the calculation of the false positive fraction.
    residuals : str
        Method used for combining the residuals ('mean', 'median', 'weighted', or 'clipped').
    snr_inject : float
        Signal-to-noise ratio of the injected planet signal that is used to measure the amount
        of self-subtraction.
    position : tuple(float, float)
        The separation (pix) and position angle (deg) of the fake planet.

    Returns
    -------
    float
        Separation (pix).
    float
        Position angle (deg).
    float
        Contrast (mag).
    float
        False positive fraction.
    """

    images = np.load(path_images)
    psf = np.load(path_psf)

    if threshold[0] == 'sigma':
        sigma = threshold[1]

        # Calculate the FPF for a given sigma level
        fpf = student_t(t_input=threshold,
                        radius=position[0],
                        size=aperture,
                        ignore=False)

    elif threshold[0] == 'fpf':
        fpf = threshold[1]

        # Calculate the sigma level for a given FPF
        sigma = student_t(t_input=threshold,
                          radius=position[0],
                          size=aperture,
                          ignore=False)

    else:
        raise ValueError('Threshold type not recognized.')

    # Cartesian coordinates of the fake planet
    yx_fake = polar_to_cartesian(images, position[0], position[1]-extra_rot)

    # Determine the noise level
    flux, t_noise, t_test, _ = false_alarm(image=noise[0, ],
                                   x_pos=yx_fake[1],
                                   y_pos=yx_fake[0],
                                   size=aperture,
                                   posang_ignore=posang_ignore,
                                   ignore=True)

    # Get average in noise apertures from false_alarm output
    avg_of_noiseaps = flux - t_test * t_noise

    # Aperture properties
    im_center = center_subpixel(images)

    # Measure the flux of the star
    ap_phot = CircularAperture((im_center[1], im_center[0]), aperture)
    phot_table = aperture_photometry(psf_scaling*psf[0, ], ap_phot, method='exact')
    star = phot_table['aperture_sum'][0]

    # Initialize iteration arrays
    flux_in_iter = np.zeros(num_iter+1)
    attenuation_iter = np.zeros(num_iter)
    noise_iter = np.zeros(num_iter)
    avg_of_noiseaps_iter = np.zeros(num_iter)
    t_test_iter = np.zeros(num_iter+1)

    # Magnitude of the injected planet
    flux_in_iter[0] = snr_inject*t_noise

    for i in range(num_iter):

        # Inject the fake planet
        mag = -2.5 * math.log10(flux_in_iter[i] / star)

        fake = fake_planet(images=images,
                           psf=psf,
                           parang=parang,
                           position=(position[0], position[1]),
                           magnitude=mag,
                           psf_scaling=psf_scaling)

        # Run the PSF subtraction
        im_res, _  = pca_psf_subtraction(images=fake*mask,
                                         angles=-1.*parang+extra_rot,
                                         pca_number=pca_number)

        # Stack the residuals
        im_res = combine_residuals(method=residuals, res_rot=im_res)

        # Measure the flux of the fake planet
        flux_out, noise_iter[i], t_test_iter[i], _ = false_alarm(image=im_res[0, ],
                                                            x_pos=yx_fake[1],
                                                            y_pos=yx_fake[0],
                                                            size=aperture,
                                                            posang_ignore=posang_ignore,
                                                            ignore=True)

        # Calculate the amount of self-subtraction
        attenuation_iter[i] = flux_out/flux_in_iter[i]

        # Get average in the noise aps, which goes into the student-t test
        avg_of_noiseaps_iter[i] = flux_out - t_test_iter[i] * noise_iter[i]

        if i in [0,1,2]:
            # Make initial guess for the limiting flux from snr_inject planet
            flux_in_iter[i+1] = (sigma*noise_iter[i] + avg_of_noiseaps_iter[i])/attenuation_iter[i]

        #elif i == 1:
            # Make second guess for the limiting flux,
            # assuming same attenuation, noise, and average in noise aps
        #    flux_in_iter[i+1] = (sigma*noise_iter[i] + avg_of_noiseaps_iter[i]) / attenuation_iter[i]

        else:
            # Make a next guess for the 5-sigma flux
            # linearly extrapolating from previous 2 values of
            # attenuation, noise, and average in noise aps
            p_noise = np.polyfit(flux_in_iter[i-1:i+1], noise_iter[i-1:i+1], deg=1)
            p_att = np.polyfit(flux_in_iter[i-1:i+1], attenuation_iter[i-1:i+1], deg=1)
            p_avg = np.polyfit(flux_in_iter[i-1:i+1], avg_of_noiseaps_iter[i-1:i+1], deg=1)

            roots = np.roots([p_att[0],(p_att[1] - sigma*p_noise[0] - p_avg[0]),-(sigma*p_noise[1] + p_avg[1])])

            # check if roots are real
            # if not, then use the method above for the next guess
            if np.isreal(roots).all():
                flux_in_iter[i+1] = np.min(roots[np.where(roots > 0)])
            else:
                flux_in_iter[i+1] = (sigma*noise_iter[i] + avg_of_noiseaps_iter[i]) / attenuation_iter[i]

        #print(f'\tt test snr = {t_test_iter[i]} for contrast of {flux_in_iter[i]/star}')

    # Calculate the detection limit
    contrast = flux_in_iter[-1]/star

    # Do final check of contrast
    mag = -2.5 * math.log10(contrast)

    fake = fake_planet(images=images,
                           psf=psf,
                           parang=parang,
                           position=(position[0], position[1]),
                           magnitude=mag,
                           psf_scaling=psf_scaling)

    # Run the PSF subtraction
    im_res, _  = pca_psf_subtraction(images=fake*mask,
                                         angles=-1.*parang+extra_rot,
                                         pca_number=pca_number)

    # Stack the residuals
    im_res = combine_residuals(method=residuals, res_rot=im_res)

    # Measure the flux of the fake planet
    flux_out, noise_out, t_test_iter[-1], _ = false_alarm(image=im_res[0, ],
                                                     x_pos=yx_fake[1],
                                                     y_pos=yx_fake[0],
                                                     size=aperture,
                                                     posang_ignore=posang_ignore,
                                                     ignore=True)

    #print(f'\tfinal t test snr = {t_test_out} for contrast of {contrast}')

    # The flux_out can be negative, for example if the aperture includes self-subtraction regions
    if contrast > 0.:
        contrast = -2.5*math.log10(contrast)
    else:
        contrast = np.nan

    contrast_iter = -2.5*np.log10(flux_in_iter/star)

    # Separation [pix], position antle [deg], contrast [mag], FPF
    return position[0], position[1], contrast, fpf, t_test_iter[-1], contrast_iter, t_test_iter
