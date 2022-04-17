import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
import os,sys

from astropy.io import fits,ascii
from astropy.convolution import Gaussian2DKernel,convolve_fft
from astropy.stats import sigma_clip


def run(file_cal, file_seg=None, plot_res=False, file_out=None, f_write=True):
    '''
    Parameters
    ----------
    file_cal : str
        cal.fits file of JWST.

    file_seg : str
        segmentation mask for file_cal. The data extension is assumed to be 0.

    plot_res : bool
        Show results of each step.

    Returns
    -------
    fd_cal_ampsub_fsub : 2d-array
        1/f noise subtracted image array.

    '''
    if file_seg == None:
        file_seg = file_cal.replace('.fits','_seg.fits')
        if not os.path.exists(file_seg):
            print('No segmap found. Exiting.')
            os.exit()

    # Open files;
    fd_cal = fits.open(file_cal)['SCI'].data
    dq_cal = fits.open(file_cal)['DQ'].data
    fd_seg = fits.open(file_seg)[0].data

    #
    # 1. Exclude positive pixels originating from sources;
    #
    sigma_1 = 3.
    maxiters_1 = 10
    con = np.where((fd_seg > 0) | (dq_cal>0))
    fd_cal_stat = fd_cal.copy()
    fd_cal_stat[con] = np.nan

    fd_cal_clip = sigma_clip(fd_cal_stat.flatten(), sigma=sigma_1, maxiters=maxiters_1,
                            cenfunc=mean, masked=False, copy=False)

    fd_stats = np.nanpercentile(fd_cal_clip, [0.1,50,99.9])
    fd_max = fd_cal_clip.max()

    if plot_res:
        print('Showing the histograms of input and sigma-clipped images;')
        plt.close()
        vmin, vmax = np.nanpercentile(fd_cal_stat.flatten(),[0.01,99.99])
        hist = plt.hist(fd_cal_stat.flatten(), bins=np.linspace(vmin, vmax, 100), label='Input')
        hist = plt.hist(fd_cal_clip, bins=np.linspace(vmin, vmax, 100), label='Sigma-clipped')
        plt.legend(loc=0)
        plt.title('Histogram of background pixels')
        plt.show()

    # This is the pure-background image.
    fd_cal_fin = fd_cal_stat.copy()
    con = (fd_cal_fin>fd_max)
    fd_cal_fin[con] = np.nan


    #
    # 2. see 1/f noise in Fourier space;
    #
    if plot_res:
        print('Showing the sigma-clipped image in Fourier space;')
        img = fd_cal_fin.copy()
        con = np.where(np.isnan(img))
        img[con] = 0

        f = np.fft.fft2(img)
        f_s = np.fft.fftshift(f)

        plt.close()
        plt.imshow(np.log(abs(f_s)), cmap='gray')
        plt.title('Input image in Fourier space')
        plt.show()


    #
    # 3. Subtract 1/f noise by following the method proposed by Schlawin et al.
    #

    # 3.1 Global background in each apmlifiers;

    dely = 512 # Maybe specific to JWST detector;
    yamp_low = np.arange(0, 2048, dely) # this should be 4
    nyamps = len(yamp_low)

    fd_cal_ampsub = fd_cal.copy()

    sky_amp = np.zeros(nyamps, float)
    for aa in range(nyamps):
        fd_cal_amp_tmp = fd_cal_fin[yamp_low[aa]:yamp_low[aa]+dely,:]
        sky_amp[aa] = np.nanmedian(fd_cal_amp_tmp)
        fd_cal_ampsub[yamp_low[aa]:yamp_low[aa]+dely,:] -= sky_amp[aa]
        
    # 3.2 Then 1/f noise;
    # This goes through each column (to x direction) at each amplifier.
    delx = 1
    xamp_low = np.arange(0, 2048, delx)
    nxamps = len(xamp_low)

    fd_cal_ampsub_fsub = fd_cal_ampsub.copy()
    sky_f = np.zeros((nyamps,nxamps), float)
    for aa in range(nyamps):
        print('Working on the %dth apmlifier'%aa)
        for bb in range(nxamps):
            fd_cal_amp_tmp = fd_cal_ampsub[yamp_low[aa]:yamp_low[aa]+dely, xamp_low[bb]:xamp_low[bb]+delx]
            sky_f[aa,bb] = np.nanmedian(fd_cal_amp_tmp)
            fd_cal_ampsub_fsub[yamp_low[aa]:yamp_low[aa]+dely, xamp_low[bb]:xamp_low[bb]+delx] -= sky_f[aa,bb]


    #
    # 4. Output
    #
    if f_write:
        if file_out == None:
            file_out = file_cal.replace('.fits','_bbpn.fits')

        os.system('cp %s %s'%(file_cal,file_out))
        with fits.open(file_out, mode='update') as hdul:
            hdul['SCI'].data = fd_cal_ampsub_fsub
            hdul.flush()


    # 
    # 5. Check results in Fourier space
    #
    if plot_res:
        plt.close()
        fd_cal_ampsub_fsub_bg = fd_cal_ampsub_fsub.copy()
        con = np.where((fd_seg > 0) | (dq_cal>0))
        fd_cal_ampsub_fsub_bg[con] = np.nan
        fd_cal_clip_fsub = sigma_clip(fd_cal_ampsub_fsub_bg.flatten(), sigma=sigma_1, maxiters=maxiters_1,
                                cenfunc=mean, masked=False, copy=False)

        fd_stats_fsub = np.nanpercentile(fd_cal_clip_fsub, [0.1,50,99.9])
        fd_max_fsub = fd_stats_fsub.max()
        con = np.where((fd_cal_ampsub_fsub_bg>fd_max_fsub))
        fd_cal_ampsub_fsub_bg[con] = np.nan

        img_fsub = fd_cal_ampsub_fsub_bg.copy()
        con = np.where(np.isnan(img_fsub))
        img_fsub[con] = -1

        f = np.fft.fft2(img_fsub)
        f_s = np.fft.fftshift(f)

        plt.imshow(np.log(abs(f_s)), cmap='gray')
        plt.title('Final image in Fourier space')
        plt.show()

    return fd_cal_ampsub_fsub