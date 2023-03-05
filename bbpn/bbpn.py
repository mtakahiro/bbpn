import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
import os,sys
from scipy import stats
import statistics

from astropy.io import fits,ascii
from astropy.convolution import Gaussian2DKernel,convolve_fft
from astropy.stats import sigma_clip


def get_sciplot(fd_cal, file_out=None, vmin=None, vmax=None, y2max=None, x3max=None,
                nxbin = 1000):
    '''Note that for NIRCam, plot is transversed, so it looks different from the input fits in e.g., DS9.
    '''
    fig_mosaic = """
    AAAAAC
    AAAAAC
    AAAAAC
    AAAAAC
    AAAAAC
    BBBBBD
    """
    fig,axes = plt.subplot_mosaic(mosaic=fig_mosaic, figsize=(5.5,5.5))
    fig.subplots_adjust(top=0.98, bottom=0.16, left=0.08, right=0.99, hspace=0.15, wspace=0.25)
    ax1 = axes['A']
    ax2 = axes['B']
    ax3 = axes['C']
    ax4 = axes['D']
    ax4.set_frame_on(False)

    if vmin==None or vmax==None:
        vmin, vmax = np.nanpercentile(fd_cal, [1,99])
    ax1.imshow(fd_cal, vmin=vmin, vmax=vmax, origin='lower')

    yy = np.linspace(0, fd_cal.shape[0], nxbin)
    xx = np.linspace(0, fd_cal.shape[1], nxbin)
    fx = np.zeros(len(xx), float)
    fy = np.zeros(len(xx), float)
    for nx,x in enumerate(xx):
        if nx == len(xx)-1:
            continue
        nx1 = int(xx[nx])
        nx2 = int(xx[nx+1])
        fx[nx] = np.nanmedian(fd_cal[:,nx1:nx2])
        fy[nx] = np.nanmedian(fd_cal[nx1:nx2,:])

    ax2.plot(xx, fx, ls='-', color='lightblue', lw=1)
    ax3.plot(fy, xx, ls='-', color='lightblue', lw=1)
    if y2max == None or x3max == None:
        y2max = np.nanpercentile(np.abs(fx),99) * 1.5
        x3max = np.nanpercentile(np.abs(fy),99) * 1.5

    # Zero point;
    xx = np.arange(0, fd_cal.shape[1], 10)
    yy = np.arange(0, fd_cal.shape[0], 10)
    ax2.plot(xx, xx*0, ls='--', color='k', lw=0.5)
    ax3.plot(yy*0, yy, ls='--', color='k', lw=0.5)

    ax2.set_ylim(-y2max,y2max)
    ax3.set_xlim(-x3max,x3max)

    plt.tick_params(
        axis='both',       # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        left='off',      # ticks along the bottom edge are off
        right='off',         # ticks along the top edge are off
        labelbottom='off', # labels along the bottom edge are off
        labelleft='off') # labels along the bottom edge are off
    plt.setp(ax4.get_xticklabels(), visible=False)
    plt.setp(ax4.get_yticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    plt.setp(ax1.get_yticklabels(), visible=False)

    plt.tight_layout()
    plt.savefig(file_out)
    plt.close()
    return vmin, vmax, y2max, x3max


def run(file_cal, file_seg=None, f_sbtr_amp=True, f_sbtr_each_amp=True, f_only_global=False,
    plot_res=False, plot_out='./bbpn_out/', file_out=None, f_write=True, sigma=1.5, 
    maxiters=5, sigma_1=1.5, maxiters_1=10,
    verbose=True):
    '''
    Parameters
    ----------
    file_cal : str
        cal.fits file of JWST.
    file_seg : str
        segmentation mask for file_cal. The data extension is assumed to be 0.
    f_sbtr_each_amp : bool
        Subtract 1/f noise at each of four amplifiers.
    plot_res : bool
        Show results of each step.
    f_sbtr_amp : bool
        Subtract (non-1/f) bkg at each of four amplifiers.

    Returns
    -------
    fd_cal_ampsub_fsub : 2d-array
        1/f noise subtracted image array.

    '''
    if file_seg == None:
        file_seg = file_cal.replace('.fits','_seg.fits')
        if not os.path.exists(file_seg):
            print('Segmap, %s, is not found. Exiting.'%file_seg)
            return False

    # Open files;
    hd_cal = fits.open(file_cal)[0].header
    INSTRUME = hd_cal['INSTRUME']
    fd_cal = fits.open(file_cal)['SCI'].data
    dq_cal = fits.open(file_cal)['DQ'].data
    fd_seg = fits.open(file_seg)[0].data

    if INSTRUME == 'NIRCAM':
        if verbose:
            print('NIRCam image. Transversing')
        fd_cal = fd_cal.T
        dq_cal = dq_cal.T
        fd_seg = fd_seg.T


    #
    # 1. Exclude positive pixels originating from sources;
    #
    con = np.where((fd_seg > 0) | (dq_cal>0))
    fd_cal_stat = fd_cal.copy()
    fd_cal_stat[con] = np.nan

    fd_cal_clip = sigma_clip(fd_cal_stat.flatten(), sigma=sigma_1, maxiters=maxiters_1,
                            cenfunc=mean, masked=False, copy=False)

    fd_stats = np.nanpercentile(fd_cal_clip, [0.1,50,99.9])
    fd_max = fd_cal_clip.max()

    if plot_res:
        if not os.path.exists(plot_out):
            os.mkdir(plot_out)

        plt.close()
        file_png_out = os.path.join(plot_out,'%s'%(file_cal.split('/')[-1].replace('_cal.fits', '_cal.png')))
        fd_cal_masked = fd_cal.copy()
        conseg = np.where((fd_seg > 0))
        fd_cal_masked[conseg] = np.nan
        vmin, vmax, y2max, x3max = get_sciplot(fd_cal_masked, file_out=file_png_out)
        plt.close()

        print('Showing the histograms of input and sigma-clipped images;')
        vminh, vmaxh = np.nanpercentile(fd_cal_stat.flatten(),[0.01,99.99])
        hist = plt.hist(fd_cal_stat.flatten(), bins=np.linspace(vminh, vmaxh, 100), label='Input')
        hist = plt.hist(fd_cal_clip, bins=np.linspace(vminh, vmaxh, 100), label='Sigma-clipped')
        plt.legend(loc=0)
        plt.title('Histogram of background pixels')
        plt.savefig(os.path.join(plot_out,'%s'%(file_cal.split('/')[-1].replace('_cal.fits', '_cal_hist.png'))))
        # plt.show()

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
        # plt.show()
        plt.savefig(os.path.join(plot_out,'%s'%(file_cal.split('/')[-1].replace('_cal.fits', '_cal_fourier.png'))))

    #
    # 3. Subtract 1/f noise by following the method proposed by Schlawin et al.
    #

    # 3.1 Global background in each apmlifiers;
    # Note that NIRISS and NIRCam is different...
    dely = 512 # Maybe specific to JWST detector;
    yamp_low = np.arange(0, 2048, dely) # this should be 4
    nyamps = len(yamp_low)

    fd_cal_ampsub = fd_cal.copy()
    if f_sbtr_amp:
        sky_amp = np.zeros(nyamps, float)
        for aa in range(nyamps):
            print('Working on the %dth apmlifier'%aa)
            fd_cal_amp_tmp = fd_cal_fin[:,yamp_low[aa]:yamp_low[aa]+dely]
            sky_amp[aa] = np.nanmedian(fd_cal_amp_tmp)
            fd_cal_ampsub[:,yamp_low[aa]:yamp_low[aa]+dely] -= sky_amp[aa]

    # 3.2 Then 1/f noise;
    # This goes through each column (to x direction) at each amplifier.
    delx = 1
    xamp_low = np.arange(4, 2044, delx)
    nxamps = len(xamp_low)

    if not f_sbtr_each_amp:
        # Then, subtract global sky
        dely = 2048
        yamp_low = np.arange(0, 2048, dely)
        nyamps = len(yamp_low)

    fd_cal_ampsub_fsub = fd_cal_ampsub.copy()
    sky_f = np.zeros((nyamps,nxamps), float)
    
    if f_only_global:
        delx = 2040
        xamp_low = np.arange(4, 2044, 2040)
        nxamps = len(xamp_low)

    for aa in range(nyamps):
        print('Working on the %dth apmlifier'%aa)
        for bb in range(nxamps):
            fd_cal_amp_tmp = fd_cal_ampsub[yamp_low[aa]:yamp_low[aa]+dely, xamp_low[bb]:xamp_low[bb]+delx]
            filtered_data = sigma_clip(fd_cal_amp_tmp, sigma=sigma, maxiters=maxiters)
            sky_f[aa,bb] = np.nanmedian(fd_cal_amp_tmp[~filtered_data.mask])
            fd_cal_ampsub_fsub[yamp_low[aa]:yamp_low[aa]+dely, xamp_low[bb]:xamp_low[bb]+delx] -= sky_f[aa,bb]


    # 
    # 4. Check results in Fourier space
    #
    if plot_res:

        plt.close()
        file_png_out = os.path.join(plot_out,'%s'%(file_cal.split('/')[-1].replace('_cal.fits', '_cal_cor.png')))
        fd_cal_ampsub_fsub_masked = fd_cal_ampsub_fsub.copy()
        conseg = np.where((fd_seg > 0))
        fd_cal_ampsub_fsub_masked[conseg] = np.nan
        get_sciplot(fd_cal_ampsub_fsub_masked, file_out=file_png_out, y2max=y2max, x3max=x3max)#, vmin=vmin, vmax=vmax
        plt.close()

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
        plt.savefig(os.path.join(plot_out,'%s'%(file_cal.split('/')[-1].replace('_cal.fits', '_cal_cor_fourier.png'))))
        # plt.show()

    #
    # 5. Output
    #
    if f_write:
        if file_out == None:
            file_out = file_cal.replace('.fits','_bbpn.fits')

        os.system('cp %s %s'%(file_cal,file_out))
        with fits.open(file_out, mode='update') as hdul:

            if INSTRUME == 'NIRCAM':
                if verbose:
                    print('NIRCam image. Transversing')
                fd_cal_ampsub_fsub = fd_cal_ampsub_fsub.T

            hdul['SCI'].data = fd_cal_ampsub_fsub
            hdul.flush()

    return fd_cal_ampsub_fsub
