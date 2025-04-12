import numpy as np
from numpy import mean
import matplotlib.pyplot as plt
import os,sys
from scipy import stats
import statistics

from astropy.io import fits,ascii
from astropy.convolution import Gaussian2DKernel,convolve_fft
from astropy.stats import sigma_clip
from photutils import Background2D, MedianBackground, detect_sources, deblend_sources#, source_properties


def get_sciplot(fd_cal, file_out=None, vmin=None, vmax=None, y2max=None, x3max=None,
                nxbin=2048, perc=[5,95], dpi=150, pdf=False, scl_ax=3.0, lock_ax=False):
    '''Note that for NIRCam, plot is transversed, so it looks different from the input fits in e.g., DS9.
    '''
    fig_mosaic = """
    AAAAAAC
    AAAAAAC
    AAAAAAC
    AAAAAAC
    AAAAAAC
    AAAAAAC
    BBBBBB.
    """
    fig,axes = plt.subplot_mosaic(mosaic=fig_mosaic, figsize=(5.5,5.5), constrained_layout=True)
    fig.subplots_adjust(top=0.98, bottom=0.1, left=0.1, right=0.98, hspace=0.01, wspace=0.01)
    ax1 = axes['A']
    ax2 = axes['B']
    ax3 = axes['C']

    if vmin==None or vmax==None:
        fd_cal -= np.nanmedian(fd_cal)
        vmin, vmax = np.nanpercentile(fd_cal, perc)
    ax1.imshow(fd_cal, vmin=vmin, vmax=vmax, origin='lower', aspect='auto')

    # yy = np.linspace(0, fd_cal.shape[0], nxbin)
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

    ax2.plot(xx, fx, ls='-', color='r', lw=1.)
    ax3.plot(fy, xx, ls='-', color='r', lw=1.)
    lock_ax = True
    if y2max == None or x3max == None:
        y2max = np.nanpercentile(np.abs(fx),perc[1]) * scl_ax
        if lock_ax:
            x3max = y2max
        else:
            x3max = np.nanpercentile(np.abs(fy),perc[1]) * scl_ax

    # Zero point;
    xx = np.arange(0, fd_cal.shape[1], 10)
    yy = np.arange(0, fd_cal.shape[0], 10)
    ax2.plot(xx, xx*0, ls='--', color='k', lw=0.5)
    ax3.plot(yy*0, yy, ls='--', color='k', lw=0.5)

    ax2.set_ylim(-y2max,y2max)
    ax3.set_xlim(-x3max,x3max)

    # plt.tick_params(
    #     axis='both',       # changes apply to the x-axis
    #     which='both',      # both major and minor ticks are affected
    #     bottom='off',      # ticks along the bottom edge are off
    #     top='off',         # ticks along the top edge are off
    #     left='off',      # ticks along the bottom edge are off
    #     right='off',         # ticks along the top edge are off
    #     labelbottom='off', # labels along the bottom edge are off
    #     labelleft='off') # labels along the bottom edge are off

    plt.setp(ax3.get_xticklabels(), visible=False)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.setp(ax1.get_xticklabels(), visible=False)
    # plt.setp(ax1.get_yticklabels(), visible=False)

    # plt.tight_layout()
    # pdf = True
    if pdf:
        plt.savefig(file_out.replace('.png', '.pdf'), dpi=dpi)
    else:
        plt.savefig(file_out, dpi=dpi)
    plt.close()
    return vmin, vmax, y2max, x3max


def run(file_cal, file_seg=None, f_sbtr_amp=True, f_sbtr_each_amp=True, f_only_global=False,
    plot_res=False, plot_out='./bbpn_out/', file_out=None, f_write=True, 
    sigma=2.5, maxiters=5, sigma_1=1.5, maxiters_1=30, nfracpix_min=0.5, nsig_sky=1.5,
    verbose=True, ymax=2048, mask_jump=False,
    bkg_size=20, bkg_filt_size=3, 
    cluster_field=True):
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
    ymax : int
         detector size
    nfracpix_min : float
        Minimum fraction of number for the bkg pixels required for subtraction

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
    READPATT = hd_cal['READPATT']
    INSTRUME = hd_cal['INSTRUME']
    fd_cal = fits.open(file_cal)['SCI'].data
    dq_cal = fits.open(file_cal)['DQ'].data
    fd_seg = fits.open(file_seg)[0].data

    if cluster_field:
        bkg_size_icl = 64
        bkg_filt_size_icl = 9
        bkg_estimator = MedianBackground()
        bkg_icl = Background2D(fd_cal, (bkg_size_icl,bkg_size_icl), filter_size=(bkg_filt_size_icl,bkg_filt_size_icl), bkg_estimator=bkg_estimator, exclude_percentile=100)
        fd_cal -= bkg_icl.background

    # Mask miri;
    if INSTRUME == 'MIRI':
        fd_cal[:748,:359] = np.nan
        fd_cal[748:,280:359] = np.nan
        fd_cal[:,1020:] = np.nan
        # DQ
        dq_cal[:748,:359] = 1
        dq_cal[748:,280:359] = 1
        dq_cal[:,1020:] = 1

        arr_y = np.tile(np.arange(fd_cal.shape[0]), (fd_cal.shape[1], 1)).T
        arr_x = np.tile(np.arange(fd_cal.shape[1]), (fd_cal.shape[0], 1))
        con_lyot = np.where( (dq_cal>200000) & (arr_y>748) & (arr_x<359))
        fd_cal[con_lyot] = np.nan
        if mask_jump:
            con_jump = np.where( (dq_cal==4) )
            fd_cal[con_jump] = np.nan

    # Keep record of 0 array;
    # These are from Grizli's level 1 pipeline;
    con_zero = np.where((fd_cal == 0) & (dq_cal > 0))
    fd_cal[con_zero] = np.nan

    if INSTRUME == 'NIRCAM' or INSTRUME == 'MIRI':
        if verbose:
            print('NIRCam image. Transversing')
        fd_cal = fd_cal.T
        dq_cal = dq_cal.T
        fd_seg = fd_seg.T

    #
    # 1. Exclude positive pixels originating from sources;
    #
    print('Working on %s'%file_cal.split('/')[-1])
    if INSTRUME == 'MIRI':
        var_flat_lim = 1e-4
        flat_cal = fits.open(file_cal)['VAR_FLAT'].data.T
        con = np.where((fd_seg > 0) | (dq_cal>0) | (flat_cal < var_flat_lim))
        # f_only_global = True
        # f_sbtr_amp = False
        f_sbtr_each_amp = False
        ymax = 1024
        # print('MIRI image - only global sky subtraction is applied.')
        # file_bkg = file_cal.replace('.fits','_bkg.fits')
        # fd_bkg = fits.open(file_bkg)[0].data
    else:
        con = np.where((fd_seg > 0) | (dq_cal>0))

    fd_cal_stat = fd_cal.copy()
    fd_cal_stat[con] = np.nan

    fd_cal_clip = sigma_clip(fd_cal_stat.flatten(), sigma=sigma_1, maxiters=maxiters_1,
                            cenfunc=mean, masked=False, copy=False)

    # fd_stats = np.nanpercentile(fd_cal_clip, [0.1,50,99.9])
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
    # if INSTRUME == 'MIRI':
    #     dely = 100
    yamp_low = np.arange(0, ymax, dely) # this should be 4
    nyamps = len(yamp_low)

    print('3.1 Global background in each apmlifiers')
    fd_cal_ampsub = fd_cal.copy()
    if f_sbtr_amp:
        sky_amp = np.zeros(nyamps, float)
        sky_sigma_amp = np.zeros(nyamps, float)
        for aa in range(nyamps):
            if verbose:
                print('Working on the %dth apmlifier'%aa)
            fd_cal_amp_tmp = fd_cal_fin[:,yamp_low[aa]:yamp_low[aa]+dely]
            if False:#True:
                # sky_amp[aa] = np.nanmedian(fd_cal_amp_tmp)
                sky_amp[aa] = 0
                sky_sigma_amp[aa] = np.nanstd(fd_cal_amp_tmp)
            else:
                filtered_data = sigma_clip(fd_cal_amp_tmp, sigma=sigma, maxiters=maxiters, cenfunc='median')
                sky_amp[aa] = np.nanmedian(fd_cal_amp_tmp[~filtered_data.mask])
                sky_sigma_amp[aa] = np.nanstd(fd_cal_amp_tmp[~filtered_data.mask]-sky_amp[aa])

            fd_cal_ampsub[:,yamp_low[aa]:yamp_low[aa]+dely] -= sky_amp[aa]

    # 3.2 Then 1/f noise;
    # This goes through each column (to x direction) at each amplifier.
    delx = 1
    xamp_low = np.arange(4, ymax-4, delx)
    nxamps = len(xamp_low)

    if not f_sbtr_each_amp:
        # Then, subtract global sky
        dely = ymax
        yamp_low = np.arange(0, ymax, dely)
        nyamps = len(yamp_low)

    fd_cal_ampsub_fsub = fd_cal_ampsub.copy()
    
    if f_only_global:
        delx = 2040
        xamp_low = np.arange(4, ymax-4, delx)
        nxamps = len(xamp_low)

    # For deep read, reduce nyamps;
    if READPATT[:4] == 'DEEP':
        dely = int(dely*2)
        yamp_low = np.arange(0, ymax, dely)
        nyamps = len(yamp_low)

    sky_f = np.zeros((nyamps,nxamps), float)

    print('3.2 1/f subtraction in each apmlifiers')
    if False:#INSTRUME == 'MIRI':
        fd_cal_ampsub_fsub[:,:] -= fd_bkg
    else:
        flags_skip = np.zeros((nyamps,nxamps),int)
        for aa in range(nyamps):
            if verbose:
                print('Working on the %dth apmlifier'%aa)
            for bb in range(nxamps):
                fd_cal_amp_tmp = fd_cal_ampsub[yamp_low[aa]:yamp_low[aa]+dely, xamp_low[bb]:xamp_low[bb]+delx].flatten()
                if True:#False:
                    # res = stats.sigma_clipped_stats(fd_cal_amp_tmp, sigma=sigma, maxiters=maxiters, cenfunc=mean)  
                    filtered_data = sigma_clip(fd_cal_amp_tmp, sigma=sigma, maxiters=maxiters, cenfunc=mean)
                    mask = np.where(np.abs(fd_cal_amp_tmp[~filtered_data.mask]) < sky_amp[aa] + sky_sigma_amp[aa] * nsig_sky)
                    npix = len(fd_cal_amp_tmp[~filtered_data.mask][mask])
                    sky_f[aa,bb] = np.nanmedian(fd_cal_amp_tmp[~filtered_data.mask][mask])
                else:
                    con = np.where(np.abs(fd_cal_amp_tmp) < sky_amp[aa] + sky_sigma_amp[aa] * nsig_sky)
                    npix = len(fd_cal_amp_tmp[con])

                npix_min = nfracpix_min * fd_cal_amp_tmp.shape[0]
                # if aa == 2 and bb>1300 and bb<1700:
                #     print(nfracpix_min, fd_cal_amp_tmp.shape[0], npix)
                #     print(bb, res)
                if npix < npix_min:
                    # if verbose:
                    #     print('not enough pixel; skipping 1/f subtraction.')
                    flags_skip[aa,bb] = 1
                    continue
                # sky_f[aa,bb] = np.nanmedian(fd_cal_amp_tmp[con])
                # sky_f[aa,bb] = np.nanmedian(filtered_data)
                fd_cal_ampsub_fsub[yamp_low[aa]:yamp_low[aa]+dely, xamp_low[bb]:xamp_low[bb]+delx] -= sky_f[aa,bb]

        # Revisit only flagged;
        if True:
            mask = np.where(flags_skip)
            for ii in range(len(mask[0])):
                aa = mask[0][ii]
                bb = mask[1][ii]
                con = (flags_skip[:,bb] == 0)
                if len(sky_f[:,bb][con])>0:

                    # Current estimate of bkg, which is likely overestimated;
                    # Correct sky_f[aa,bb] using one of the two ref sky values;
            
                    # Ref sky - Good pixels in the same row, including other amplifier;
                    fd_tmp = fd_cal_ampsub_fsub[:, xamp_low[bb]:xamp_low[bb]+delx].flatten()
                    filtered_data = sigma_clip(fd_tmp, sigma=sigma, maxiters=maxiters, cenfunc=mean)
                    skydata1 = fd_tmp[~filtered_data.mask]
                    sky_ref = np.nanmedian(skydata1)

                    # Ref sky - Good pixels in the same column #, including other amplifier;
                    if False:
                        fd_tmp = sky_f[aa,:]
                        filtered_data = sigma_clip(fd_tmp, sigma=sigma, maxiters=maxiters, cenfunc=mean)
                        skydata2 = fd_tmp[~filtered_data.mask]
                        sky_ref_amp = np.nanmedian(skydata2)
                        
                        # Combine?
                        skydata3 = (np.concatenate([skydata1,skydata2]))
                        sky_ref_both = np.nanmedian(skydata3)

                    # Correct sky;
                    sky_f[aa,bb] -= sky_ref
                    fd_cal_ampsub_fsub[yamp_low[aa]:yamp_low[aa]+dely, xamp_low[bb]:xamp_low[bb]+delx] -= sky_f[aa,bb]

        if INSTRUME == 'MIRI' and not f_only_global:
            # subtract stripes in the other direction;
            print('MIRI; subtracting 1/f in the other direction;')
            for aa in range(nyamps):
                if verbose:
                    print('Working on the %dth apmlifier'%aa)
                for bb in range(nxamps):
                    fd_cal_amp_tmp = fd_cal_ampsub[xamp_low[bb]:xamp_low[bb]+delx, yamp_low[aa]:yamp_low[aa]+dely]
                    filtered_data = sigma_clip(fd_cal_amp_tmp, sigma=sigma, maxiters=maxiters)
                    # sky_f[aa,bb] = np.nanmedian(fd_cal_amp_tmp[~filtered_data.mask])
                    sky_tmp = np.nanmedian(fd_cal_amp_tmp[~filtered_data.mask])
                    fd_cal_ampsub_fsub[xamp_low[bb]:xamp_low[bb]+delx, yamp_low[aa]:yamp_low[aa]+dely] -= sky_tmp


    if not cluster_field:
        # One last bkg tweak, to eliminate discontinuity;
        data_for_bkg = fd_cal_ampsub_fsub.copy()
        con_for_bkg = np.where((fd_seg > 0) | (dq_cal>0))
        data_for_bkg[con_for_bkg] = np.nan

        bkg_estimator = MedianBackground()
        bkg = Background2D(data_for_bkg, (bkg_size,bkg_size), filter_size=(bkg_filt_size,bkg_filt_size), bkg_estimator=bkg_estimator, exclude_percentile=100)
        fd_cal_ampsub_fsub -= bkg.background

    # mask nan for miri;
    # if INSTRUME == 'MIRI':
        # con_nan = np.where(flat_cal < var_flat_lim)
        # fd_cal_ampsub_fsub[con_nan] = np.nan
        # fd_cal_ampsub_fsub[:] = np.nan

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
        plt.close()
        # plt.show()

    #
    # 5. Output
    #
    if cluster_field:
        # One last bkg tweak, to eliminate discontinuity;
        fd_cal_ampsub_fsub += bkg_icl.background

    if f_write:
        if file_out == None:
            file_out = file_cal.replace('.fits','_bbpn.fits')

        os.system('cp %s %s'%(file_cal,file_out))
        with fits.open(file_out, mode='update') as hdul:

            if INSTRUME == 'NIRCAM' or INSTRUME == 'MIRI':
                if verbose:
                    print('NIRCam image. Transversing')
                fd_cal_ampsub_fsub = fd_cal_ampsub_fsub.T
                dq_cal = dq_cal.T

            # Retrieve 0-value pixels back?
            # fd_cal_ampsub_fsub[con_zero] = 0
            fd_cal_ampsub_fsub[con_zero] = np.nan

            hdul['SCI'].data = fd_cal_ampsub_fsub
            hdul['DQ'].data = dq_cal
            hdul.flush()

    return fd_cal_ampsub_fsub
