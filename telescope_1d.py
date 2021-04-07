#
# This code runs with CCL + python3
#
# Will run with https://hub.docker.com/repository/docker/slosar/seconda
# on 3/15/21
#

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
from numpy.fft import rfft,irfft
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
from matplotlib.colors import LogNorm
from itertools import combinations
from astropy.cosmology import Planck15 as cosmo
from joblib import Parallel, delayed
import functools

class Telescope1D:
    def __init__(self, Nfreq=256, Ndishes=32, DDish=6, Npix_fft=2**12, Npad=2**8,
                 minfreq=400, maxfreq=800, redundant=False, seed=0):
        '''
        DDish is the diameter of the dishes in meters.
        The minfreq, maxfreq should be in MHz.
        If redundant is True, it is perfectly redundant; if False, it is
        less redundant.
        '''
        self.Npix_fft = Npix_fft
        self.Npix = self.Npix_fft+1
        self.Npad = Npad
        self.Nfft = Npix_fft*Npad
        self.Nfreq = Nfreq
        self.minfreq = minfreq
        self.maxfreq = maxfreq
        self.Ndishes = Ndishes
        self.DDish = DDish
        self.redundant = redundant
        self.matrix = None

        # If redundant = True, the distance between consecutive dishes is DDish
        dish_locations = np.arange(0,Ndishes,dtype=float)
        np.random.seed(seed)
        for ii in np.arange(1,Ndishes):
            if redundant:
                distance = DDish # This makes redundant array
            else:
                #distance = DDish*((ii-1)%5*0.25+1)
                # Distances between two consecutive dishes are
                # chosen randomly from DDish, 1.25*DDish, 1.5*DDish
                distance = DDish*(1+np.random.randint(0,3)/4)
            dish_locations[ii] = dish_locations[ii-1]+distance
        self.dish_locations = dish_locations
        
        # Get every baseline combination, compute baseline lengths
        self.baseline_lengths = np.array([d2-d1 for d1, d2 in
                                          combinations(dish_locations,2)])
        self.unique_baseline_lengths = np.unique(self.baseline_lengths)

        # Get 2D array of (N freqs x N unique baselines) containg D/lambda
        self.DoL = np.outer(1/self.lams, self.unique_baseline_lengths)

    @property
    def freqs(self):
        '''
        Get array of frequencies.
        '''
        return np.linspace(self.minfreq,self.maxfreq,self.Nfreq)

    @property
    def lams(self):
        '''
        Get array of wavelengths.
        '''
        return self.freq2lam(self.freqs)

    def predict_point(self, alpha):
        '''
        Predicts signal from a point source (predicts what the foregrounds
        at this alpha look like).
        '''
        return np.exp(-2j*np.pi*self.DoL*alpha)

    @functools.lru_cache()
    def get_FG_filtering_matrix_inverse(self, step=1):
        '''
        Get the matrix to use in filter_FG.
        '''
        p2fac = self.get_p2fac()
        C = None
        for i,alpha in list(enumerate(self.alpha))[::step]:
            fsup = p2fac[:,i]
            p = (self.predict_point(alpha)*fsup[:,None]).flatten()
            if C is None:
                C=np.outer(p,np.conj(p))
            else:
                C+=np.outer(p,np.conj(p))
        return C


    def decompose_filtering(self, matrix):
        eva, eve = np.linalg.eigh(matrix,'L')
        return eva, eve
        

    def filter_FG(self, uvplane, scale=1e-11, matrix=None):
        '''
        Filter out the foregrounds (foregrounds are frequency independent).
        Adjust scale to tune the number of eigenvalues being filtered.
        '''
        if matrix is not None:
            eva,eve = self.decompose_filtering(matrix)
        else:
            if not hasattr(self,"eva"):
                print ("caching matrix")
                self.eva = self.decompose_filtering(
                    self.get_FG_filtering_matrix_inverse())
            eva,eve = self.eva

        minval = np.abs(eva).max()*scale
        out = np.copy(uvplane).flatten()
        cc = 0
        for val, vec in zip(eva, eve.T):
            if np.abs(val)>minval:
                cvec = np.conj(vec)
                x = np.dot(out,cvec)
                out -= x*vec
                x = np.dot(out,vec)
                out -= x*cvec
                cc += 1
        print(f"Filtered {cc} modes.")
        return out.reshape(uvplane.shape)

    def filter_FG_per_baseline(self, uvplane, scale=1e-11):
        '''
        Filter out the foregrounds (foregrounds are frequency independent).
        Some as filter_FG, but filters per unique baseline
        '''
        if not hasattr(self,"sing_eva"): ## 
            print ("caching matrices")
            matrix = self.get_FG_filtering_matrix_inverse()
            evas = []
            Nb=len(self.unique_baseline_lengths)
            for i in range(Nb):
                st,en = i*self.Nfreq, (i+1)*self.Nfreq
                
                #evas.append(self.decompose_filtering(matrix[st:en,st:en]))
                evas.append(self.decompose_filtering(matrix[i::Nb,i::Nb]))
            self.sing_eva=evas
        ## now we do ths per baseline

        out = np.copy(uvplane)
        for i,(eva,eve)  in enumerate(self.sing_eva):
            minval = np.abs(eva).max()*scale
            cc = 0
            for val, vec in zip(eva, eve.T):
                if np.abs(val)>minval:
                    cvec = np.conj(vec)
                    x = np.dot(out[:,i],cvec)
                    out[:,i] -= x*vec
                    x = np.dot(out[:,i],vec)
                    out[:,i] -= x*cvec
                    cc += 1
            print(f"Filtered {cc} modes for baseline {i}.")
        return out



    
    def freq2lam(self, freq_MHz):
        '''
        Turns frequency (in MHz) into wavelength (in m).
        '''
        return 3e8/(freq_MHz*1e6)

    def image2uv(self, image):
        '''
        Convert from pixel space to uv space by FFT.
        '''
        #assert(len(image)==self.Npix_fft)
        bigimage = np.zeros(self.Npix_fft*self.Npad)
        # Put last half of image in beginning of bigimage
        bigimage[:self.Npix_fft//2+1] = image[-self.Npix_fft//2-1:]
        # Put first half of image at end of bigimage
        bigimage[-self.Npix_fft//2:] = image[:self.Npix_fft//2]
        return rfft(bigimage)

    def uv2image(self, uv):
        '''
        Convert from uv space to pixel space by inverse FFT.
        '''
        assert(len(uv) == self.Nfft//2+1)
        bigimage = irfft(uv)
        # Concatenate the last chunk of bigimage to first chunk of bigimage
        image = np.hstack((bigimage[-self.Npix_fft//2:],bigimage[:self.Npix_fft//2+1]))
        return image

    @property
    def alpha(self):
        '''
        Returns sin(theta), where theta describes angular coordinates from
        -pi/2 to +pi/2 with theta = 0 pointing at the zenith.
        '''
        #return (np.arange(-self.Npix_fft//2,self.Npix_fft//2)+0.5)*(2/self.Npix_fft)
        return np.linspace(-1,1,self.Npix)

    def empty_uv(self):
        return np.zeros(self.Nfft//2+1,np.complex)

    def empty_image(self):
        return np.zeros(self.Npix,np.float)

    def DoL2ndx(self, DoL):
        '''
        Returns the index in the uv plane of the D/lambda.
        DoL can be a vector or array.
        Not nearest integer; this returns a float.
        '''
        return DoL*self.Npad*2

    def primary_beam_1(self, freq_MHz):
        '''
        Take frequency in MHz, return beam.
        '''
        lam = self.freq2lam(freq_MHz)
        t = self.empty_uv()
        t[:int(self.DoL2ndx(self.DDish/lam/2))] = 1.0
        return self.uv2image(t)

    def uv2uvplane(self, uv, indices=None):
        '''
        Instead of e.g. uvplane = uv[indices.astype(int)] (nearest
        neighbor), use linear interpolation approach to assign uv values to
        uvplane.
        '''
        if indices is None:
            indices = self.DoL2ndx(self.DoL)
        uvplane = np.zeros_like(indices, np.complex)
        if indices.ndim == 1:
            # Getting uvplane for one frequency bin
            for ii, idx_val in enumerate(indices):
                # Looping through every element of indices
                if np.ceil(idx_val).astype(int) < len(uv):
                    val1 = uv[np.floor(idx_val).astype(int)]
                    val2 = uv[np.ceil(idx_val).astype(int)]
                    # If the idx_val is 1.2, we'd do 0.2*(uv[2]-uv[1]) + uv[1]
                    val = (idx_val%1) * (val2-val1) + val1
                    uvplane[ii] = val
        elif indices.ndim == 2:
            for ii, idx_vals in enumerate(indices):
                for jj, idx_val in enumerate(idx_vals):
                    # Looping through every element of indices
                    if np.ceil(idx_val).astype(int) < len(uv):
                        val1 = uv[np.floor(idx_val).astype(int)]
                        val2 = uv[np.ceil(idx_val).astype(int)]
                        # If the idx_val is 1.2, we'd do 0.2*(uv[2]-uv[1]) + uv[1]
                        val = (idx_val%1) * (val2-val1) + val1
                        uvplane[ii,jj] = val
        return uvplane

    def get_errors(self, error_sigma=10e-12, correlated=True, seed=0, r0=None):
        '''
        Get array of time/amplitude errors for each dish.
        Make argument correlated True for correlated time errors (errors of
        neighboring dishes are more similar than errors of far away dish pairs).
        Make argument r0 bigger to make more correlated.
        '''
        np.random.seed(seed)
        if correlated:
            cov = np.zeros((self.Ndishes,self.Ndishes))
            if r0 is None:
                r0 = self.DDish
            for i in range(self.Ndishes):
                for j in range(self.Ndishes):
                    if i==j:
                        cov[i,j] = error_sigma**2
                    else:
                        baseline_distance = np.abs(self.dish_locations[j]-self.dish_locations[i]).astype(float)
                        cov[i,j] = error_sigma**2/np.sqrt(baseline_distance/r0)
            mean = np.zeros(self.Ndishes)
            errors = np.random.multivariate_normal(mean,cov)
        else:
            errors = np.random.normal(0, error_sigma,self.Ndishes)
        return errors

    def get_obs_uvplane(self, uvplane, error_sigma=10e-12, correlated=True, time_error = True, seed=0, filter_FG=True):
        '''
        Get the uvplane with time error. Argument correlated refers to whether
        or not the errors are correlated among dishes.
        The time errors are converted to phase errors, then we obtain the new
        uvplane with the errors.
        If filter_FG is True, we filter out the foregrounds after adding the
        timing errors, both the version with single baseline filtering and combined and return all three:
        If error is not time_err, it is amplitude error.

        '''
        # Add time/ampltiude errors
        if error_sigma > 0:
            errors = self.get_errors(error_sigma=error_sigma, correlated=correlated, seed=seed)
            uvplane_obs = np.zeros_like(uvplane, np.complex)
            for i, f in enumerate(self.freqs):
                if time_error:
                    phase_errors = errors*f*1e6*2*np.pi
                # Loop through each unique baseline length
                # Get and average all the observed visibilities for each
                for j, baseline_len in enumerate(self.unique_baseline_lengths):
                    redundant_baseline_idxs = np.where(self.baseline_lengths==baseline_len)[0]
                    uvplane_j = []
                    for k in redundant_baseline_idxs:
                        dish1_loc, dish2_loc = list(combinations(self.dish_locations,2))[k]
                        dish1_idx = np.where(self.dish_locations==dish1_loc)[0][0]
                        dish2_idx = np.where(self.dish_locations==dish2_loc)[0][0]
                        if time_error:
                            uvplane_j.append((np.exp(1j*(phase_errors[dish2_idx]-phase_errors[dish1_idx])))*uvplane[i,j])
                        else:
                            uvplane_j.append((1+errors[dish1_idx])*(1+errors[dish2_idx])*uvplane[i,j])
                    uvplane_j = np.array(uvplane_j, np.complex)
                    uvplane_obs[i,j] = np.mean(uvplane_j, axis=0)
        else:
            uvplane_obs = uvplane
        # Filter foregrounds
        if filter_FG:
            uvplane_obs_f = self.filter_FG(uvplane_obs)
            uvplane_obs_f1 = self.filter_FG_per_baseline(uvplane_obs)
            return (uvplane_obs,uvplane_obs_f,uvplane_obs_f1)
        else:
            return uvplane_obs
    

    def get_obs_rmap(self, uvplane, time_error_sigma=10e-12, correlated=True, seed=0, filter_FG=True):
        '''
        From the unobserved uvplane (Fourier space), get the
        rmap (real space map) observed by the telescope array.
        For no time/phase error, do time_error_sigma = 0 seconds.
        Returns (N frequencies x Npix+1) array rmap_obs.
        '''
        indices = (self.DoL2ndx(self.DoL)+0.5).astype(int)
        rmap_obs = []
        # Add time error first, then filter foregrounds
        uvplane_obs = self.get_obs_uvplane(uvplane, time_error_sigma, correlated, seed, filter_FG)
        # Then, convert to rmap
        def process_freq(i, f):
            '''
            Wrapper for multiprocessing.
            '''
            uvi = self.empty_uv()
            for ii, ind in enumerate(indices[i,:]):
                if ind < len(uvi):
                    uvi[ind] = uvplane_obs[i,ii]
            return self.uv2image(uvi)
        rmap_obs = np.array([process_freq(i,f) for i,f in enumerate(self.freqs)])
        return rmap_obs

    def plot_rmap(self, rmap, vmax=None, vmin=None):
        '''
        Plot the dirty rmap.
        '''
        plt.figure(figsize=(20,10))
        plt.imshow(rmap,aspect='auto',origin='lower', vmax=vmax, vmin=vmin,
                   extent=(self.alpha[0],self.alpha[-1],self.freqs[0],self.freqs[-1]))
        plt.ylabel('frequency [MHz]')
        plt.xlabel(r'sin($\theta$)')
        plt.colorbar()
        plt.show()

    def get_p2fac(self):
        '''
        Returns (Nfreq x Npix) array of the beam^2/cos(alpha) for convolution.
        '''
        return np.array([np.abs(self.primary_beam_1(f)**2)/np.cos(self.alpha) for f in self.freqs])

    def plot_wedge(self, Nreal=100, time_error_sigma=0, correlated=True, filter_FG=False):
        '''
        Simulate various skies, and plot the wedge.
        '''
        Nuniquebaselines = self.unique_baseline_lengths.shape[0]
        ps = np.zeros((self.Nfreq+1,Nuniquebaselines)) # (2*Nfreq/2)+1 = Nfreq+1
        for c in range(Nreal):
            # Create a random sky, this sky will be the same at all frequencies
            sky = self.get_uniform_sky(high=1, seed=c)
            uvplane = self.observe_image(sky)
            if time_error_sigma > 0:
                uvplane = self.get_obs_uvplane(uvplane=uvplane, time_error_sigma=time_error_sigma, correlated=correlated, seed=c, filter_FG=filter_FG)
            # After uvplane is done, calculate power spectrum in the frequency direction
            # This gives delay spectrum
            for j in range(Nuniquebaselines):
                # FFT along frequency axis, get the power in the frequency domain
                # Power in the frequency domain gives us structure in the redshift direction, for 21 cm
                ps[:,j] += np.abs(rfft(np.hstack((uvplane[:,j],np.zeros(self.Nfreq))))**2)
        plt.imshow(ps[:,:],origin='lower',aspect='auto',interpolation='nearest', norm=LogNorm(), cmap='jet')
        plt.xlabel(r'Baseline Length - $k_\perp$')
        plt.ylabel('Delay - FT Along Frequency Direction')
        plt.colorbar()
        plt.show()
        return ps

    def observe_image(self, image):
        '''
        Take the image, multiply it by the beam^2/cos(alpha) (this is called
        convolution). This gives us the telescope response (what the telescope array sees).
        Do for each frequency, return the uvplane (in Fourier space).
        '''
        p2fac = self.get_p2fac()
        if len(image.shape) == 1:
            msky2d = image[None,:] * p2fac
        else:
            msky2d = image * p2fac
            
        Nuniquebaselines = self.unique_baseline_lengths.shape[0]
        #uvplane = np.zeros((self.Nfreq,Nuniquebaselines),np.complex)
        def process_freq(i,f):
            msky = msky2d[i,:]
            # FT to the uvplane and sample at indices corresponding to D/lambda
            uv = self.image2uv(msky)
            return self.uv2uvplane(uv,indices=self.DoL2ndx(self.DoL)[i,:])
        # Loop over frequencies
        #uvplane = np.array([process_freq(i,f) for i,f in enumerate(self.freqs)])
        uvplane = np.array(Parallel(n_jobs=-1)(delayed(process_freq)(i,f) for i,f in enumerate(self.freqs)))
        return uvplane

    def beam_convolution(self, image):
        print ("beam_convolution deprecated due to bad name. Use observe_image instead.")
    
    def get_rmap_residuals(self, rmap_no_error, rmap_with_error, n=1,
                           vmax=None, vmin=None):
        '''
        Plot the residuals for rmap.
        n is how many frequency bins of freqs should we bin together.
        '''
        freq_vec = np.zeros(len(self.freqs)//n)
        max_rmap_no_error = np.zeros_like(rmap_no_error)
        for i in range(len(self.freqs)//n):
            freq_vec[i] = np.mean(self.freqs[i*n:(1+i)*n])
            max_rmap_no_error[i*n:(1+i)*n,:] = np.max(rmap_no_error[i*n:(1+i)*n,:])
        residuals = (rmap_with_error-rmap_no_error)/max_rmap_no_error

        # Check that max_rmap_no_error is frequency independent for the most part
        plt.plot(self.freqs, max_rmap_no_error[:,0])
        plt.xlabel('Frequency [MHz]')
        plt.ylabel('Maximum rmap_no_error Value')
        plt.show()

        # Plot the residuals
        plt.figure(figsize=(20,10))
        plt.imshow(residuals,aspect='auto',origin='lower', vmax=vmax, vmin=vmin,
                   extent=(self.alpha[0],self.alpha[-1],self.freqs[0],self.freqs[-1]))
        plt.ylabel('frequency [MHz]')
        plt.xlabel(r'sin($\theta$)')
        plt.title('Residuals')
        plt.colorbar()
        plt.show()
        return residuals

    def get_point_source_sky(self, idx=None, n=50, seed=0):
        '''
        Make sky image with point sources at locations specified by list idx.
        If idx is not specified, n point source locations are chosen at random.
        '''
        np.random.seed(seed)
        image = self.empty_image()
        if idx is None:
            idx = []
            for i in range(n):
                idx.append(np.random.randint(low=0, high=self.Npix_fft))
        else:
            n = len(idx)
        for i in range(n):
            image[idx[i]] = 1e4
        return image

    def get_gaussian_sky(self, mean=0, sigma_o=1.4e4, sigma_f=60, seed=0):
        '''
        Get a correlated Gaussian sky with the specified mean, sigma_o, and
        sigma_f.
        '''
        np.random.seed(seed)
        g = np.random.normal(mean,sigma_o,self.Npix)
        g = gaussian_filter(g,sigma=sigma_f)
        return g

    def get_poisson_sky(self, lam=0.01, seed=0):
        '''
        Get Poisson sky.
        '''
        np.random.seed(seed)
        p = np.random.poisson(lam=lam, size=self.Npix).astype(float)
        p = p * 100
        return p

    def get_uniform_sky(self, high=3500, seed=0):
        '''
        Get a uniform random sky from 0 to high.
        '''
        np.random.seed(seed)
        return np.random.uniform(0,high,self.Npix)

    def get_signal(self, level=1, seed=0):
        np.random.seed(seed)
        return np.random.normal(0,1,(self.Nfreq,self.Npix))
    

    def get_rmap_ps(self, rmap, Nfreqchunks=4, m_alpha=2, m_freq=2, padding=1, window_fn=np.blackman, plot=False, vmin=None, vmax=None, log=True):
        '''
        Get and plot the power spectrum for rmap.
        For just one full plot of the power spectrum, set Nfreqchunks as 1,
        otherwise we divide the rmap into frequency chunks and
        compute the power spectrum independently for each chunk.
        After getting the power spectra, we bin in both x and y directions;
        m is how many of the existing bins we want to put in each bin.
        '''
        # Divide into frequency chunks
        # In each chunk, FT along the line of sight and square
        n = self.Nfreq//Nfreqchunks
        ps = []
        for i in range(Nfreqchunks):
            #ps_chunk = np.zeros((n+1,self.Npix))
            #for j in range(self.Npix):
            #    ps_chunk[:,j] = np.abs(rfft(np.hstack((rmap[i*n:(1+i)*n,j],np.zeros(n))))**2)
            if window_fn is not None:
                tofft = rmap[i*n:(1+i)*n,:]*(window_fn(n)[:,None])
            else:
                tofft = rmap[i*n:(1+i)*n,:]
            if padding > 0:
                tofft = np.vstack((tofft,np.zeros((n*padding,self.Npix))))
            ps_chunk = np.abs(rfft(tofft,axis=0)**2)
            ps.append(ps_chunk)

        # After getting the power spectra, bin in both x and y directions
        n_rows = n*(1+padding)//2+1
        n_cols = self.Npix
        n_row_bins = n_rows//m_freq
        n_col_bins = n_cols//m_alpha
        # Discard some values if necessary
        for i, ps_chunk in enumerate(ps):
            ps[i] = ps_chunk[:m_freq*n_row_bins,:m_alpha*n_col_bins]
        ps_binned = []
        # Bin
        for ps_chunk in ps:
            ps_chunk_binned = ps_chunk.reshape(n_row_bins, n_rows//n_row_bins, n_col_bins, n_cols//n_col_bins).sum(axis=3).sum(axis=1)
            ps_binned.append(ps_chunk_binned)
        
        # Convert from frequency to distance (Mpc/h)
        k_modes_unbinned = []
        last_modes = []
        for i in range(Nfreqchunks):
            freq_first = self.freqs[i*n]
            freq_last = self.freqs[(1+i)*n-1]
            # Get size of the chunk (dist_max) then the fundamental mode is 2*pi/dist_max
            dist_max = self.freq2distance(freq_first, freq_last)
            # Scale k0 by m_freq/(1+padding), where m_freq is the downsampling and 1+padding is the upsampling factor
            k0 = 2*np.pi/dist_max/(1+padding)
            #print (f"Fundamental mode for chunk {i} is {k0}")
            k_modes_unbinned.append(np.arange(n_row_bins*m_freq)*k0) # In h/Mpc

        if plot:    
            fig = plt.figure(figsize=(50,25))
            if log:
                for i in range(Nfreqchunks):
                    plt.subplot(2,Nfreqchunks//2,i+1)
                    plt.imshow(ps_binned[i],origin='lower',aspect='auto',
                               interpolation='nearest', norm=LogNorm(), vmin=vmin, vmax=vmax,
                               extent=(self.alpha[0],self.alpha[-1],k_modes_unbinned[i][0],
                               k_modes_unbinned[i][-1]))
                    plt.xlabel(r'sin($\theta$)')
                    plt.ylabel('[h/Mpc]')
                    plt.title('Frequency Chunk {}'.format(i+1))
                    plt.colorbar()
            else:
                for i in range(Nfreqchunks):
                    plt.subplot(2,Nfreqchunks//2,i+1)
                    plt.imshow(ps_binned[i],origin='lower',aspect='auto',
                               interpolation='nearest', vmin=vmin, vmax=vmax,
                               extent=(self.alpha[0],self.alpha[-1],k_modes_unbinned[i][0],
                               k_modes_unbinned[i][-1]))
                    plt.xlabel(r'sin($\theta$)')
                    plt.ylabel('[h/Mpc]')
                    plt.title('Frequency Chunk {}'.format(i+1))
                    plt.colorbar()
            fig.subplots_adjust(wspace=0, hspace=0.1, top=0.95)
            plt.show()
        k_modes = [ks[:n_row_bins*m_freq].reshape((n_row_bins,-1)).mean(axis=1) for ks in k_modes_unbinned]
        alpha_binned = self.alpha[:m_alpha*n_col_bins].reshape((n_col_bins,-1)).mean(axis=1)
        return (ps_binned, k_modes, alpha_binned)

    def freq2distance(self, freq1, freq2=1420.4):
        '''
        Default for the second frequency is the 21 cm frequency, 1420.4 MHz.
        Convert from a pair of frequency to distance in Mpc/h bounded
        by this pair of frequencies. By default freq2 corresponds to z2=0.
        '''
        freq_21 = 1420.4
        z1 = freq_21/freq1 - 1
        z2 = freq_21/freq2 - 1
        distance1 = cosmo.comoving_distance(z=z1).value * cosmo.h
        distance2 = cosmo.comoving_distance(z=z2).value * cosmo.h
        return distance1-distance2

    def plot_rmap_ps_slice(self, rmap_ps_binned_no_error, rmap_ps_binned_with_error,
                           k_modes, alpha_binned,
                           alpha_idx_source, alpha_idx_no_source=[],
                           Nfreqchunks=4, plot=False, difference_ps_binned=None):
        '''
        Plot the power spectrum (of the specified chunk) returned by
        get_rmap_ps for a specific alpha.
        '''
        #fig = plt.figure(figsize=(50,12))
        fig = plt.figure(figsize=(15,15))
        gs = gridspec.GridSpec(4, Nfreqchunks//2, height_ratios=[4, 1, 4, 1])
        for i in range(Nfreqchunks):
            max_with_error = np.max(rmap_ps_binned_with_error[i])
            ncol = Nfreqchunks//2
            ax = plt.subplot(gs[i%ncol+(i//ncol)*(ncol*2)])
            modes = k_modes[i]
            m = self.alpha.shape[0]//alpha_binned.shape[0]
            for a in alpha_idx_source:
                alpha_idx_binned = a//m # Divide by the m argument of get_rmap_ps
                alpha = self.alpha[a]
                color = next(ax._get_lines.prop_cycler)['color']
                ax.loglog(modes, rmap_ps_binned_no_error[i][:,alpha_idx_binned]/max_with_error,
                           linestyle=':', color=color, label=fr'$\alpha$ = {alpha} (source, no noise)')
                ax.loglog(modes, rmap_ps_binned_with_error[i][:,alpha_idx_binned]/max_with_error,
                           linestyle='-', color=color, label=fr'$\alpha$ = {alpha} (source, with noise)')
            if not alpha_idx_no_source:
                alpha_idx_no_source.append(self.Npix_fft//2)
                alpha_idx_no_source.append(self.Npix_fft//2+10)
                for a in alpha_idx_source:
                    if a+10<=self.Npix_fft:
                        alpha_idx_no_source.append(a+5)
                    if a-10>=0:
                        alpha_idx_no_source.append(a-5)
                    if a+25<=self.Npix_fft:
                        alpha_idx_no_source.append(a+25)
                    if a-25>=0:
                        alpha_idx_no_source.append(a-25)
            for a in alpha_idx_no_source:
                alpha_idx_binned = a//m # Divide by the m argument of get_rmap_ps
                alpha = self.alpha[a]
                color = next(ax._get_lines.prop_cycler)['color']
                ax.loglog(modes, rmap_ps_binned_no_error[i][:,alpha_idx_binned]/max_with_error,
                           linestyle=':', color=color, label=fr'$\alpha$ = {alpha} (no noise)')
                ax.loglog(modes, rmap_ps_binned_with_error[i][:,alpha_idx_binned]/max_with_error,
                           linestyle='-', color=color, label=fr'$\alpha$ = {alpha} (with noise)')
            # Add line at 1e-6
            line = np.array([1e-6 for i in range(len(modes))])
            color = next(ax._get_lines.prop_cycler)['color']
            ax.loglog(modes, line, linestyle='-.', color=color)
            ax.set_xlabel('modes [h/Mpc]')
            ax.set_ylabel('power spectrum')
            ax.set_ylim(1e-11, 1)
            ax.set_title('frequency chunk {}'.format(i+1))
            # Plot the differences (errors - no errors)
            ax1 = plt.subplot(gs[i%ncol+(i//ncol)*(ncol*2)+ncol])
            ax1.loglog(modes, rmap_ps_binned_with_error[i][:,(self.Npix//2)//m]/max_with_error - rmap_ps_binned_no_error[i][:,(self.Npix//2)//m]/max_with_error,
                    color=next(ax._get_lines.prop_cycler)['color'], linestyle='--', label=r'(ps with noise - ps no noise) for $\alpha$ = 0')
            if difference_ps_binned is not None:
                max_diff = np.max(difference_ps_binned[i])
                ax1.loglog(modes, difference_ps_binned[i][:,(self.Npix//2)//m]/max_diff, color=next(ax._get_lines.prop_cycler)['color'], linestyle='--', label="ps of (rmap with noise - rmap no noise)\n" r"for $\alpha$ = 0")
            ax1.grid()
        fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.93, right=0.75)
        ax = plt.subplot(gs[ncol-1])
        ax1 = plt.subplot(gs[2*ncol-1])
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.suptitle('rmap power spectrum')
        if plot:
            plt.show()
        return fig

    def beam_no_interferometry(self, freq_MHz):
        '''
        Returns what the beam would be, ignoring the interferometry
        (so treating array as one huge dish, no interferometry).
        '''
        size = self.dish_locations[-1]
        lam = self.freq2lam(freq_MHz)
        fwhm = lam/size
        # For normal distribution, FWHM = 2sqrt(2ln2)*sigma
        sigma = fwhm/(2*np.sqrt(2*np.log(2)))
        x = np.arcsin(self.alpha)
        beam = norm.pdf(x, 0, sigma)
        return beam

    def get_uvplane_ps(self, uvplane, uvplane2 = None, Nfreqchunks=4, m_baselines=2, m_freq=2, padding=1, window_fn=np.blackman, plot=False, vmin=None, vmax=None, log=True):
        '''
        Get and plot the power spectrum for uvplane.
        For just one full plot of the power spectrum, set Nfreqchunks as 1,
        otherwise we divide the rmap into frequency chunks and
        compute the power spectrum independently for each chunk.
        After getting the power spectra, we bin in both x and y directions;
        m is how many of the existing bins we want to put in each bin.
        '''
        # Divide into frequency chunks
        # In each chunk, FT along the line of sight and square
        assert (uvplane.shape==(self.Nfreq,len(self.unique_baseline_lengths)))
        n = self.Nfreq//Nfreqchunks
        ps = []
        for i in range(Nfreqchunks):
            #ps_chunk = np.zeros((n+1,self.Npix))
            #for j in range(self.Npix):
            #    ps_chunk[:,j] = np.abs(rfft(np.hstack((rmap[i*n:(1+i)*n,j],np.zeros(n))))**2)
            ffts =[]
            for uvpl in [uvplane, uvplane2]:
                if uvpl is None:
                    continue
                if window_fn is not None:
                    tofft = uvpl[i*n:(1+i)*n,:]*(window_fn(n)[:,None])
                else:
                    tofft = uvpl[i*n:(1+i)*n,:]
                if padding > 0:
                    tofft = np.vstack((tofft,np.zeros((n*padding,self.unique_baseline_lengths.shape[0]))))
                ffts.append(rfft(tofft,axis=0))
            if len(ffts)==1:
                ps_chunk = np.abs(ffts[0]**2)
            elif len(ffts)==2:
                ps_chunk = np.real(ffts[0]*np.conj(ffts[1]))
            else:
                print ('something is wrong')
                stop()
            ps.append(ps_chunk)
        # After getting the power spectra, bin in both x and y directions
        n_rows = n*(1+padding)//2+1
        n_cols = self.unique_baseline_lengths.shape[0]
        n_row_bins = n_rows//m_freq
        n_col_bins = n_cols//m_baselines
        # Discard some values if necessary
        for i, ps_chunk in enumerate(ps):
            ps[i] = ps_chunk[:m_freq*n_row_bins,:m_baselines*n_col_bins]
        ps_binned = []
        # Bin
        for ps_chunk in ps:
            ps_chunk_binned = ps_chunk.reshape(n_row_bins, n_rows//n_row_bins, n_col_bins, n_cols//n_col_bins).sum(axis=3).sum(axis=1)
            ps_binned.append(ps_chunk_binned)
        
        # Convert from frequency to distance (Mpc/h)
        k_modes_unbinned = []
        last_modes = []
        for i in range(Nfreqchunks):
            freq_first = self.freqs[i*n]
            freq_last = self.freqs[(1+i)*n-1]
            # Get size of the chunk (dist_max) then the fundamental mode is 2*pi/dist_max
            dist_max = self.freq2distance(freq_first, freq_last)
            # Scale k0 by m_freq/(1+padding), where m_freq is the downsampling and 1+padding is the upsampling factor
            k0 = 2*np.pi/dist_max/(1+padding)
            #print (f"Fundamental mode for chunk {i} is {k0}")
            k_modes_unbinned.append(np.arange(n_row_bins*m_freq)*k0) # In h/Mpc

        if plot:    
            fig = plt.figure(figsize=(50,25))
            if log:
                for i in range(Nfreqchunks):
                    plt.subplot(2,Nfreqchunks//2,i+1)
                    plt.imshow(ps_binned[i],origin='lower',aspect='auto',
                               interpolation='nearest', norm=LogNorm(), vmin=vmin, vmax=vmax,
                               extent=(self.alpha[0],self.alpha[-1],k_modes_unbinned[i][0],
                               k_modes_unbinned[i][-1]))
                    plt.xlabel(r'sin($\theta$)')
                    plt.ylabel('[h/Mpc]')
                    plt.title('Frequency Chunk {}'.format(i+1))
                    plt.colorbar()
            else:
                for i in range(Nfreqchunks):
                    plt.subplot(2,Nfreqchunks//2,i+1)
                    plt.imshow(ps_binned[i],origin='lower',aspect='auto',
                               interpolation='nearest', vmin=vmin, vmax=vmax,
                               extent=(self.alpha[0],self.alpha[-1],k_modes_unbinned[i][0],
                               k_modes_unbinned[i][-1]))
                    plt.xlabel(r'sin($\theta$)')
                    plt.ylabel('[h/Mpc]')
                    plt.title('Frequency Chunk {}'.format(i+1))
                    plt.colorbar()
            fig.subplots_adjust(wspace=0, hspace=0.1, top=0.95)
            plt.show()
        k_modes = [ks[:n_row_bins*m_freq].reshape((n_row_bins,-1)).mean(axis=1) for ks in k_modes_unbinned]
        baselines_binned = self.unique_baseline_lengths[:m_baselines*n_col_bins].reshape((n_col_bins,-1)).mean(axis=1)
        return (ps_binned, k_modes, baselines_binned)

    def plot_uvplane_ps_slice(self, uvplane_ps_binned_no_error, uvplane_ps_binned_with_error,
                           k_modes, baselines_binned,
                           baselines=[],
                           Nfreqchunks=4, plot=False, difference_ps_binned=None):
        '''
        Plot the power spectrum (of the specified chunk) returned by
        get_uvplane_ps for a specific baseline.
        '''
        #fig = plt.figure(figsize=(50,12))
        fig = plt.figure(figsize=(15,15))
        gs = gridspec.GridSpec(4, Nfreqchunks//2, height_ratios=[4, 1, 4, 1])
        for i in range(Nfreqchunks):
            max_with_error = np.max(uvplane_ps_binned_with_error[i])
            ncol = Nfreqchunks//2
            ax = plt.subplot(gs[i%ncol+(i//ncol)*(ncol*2)])
            modes = k_modes[i]
            m = self.unique_baseline_lengths.shape[0]//baselines_binned.shape[0]
            if not baselines:
                baselines.append(0)
                baselines.append(self.unique_baseline_lengths.shape[0]//2)
                baselines.append(self.unique_baseline_lengths.shape[0]-1)
            for b in baselines:
                baseline_idx_binned = b//m # Divide by the m argument of get_uvplane_ps
                if baseline_idx_binned == len(uvplane_ps_binned_no_error[i][0,:]):
                    baseline_idx_binned -= 1
                bl = self.unique_baseline_lengths[b]
                color = next(ax._get_lines.prop_cycler)['color']
                ax.loglog(modes, uvplane_ps_binned_no_error[i][:,baseline_idx_binned]/max_with_error,
                           linestyle=':', color=color, label=f'baseline length = {bl} m (no noise)')
                ax.loglog(modes, uvplane_ps_binned_with_error[i][:,baseline_idx_binned]/max_with_error,
                           linestyle='-', color=color, label=f'baseline length = {bl} m (with noise)')
            # Add line at 1e-6
            line = np.array([1e-6 for i in range(len(modes))])
            color = next(ax._get_lines.prop_cycler)['color']
            ax.loglog(modes, line, linestyle='-.', color=color)
            ax.set_xlabel('modes [h/Mpc]')
            ax.set_ylabel('power spectrum')
            ax.set_ylim(1e-12, 1)
            ax.set_title('frequency chunk {}'.format(i+1))
            # Plot the differences (errors - no errors)
            ax1 = plt.subplot(gs[i%ncol+(i//ncol)*(ncol*2)+ncol])
            bl = self.unique_baseline_lengths[self.unique_baseline_lengths.shape[0]//2]
            ax1.loglog(modes,
                    uvplane_ps_binned_with_error[i][:,(self.unique_baseline_lengths.shape[0]//2)//m]/max_with_error - uvplane_ps_binned_no_error[i][:,(self.unique_baseline_lengths.shape[0]//2)//m]/max_with_error,
                    color=next(ax._get_lines.prop_cycler)['color'], linestyle='--', label=f"(ps with noise - ps no noise)\n" "for baseline = {bl} m")
            if difference_ps_binned is not None:
                max_diff = np.max(difference_ps_binned[i])
                ax1.loglog(modes, difference_ps_binned[i][:,(self.unique_baseline_lengths.shape[0]//2)//m]/max_diff, color=next(ax._get_lines.prop_cycler)['color'], linestyle='--',
                           label="ps of\n" "(uvplane with noise - uvplane no noise)\n" f"for baseline = {bl} m")
            ax1.grid()
        fig.subplots_adjust(wspace=0.2, hspace=0.3, top=0.93, right=0.75)
        ax = plt.subplot(gs[ncol-1])
        ax1 = plt.subplot(gs[2*ncol-1])
        ax.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        ax1.legend(bbox_to_anchor=(1.04,1), loc="upper left")
        plt.suptitle('uvplane power spectrum')
        if plot:
            plt.show()
        return fig
