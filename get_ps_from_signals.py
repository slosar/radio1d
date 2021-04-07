#!/usr/bin/env python
import numpy as np
import glob
import telescope_1d

flist = glob.glob ('out/*_*_*_0_*_*.npy')
#flist = glob.glob ('out/16_4096_*_0_*_*.npy')
#flist += glob.glob ('out/20_4096_*_0_*_*.npy')

for fname in flist:
    fname = fname.replace('.npy','').replace('out/','').split('_')
    ndishes = int(fname[0])
    npix = int (fname[1])
    redstr = fname[2]
    redundant = (redstr == 'red')
    sigt = fname[4]
    te = fname[5]
    #if te == 0.0:
    #    te = int(te)
    newglobname = f"out/{ndishes}_{npix}_{redstr}_*_{sigt}_{te}.npy"
    outname = f"outp/{ndishes}_{npix}_{redstr}_{sigt}_{te}"
    sublist = glob.glob(newglobname)
    print (outname, len(sublist), redundant)
    redundant == redstr=='red'
    if len(sublist)==30:
        t = None
        for Nfreqchicks in [1,2,4]:
            lps = []
            lpsf = []
            lpsfx = []
            lpsf1 = []
            lpsf1x = []

            for fname in sublist:
                print(fname)
                uvplane, uvplane_f, uvplane_f1 = np.load(fname)
                if t is None:
                    t = telescope_1d.Telescope1D(Ndishes=ndishes, Npix_fft=npix, Nfreq=512, redundant=redundant, seed=22)
                ps, k_modes, baselines_binned = t.get_uvplane_ps(uvplane, Nfreqchunks=Nfreqchicks, m_baselines=1, m_freq=1, padding=1, window_fn=np.blackman)
                psf, k_modes, baselines_binned = t.get_uvplane_ps(uvplane_f, Nfreqchunks=Nfreqchicks, m_baselines=1, m_freq=1, padding=1, window_fn=np.blackman)
                psfx, k_modes, baselines_binned = t.get_uvplane_ps(uvplane_f, uvplane, Nfreqchunks=Nfreqchicks, m_baselines=1, m_freq=1, padding=1, window_fn=np.blackman)
                psf1, k_modes, baselines_binned = t.get_uvplane_ps(uvplane_f1, Nfreqchunks=Nfreqchicks, m_baselines=1, m_freq=1, padding=1, window_fn=np.blackman)
                psf1x, k_modes, baselines_binned = t.get_uvplane_ps(uvplane_f, uvplane,  Nfreqchunks=Nfreqchicks, m_baselines=1, m_freq=1, padding=1, window_fn=np.blackman)

                lps.append(ps)
                lpsf.append(psf)
                lpsfx.append(psfx)
                lpsf1.append(psf1)
                lpsf1x.append(psf1)

            lps = np.array(lps).mean(axis=0)
            lpsf = np.array(lpsf).mean(axis=0)
            lpsfx = np.array(lpsfx).mean(axis=0)
            lpsf1 = np.array(lpsf1).mean(axis=0)
            lpsf1x = np.array(lpsf1x).mean(axis=0)
            np.save(outname+f'_{Nfreqchicks}.npy',(lps,lpsf,lpsfx,lpsf1,lpsf1x))
            np.save(outname+f'_{Nfreqchicks}_kmodes.npy',k_modes)
            np.save(outname+f'_{Nfreqchicks}_baselines.npy',baselines_binned)

            

