#!/usr/bin/env python
import sys
import numpy as np
import telescope_1d
import os
ndishes = int(sys.argv[1])
npix = int(sys.argv[2])
redundant = bool(sys.argv[3]=="1")
redstr = 'red' if redundant else 'nred'
print ("Creating telescope...")
Nfreq = 512
t = telescope_1d.Telescope1D(Ndishes=ndishes, Npix_fft=npix, redundant=redundant, Nfreq=Nfreq, seed=22)


for seed in range(30):
    print (f"Doing seed: {seed}\n--------------")
    for sigt in 'sig point gauss unif'.split():
        if sigt == 'sig':
            sig = t.get_signal(seed=seed)
        elif sigt == 'point':
            sig =  t.get_point_source_sky(seed=seed)
        elif sigt =='gauss':
            sig = t.get_gaussian_sky(seed=seed)
        elif sigt =='unif':
            sig = t.get_uniform_sky(seed=seed)
        else:
            print ("Shit!")
            stop
        uvsig = t.observe_image(sig)
        for correlated in "uc":
            for errortype in "ta":
                for te in [0,0.05,0.1,1.,10.,100.]:
                    outfname = f"out/{ndishes}_{npix}_{redstr}_{seed}_{sigt}_{correlated}{errortype}{te}.npy"
                    error_sigma = te * (1e-12 if errortype=="t" else 1e-2)    
                    print (f"Working on {outfname}", sig.sum())
                    uvplane, uvplane_f, uvplane_1 = t.get_obs_uvplane(uvsig, error_sigma=error_sigma, 
                                    correlated=(correlated=="c"), time_error = (errortype=="t") ,filter_FG=True)

                    np.save(outfname,(uvplane, uvplane_f, uvplane_1))




