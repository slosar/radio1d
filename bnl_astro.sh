for nd in 4 8 12 16 24 32 36 40 44 ; do for npix in 2048 4096; do for redundant in 1 0;  do echo $nd+$npix+$redundant; nohup wq sub -r "N:1; mode:bynode; group:new; job_name:$nd+$npix+$redundant" -c "source ~/.bashrc; python get_signals.py $nd $npix $redundant >$nd+$npix+$redundant.log 2> $nd+$npix+$redundant.err" &  done; done; done
