import sylimage as sy
import matplotlib.mlab as mlab

song_files = ["/mnt/bengal_home/song/wh27pk57/2017-08-04/output_1501877470.wav"]
n_processors = 1
filetype = "wav"
use_evsonganaly = True
params = {'model_selection':True, 'n_models':9}

sylables, song = sy.analyze_and_label_songs(song_files,
                                                                                 n_processors = n_processors, 
                                                                                 filetype = filetype, 
                                                                                 use_evsonganaly = use_evsonganaly, 
                                                                                 do_model_selection=params['model_selection'], 
                                                                                 n_models=params['n_models'])

a = (song[0].flatten())
Fs = song[1]
px, faxis, t = mlab.specgram(a, Fs=Fs, NFFT=1024, noverlap = 1000, window = mlab.window_hanning)
