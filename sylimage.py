# -*- coding: utf-8 -*-
"""

"""

import numpy as np
import scipy as sp
from multiprocessing import Pool

import songtools
import sylable_cluster_tools as cluster
import evsonganaly
import sylable_sequence_analysis as sequence

import pdb


def analyze_and_label_songs(song_files, 
                            run_name = '', 
                            plot = False, 
                            n_processors = 1, 
                            use_evsonganaly = False,
                            use_autodata_dir = True, 
                            xcorr_allign = False, 
                            filetype = 'wav', 
                            do_model_selection = False, 
                            n_models=10):
    # map songs to pool to gather song data
    pool = Pool(processes = int(n_processors))
    results = []
    for ksong, song_file in enumerate(song_files):
        results.append(pool.apply_async(analyze_song, (song_file,), dict(use_evsonganaly = use_evsonganaly, filetype = filetype)))
        #song, syls ,song_data = analyze_song(song_file, use_evsonganaly = use_evsonganaly, filetype = filetype) # this is here for testing without pool
       #pdb.set_trace()
       
    # gather song data from results
    sylables = []
    sylable_ids = []
    song_data_rec = []
    song_files = song_files
    for ksong, result in enumerate(results):
#        pdb.set_trace()
        song, syls, song_data = result.get()
        sylables.extend(syls)
        sylable_ids.extend([ksong]*len(syls))
        song_data_rec.append(song_data)
    sylable_ids = np.array(sylable_ids)
    fs = song[1]
    if xcorr_allign:
        sylables = songtools.allign_by_xcorr(sylables)
       
    return(sylables, song)
    
def analyze_song(song_file, use_evsonganaly = False, filetype = 'wav'):
    if filetype=='wav' or 'WAV':
        song = songtools.impwav(song_file)
        #pdb.set_trace()
    elif filetype=='cbin':
        song = songtools.impcbin(song_file)
    elif filetype=='raw':
        song = songtools.impraw(song_file)
    elif filetype == 'int16':
        song = songtools.impmouseraw(song_file)
    else:
        raise Exception('Filetype ' + filetype + ' not supported')
    if use_evsonganaly:
        song_data = evsonganaly.load_ev_file(song_file, load_song = False)
        song_data['a'] = song
        syls, labels = evsonganaly.get_ev_sylables(song_data)
    else:
        sm_win = 2
        min_dur = 0.015
        song_data = {}
        song_data['threshold']=0 # threshold in units of std
        syls, times = songtools.getsyls(song,
                                        min_length = min_dur,
                                        window = sm_win,
                                        threshold = song_data['threshold'])
        song_data['Fs'] = float(song[1])
        song_data['fname'] = song_file
        song_data['onsets'] = np.array([syl[0] for syl in times])*1e3
        song_data['offsets'] = np.array([syl[1] for syl in times])*1e3
        song_data['sm_win'] = sm_win
        song_data['min_dur'] = min_dur * 1e3
        song_data['min_int'] = 0

    return song, syls, song_data 

letter_labels = np.array(['a','b','c','d','e','f','g','h','i','j','k','l','m', 
                    'n','o','p','q','r','s','t','u','v','w','x','y','z','-'])
letter_to_num_dict = {}
for kl,letter in enumerate(letter_labels):
    letter_to_num_dict[letter] = kl
    
    
if __name__=="__main__":
    import argparse, os
    parser=argparse.ArgumentParser()
    parser.add_argument('path')
    parser.add_argument('-t', '--type',help='type of data collection.  dir, nest_dir or batch_file')
    parser.add_argument('-f','--file-type',help='filetype (wav)')
    parser.add_argument('-p','--n-processors',type=int)
    parser.add_argument('-n','--n-models',type=int,help='number of syllable types to cluster.  If --model-selection=1, search up to n_models')
    #parser.add_argument('--model-selection',type=int)
    parser.add_argument('--model-selection', action='store_true')
    parser.add_argument('--use-evsonganaly',help='use segmentation data from evsonganly .mat files.')
    args = vars(parser.parse_args())
    # path=sys.argv[1]
    path = args['path']
    if not os.path.exists(path):
        raise Exception('No directory or file at ' + path)
    # set other arguments to defualts
    defaults = {}
    defaults['type']='dir'
    defaults['file_type']='wav'
    defaults['n_processors']=1
    defaults['use_evsonganaly']=False
    defaults['model_selection']=False
    defaults['n_models']=10
    for key in args.keys():
        if args[key] is not None:
            defaults[key]=args[key]
    params=defaults
    filetype=params['file_type'].lower()
    use_evsonganaly=params['use_evsonganaly']
    n_processors = params['n_processors']
    if params['type'] == 'dir':
        song_files=[path+x for x in os.listdir(path)]
        song_files = np.array(filter(lambda x: x[-(len(filetype)+1):].lower()=='.'+ filetype.lower(), song_files))
        print(song_files)
        sylables, PSDMAT, labels = analyze_and_label_songs(song_files, n_processors = n_processors, filetype = filetype, use_evsonganaly = use_evsonganaly, do_model_selection=params['model_selection'], n_models=params['n_models'])
        cluster.plot_clusters(sylables,PSDMAT,labels)
    elif params['type'] == 'nest_dir':
        analyze_birds_in_nest_direc(direc, n_processors = n_processors, filetype = filetype)
    elif params['type'] == 'batch_file':
        song_files = []
        f = open(path)
        dirpath = path[:path.rfind('/')+1]
        batch_fname = path[path.rfind('/'):]
        for line in f:
            if len(line.rstrip())>0:
                song_files.append(dirpath + line.rstrip())
        print(song_files)
        sylables, PSDMAT, labels = analyze_and_label_songs(song_files, run_name = batch_fname, n_processors = n_processors, use_evsonganaly = use_evsonganaly, filetype = filetype, do_model_selection=params['model_selection'], n_models=params['n_models'])
        cluster.compare_labels(dirpath, dirpath+'autodata/', use_batch_file = True, batch_file = dirpath+batch_fname)
        cluster.plot_clusters(sylables,PSDMAT,labels)
    else:
        raise Exception('no directory structure settings for ' + analysis_type)