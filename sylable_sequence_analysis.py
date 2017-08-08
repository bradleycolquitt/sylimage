import sys as sys
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
from scipy import io
from scipy.io import wavfile
from scipy import stats
import os as os
from scipy import signal
from scipy import ndimage
from multiprocessing import Pool
import pickle
from hmmlearn.hmm import MultinomialHMM

import songtools
import local_settings
import evsonganaly


def convert_syllable_labels(song_corpus):
	data = {}
	data['conversion_dict'] = {}
	data['songs'] = np.ndarray(len(song_corpus), dtype = np.ndarray)
	for ksong,song in enumerate(song_corpus):
		data['songs'][ksong] = np.zeros(len(song), dtype = int)
		for kl,label in enumerate(song):
			if label in data['conversion_dict']: label
			else:
				data['conversion_dict'][label] = len(data['conversion_dict'])
			data['songs'][ksong][kl] = data['conversion_dict'][label]
	return data

def train_syllable_hmm(song_corpus, n_iterations = 50):
	hmm = MultinomialHMM(3)
	hmm.transmat_ = np.array([[0, 0, 1],
	 						  [1, 0, 0],
	 						  [0,.01,.99]])
	hmm.n_iter = n_iterations
	hmm.fit([np.concatenate(song_corpus)])
	return hmm

def split_up_motifs_with_hmm(song_corpus):
	translated_song_corpus = convert_syllable_labels(song_corpus)
	hmm = train_syllable_hmm(translated_song_corpus['songs'])

	motifs = []
	motif_song_idxs = []
	for ksong, song in enumerate(translated_song_corpus['songs']):
		data = hmm.decode(song)[1]
		start_idxs = np.arange(0,len(data))[data==0]
		stop_idxs = np.concatenate((start_idxs[1:].copy(), [len(data)]), axis = 0)

		#stop_idxs = np.concatenate(start_idxs[1:],np.array(len(data)))		
		for  k,start_idx in enumerate(start_idxs):
			stop_idx = stop_idxs[k]
			motifs.append(song_corpus[ksong][start_idx:stop_idx])
			motif_song_idxs.append(ksong)
		for m in motifs: print(m)

	# data = [len(m) for m in motifs]
	# data = filter(lambda x: x>2, data)
	return motifs, motif_song_idxs