import sys as sys
import os as os
import scipy as sc
from scipy import io
from scipy.io import wavfile
from scipy import ndimage
from scipy import signal
from matplotlib.mlab import specgram, psd
import numpy as np
import sklearn as skl
from sklearn import cluster
from sklearn import metrics
from scipy import spatial
from scipy.stats.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
from sklearn import mixture
from sklearn import decomposition
import random as rnd
from multiprocessing import Pool
import pdb

import songtools
import analyze_songs
import evsonganaly

def maxnorm(a):
    """normalizes a string by it's max"""
    a=a/max(a)
    return a

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    out=[]
    lens=(len(l)/n)
    for i in range(0, lens):
        out.append(l[i*n:i*n+n])
    return out

def CalculateSpectrum(syllable, fmin = 1e3, fmax = 12e3, n_fbins = 256, n_frames = 2, Fs = 32000, plot = False):
    nfft = int(np.floor(1 / (float(fmax)/(n_fbins*Fs) - float(fmin)/(n_fbins*Fs))))
    segstart=int(float(fmin)/(Fs/float(nfft)))
    segend=int(float(fmax)/(Fs/float(nfft)))
    n_samples = len(syllable)
    in_samples_per_frame = np.floor(n_samples/n_frames)
    f_samples_per_frame = (segend-segstart)
    out_vector =np.zeros(f_samples_per_frame*n_frames)
    for kframe in range(0,n_frames):
        spectrum = psd(syllable[kframe*in_samples_per_frame:(kframe+1)*in_samples_per_frame], NFFT = nfft, Fs=Fs)
        if np.max(spectrum[0]) > 0:
          spectrum = spectrum[0]/np.max(spectrum[0])
        out_vector[kframe*f_samples_per_frame:(1+kframe)*f_samples_per_frame] = np.squeeze(spectrum[segstart:segend])
    if plot:
        plt.plot(out_vector)
        plt.show()
    return out_vector

def EMofgmmcluster(array_of_syls, training_set_size = 20, model_selection_set_size = 1000, n_processors = 1, do_model_selection = False, n_models = 10, fs = 32000):
    """takes an array of segmented sylables and clusters them by 
    taking psds (welch method) and fitting a mixture model"""
    
    # set parames here (will eventaully go to kwargs)
    # set parameters here
    n_fbins = 100
    n_frames = 2
    fmin = 500
    fmax = 12e3


    PSDMAT = np.zeros((len(array_of_syls), n_frames*n_fbins))
    pool = Pool(processes = int(n_processors))
    results = []
    for k,syl in enumerate(array_of_syls):
        CalculateSpectrum(syl, n_fbins = n_fbins, n_frames = n_frames, Fs=fs) # for debugging/testing
        #print 'pool apply' + str(k) 
        results.append(pool.apply_async(CalculateSpectrum, (syl,), dict(n_fbins = n_fbins, n_frames = n_frames, fmin = fmin, fmax = fmax, Fs=fs)))
    for k,result in enumerate(results):
        #print 'pool get' + str(k)
        
        PSDMAT[k,:] = result.get()
        
    pool.close()
    pool.join()
    freq = None
    n_sylables = PSDMAT.shape[0]
    # calculate optimal number of models
    if do_model_selection:
        model_selection_data = select_models(PSDMAT, training_set_size = model_selection_set_size, n_processors = n_processors, n_models_min=4, n_models_max = n_models)
        optimal_k_models = model_selection_data['optimal_k']
    else:
        optimal_k_models = n_models
        model_selection_data = {'model_selection_mode': 'manually_set'}

    # calcualte training distmat with either whole data set or a subset of size training_set_size if n_sylables is greater
    if n_sylables > training_set_size:
        training_set_idxs = np.random.randint(0, n_sylables, training_set_size)
        d_training_set = sc.spatial.distance.squareform(sc.spatial.distance.pdist(PSDMAT[training_set_idxs,:],'euclidean'))**2
    else:
        d_training_set = sc.spatial.distance.squareform(sc.spatial.distance.pdist(PSDMAT,'euclidean'))**2
    # traing the model on the similarity matrix of the training set
    #pdb.set_trace()
    s_training_set = 1-(d_training_set/np.max(d_training_set))

    gmm = mixture.GMM(n_components=optimal_k_models, n_iter=100000, covariance_type='diag')
    gmm.fit(s_training_set)
    # if there are additional data in the set, predict remaining based on training set
    if n_sylables > training_set_size:
          # predict training set
        training_set_labels=gmm.predict(s_training_set) 
        d_predict_set = np.transpose(sc.spatial.distance.cdist(PSDMAT[training_set_idxs,:], PSDMAT,'euclidean')**2)
        s_predict_set = 1-(d_predict_set/np.max(d_training_set))
        labels = gmm.predict(s_predict_set)
    else:
        labels = gmm.predict(s_training_set)

    return model_selection_data, labels, PSDMAT, freq

def select_models(PSDMAT, n_models_min = 2, n_models_max = 20, training_set_size = 500, selection_type = 'pval', k=10, reps = 10, n_processors = 1, plot = True, do_print = True, do_pca = False):
    """select the optimal number of models """
    dataout = {'model_selection_mode': selection_type}
    # setup computational variables
    range_of_models = range(n_models_min, n_models_max)
    n_sylables_total = PSDMAT.shape[0]
    # initate pool
    pool = Pool(processes = int(n_processors))

    # iterate thru number of models to make calculations 
    results = []
    for n_models in range_of_models:
        res = pool.apply_async(test_with_n_models, (PSDMAT,), dict(n_models = n_models, training_set_size = training_set_size, k = k, reps = reps, do_pca = do_pca))
        results.append((res, n_models))
    # preallocate recording arrays and itterate thru results
    dataout['loglikelihood']= []
    dataout['bic'] = []
    likelihood_dist=[]
    for kresult, result in enumerate(results):
        if do_print:
            print('retreiving test with ' + str(range_of_models[kresult]) + ' models')
        data = result[0].get()
        dataout['loglikelihood'].append(data['loglikelihood'])
        dataout['bic'].append(data['bic'])
        likelihood_dist.append(data['likelihood_dist'])
    pool.close()
    pool.join()

    # process the various likelihood scores
    dataout['loglikelihood'] = np.array(dataout['loglikelihood'])
    dataout['bic'] = np.array(dataout['bic'])
    dataout['range_of_models'] = range_of_models
    dataout['pvals']=np.array([sc.stats.mannwhitneyu(likelihood_dist[x],likelihood_dist[x+1])[1] for x in range(len(likelihood_dist)-1)])
    
    ## calcualte optimal k in various ways
    # calculate optimal k based on liklihood
    max_idx = np.argmax(dataout['loglikelihood'])
    dataout['optimal_k_likelihood'] = range_of_models[max_idx]
    # calculate optimal k based on lack of significant improment of model
    idxs = np.arange(0,len(dataout['pvals']), dtype = int)
    first_idx = idxs[dataout['pvals']>0.005][0]
    dataout['optimal_k_pval'] = range_of_models[first_idx]
    # calculate optimal k based on bic
    min_idx = np.argmin(dataout['bic'])
    dataout['optimal_k_bic'] = range_of_models[min_idx]

    # calculate optimal k based on liklihood
    if selection_type == 'bic':
        dataout['optimal_k'] = dataout['optimal_k_bic']
    elif selection_type == 'likelihood':
        dataout['optimal_k'] = dataout['optimal_k_likelihood']
    elif selection_type == 'pval':
        dataout['optimal_k'] = dataout['optimal_k_pval']

    if plot:
        plot_model_selection_data(dataout)

    return dataout

def test_with_n_models(PSDMAT, n_models=1, training_set_size = 500, k = 10, reps = 5, do_pca = False, n_pca_components = 30):
    """This function is the nested function to test training sets from PSDMAT for the optimal number of models.  This tests 
    one number at a time to be put in a loop in select_models function above"""
    # make nessiary caluclations and preallocate
    n_sylables_total = PSDMAT.shape[0]
    dataout = {}
    liklihood_values = []
    bic_values = []
    # repeat caculations reps times
    for m in range(reps):
        training_set_idxs = np.random.randint(0, n_sylables_total, training_set_size)
        d_training_set = sc.spatial.distance.squareform(sc.spatial.distance.pdist(PSDMAT[training_set_idxs,:],'euclidean'))**2
        s = 1-(d_training_set/np.max(d_training_set))
        s_idxs = np.arange(0,s.shape[0])
        #rnd.shuffle(s_idxs) # currently no need for a double suffle (see right above)
        chunks_of_s_idxs = np.array(chunks(s_idxs,len(s)/k))
        for kset,testset_idxs in enumerate(chunks_of_s_idxs):
            chunk_idxs = range(0,len(chunks_of_s_idxs))
            chunk_idxs.remove(kset)
            trainset_idxs = np.concatenate(chunks_of_s_idxs[chunk_idxs]) 
            if do_pca: 
                pca = decomposition.PCA(n_components = n_pca_components)
                x = pca.fit_transform(s)   
            else:        
                x = s
            gmm = mixture.GMM(n_components=n_models, n_iter=100000, covariance_type='diag')
            gmm.fit(x[trainset_idxs,:])
            bic_values.append(gmm.bic(x[testset_idxs]))
            liklihood_values.append(sum(gmm.score(x[testset_idxs])))
    # populate dataout with stats
    dataout['loglikelihood'] = np.mean(liklihood_values)
    dataout['likelihood_dist'] = liklihood_values
    dataout['bic'] = np.mean(bic_values)
    dataout['bic_dist'] = bic_values
    return dataout

def plot_model_selection_data(data, showfig = True, savefig = False, fname = ''):
    print("Model Selection Info: ")
    print("optimal_k based on likelihood max: " + str(data['optimal_k_likelihood']))
    print("optimal_k based on first non-significant improvment: " + str(data['optimal_k_pval']))
    print("optimal_k based on bic minimum: " + str(data['optimal_k_bic']))
    print("optimal_k set to: " + str(data['optimal_k']))
    fig = plt.figure()
    ax = fig.add_subplot(1,3,1)
    plt.plot(data['range_of_models'], data['loglikelihood'])
    plt.xlabel('# of models')
    plt.ylabel('log liklihood')
    #plt.ylim([25e3, 50e3])
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # plt.plot(data['range_of_models'][:], np.diff(data['loglikelihood']))
    # plt.xlabel('# of models')
    # plt.ylabel('d(log liklihood)')

    # fig = plt.figure()
    ax = fig.add_subplot(1, 3, 2)
    plt.plot(data['range_of_models'][:-1], data['pvals'])
    plt.xlabel('# of models')
    plt.ylabel('p(improvment over last)')
    plt.ylim([0., 0.5])
    # fig = plt.figure()
    # ax = fig.add_subplot(1,1,1)
    # plt.hist(likelihood_dist,bins=30,histtype='step')

    # fig = plt.figure()
    ax = fig.add_subplot(1, 3, 3)
    plt.plot(data['range_of_models'], data['bic'])
    plt.xlabel('# of models')
    plt.ylabel('bic')
    #plt.ylim([-60e3, -10e3])

    if showfig:
        plt.show()

    if savefig:
        plt.savefig(fname)

def eucliddist(a,b):
    """calculates a euclidean distance between two equal length arrays"""
    return np.sqrt(sum((np.array(a)-np.array(b))**2))

def minkowskidist(a,b,p):
    """calculates a minkowski distance between two equal length arrays"""
    return sum((np.array(a)-np.array(b))**p)**(1/float(p))

def sqr_eucliddist(a,b):
    """calculates a euclidean distance between two equal length arrays"""
    return (sum((np.array(a)-np.array(b))**2))

def mahalanobisdist(a,b):
    """calculates the mahalanobis distance between tow equal lenght vectors"""
    a=np.array(a)
    b=np.array(b)
    s=np.cov(np.array([a,b]).T)
    try: sinv=np.linalg.inv(s)
    except  np.linalg.LinAlgError: sinv=np.linalg.pinv(s)
    return sc.spatial.distance.mahalanobis(a,b,sinv)

def mahaldist(a,b):
    a=np.array(a)
    b=np.array(b)
    s=np.cov(np.array([a,b]).T)
    xminy=np.array(a-b)
    try: sinv=np.linalg.inv(s)
    except  np.linalg.LinAlgError: sinv=np.linalg.pinv(s)
    m=abs(np.dot(np.dot(xminy,sinv),xminy))
    return np.sqrt(m)

def pearsonrcoeff(a,b):
    return pearsonr(a,b)[0]

def spearmanrcoeff(a,b):
    return spearmanr(a,b)[0]

def sqformdistmat(array):
    """creates a squareform distance matrix for an array.  Currently only
    uses euclidean dist"""
    out = np.zeros((len(array),len(array)))
    for kx,x in enumerate(array):
        for ky,y in enumerate(array):
            out[ky,kx] = sqr_eucliddist(x,y)
    return out

def norm(a):
    """normalizes an array by it's average and sd"""
    a=(np.array(a)-np.average(a))/np.std(a)
    return a

def sigmoid_norm(x):
    """returns a sigmoid nomalization by the sigmoid function"""
    return 1 / (1 + np.exp(-np.asarray(x)))

def plot_clusters(syllables, PSDMAT, labels):
    PSDMAT = PSDMAT / np.max(PSDMAT)
    syllables = np.array(syllables)
    labels_num = np.array(labels)#np.zeros(len(labels))
    count_dict = {}
    for k,label in enumerate(labels):
        count_dict[label] = None
        # import ipdb; ipdb.set_trace(); 
        # labels_num[k] = analyze_songs.letter_labels[letter]
    n_sylables = len(count_dict)
    for ksyl in range(0,n_sylables):
        print('Syllable ' + str(ksyl))
        print('count = ' + str(np.sum(labels_num==ksyl)) + " of " + str(PSDMAT.shape[0]))
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1) 
        plt.plot(np.mean(PSDMAT[labels_num == ksyl, :], axis = 0),'k')
        plt.plot(np.mean(PSDMAT[labels_num == ksyl, :], axis = 0) + np.std(PSDMAT[labels_num == ksyl, :], axis = 0),'r')
        plt.plot(np.mean(PSDMAT[labels_num == ksyl, :], axis = 0) - np.std(PSDMAT[labels_num == ksyl, :], axis = 0),'r')
        # plt.savefig('/home/jknowles/data/brainard_lab/figures/cluster_' + str(ksyl) + '_spectrum.png')
        fig = plt.figure()
        nplot = 20
        for kinstance, wf in enumerate(syllables[labels_num == ksyl][0:nplot]):
            ax = fig.add_subplot(1,nplot,1+kinstance)
            plt.specgram(wf,Fs=32000)
            for xlabel_i in ax.get_xticklabels():
                xlabel_i.set_visible(False)
                xlabel_i.set_fontsize(0.0)
            for xlabel_i in ax.axes.get_yticklabels():
                xlabel_i.set_fontsize(0.0)
                xlabel_i.set_visible(False)
        # plt.savefig('/home/jknowles/data/brainard_lab/figures/cluster_' + str(ksyl) + '_specgrams.png')
    plt.show()

def compare_labels(dir1, dir2, use_batch_file = False, batch_file = None):
    """ This function compares two sets of syllable labels for identically segmented syllables.
    The number of files (songs) and the number of syllables must be identical between the sets.  This
    is a (cheap) way to insure (weakly) that there are no misallignments.  

    syllable labels from dir2 files are compared to syllable 1 labels.  
    """

    if use_batch_file:
        f1 = open(batch_file)
        dir1_files = [line.rstrip() for line in f1 if len(line.rstrip())>0]
        dir2_files = dir1_files
        f1.close() 
    else:
        # aggregate all files from dir1
        dir1_files = song_files=os.listdir(dir1)
        dir1_files = filter(lambda x: x[-8:]=='.not.mat', dir1_files)
        # aggregare all files from dir2
        dir2_files = song_files=os.listdir(dir2)
        dir2_files = filter(lambda x: x[-8:]=='.not.mat', dir2_files)

    if dir1_files != dir2_files:
        dir2_files = filter(lambda x: x in dir1_files, dir2_files)
        dir1_files = filter(lambda x: x in dir2_files, dir1_files)
        if dir1_files != dir2_files:
            raise Exception('Unequal number of .not.mat files')

    labels1 = []
    for dfile in dir1_files:
        data = evsonganaly.load_ev_file(dir1+dfile)
        labels1.extend(data['labels'][0])
    labels2 = []
    for dfile in dir2_files:
        data = evsonganaly.load_ev_file(dir2+dfile)
        labels2.extend(data['labels'][0])
    if len(labels1) != len(labels2):
        raise Exception('Unequal number of syllables - segmentation must differ between sets of files.')

    nlabels1 = np.zeros(len(labels1))
    ltndict1 = {}
    for k,lab in enumerate(labels1):
        if lab not in ltndict1: ltndict1[lab] = len(ltndict1)
        nlabels1[k] = ltndict1[lab]
    nlabels2 = np.zeros(len(labels2))
    ltndict2 = {}
    for k,lab in enumerate(labels2):
        if lab not in ltndict2: ltndict2[lab] = len(ltndict2)
        nlabels2[k] = ltndict2[lab]
    ntldict1 = {v:k for k, v in ltndict1.items()}
    ntldict2 = {v:k for k, v in ltndict2.items()}

    label1_count = int(max(nlabels1))
    label2_count = int(max(nlabels2))
    bins1 = np.arange(0,label1_count+1)
    bins2 = np.arange(0,label2_count+1)
    set1_syllable_info = []
    for ksyl in range(0,label1_count):
        syldict = {'set1_lable': ntldict1[ksyl]}
        idxs = nlabels1==ksyl
        plt.figure()
        HIST = plt.hist(nlabels2[idxs], bins2); 
        ranked_labels = bins2[np.flipud(np.argsort(HIST[0]))] 
        ranked_values = HIST[0][np.flipud(np.argsort(HIST[0]))]
        print('dir1 cluster ' + str(ksyl))
        for k in range(0,len(ranked_labels)):
            if ranked_values[k] > 0:
                print(str(k) + " most frequent dir2 assignment is " + str(ranked_labels[k]) + " with " + str(ranked_values[k]) + " (" + str(np.round(100*float(ranked_values[k])/sum(ranked_values))) + " %)")
        
    plt.show()


#main program
if __name__=='__main__':
    path1 = "/data/brainard_lab/testsongs/"
    path2 = "/data/brainard_lab/testsongs/autodata/"
  # path1 = "/data/doupe_lab/mimi/042511/"
  # path2 = "/data/doupe_lab/mimi/042511/autodata/"
    compare_labels(path1, path2, use_batch_file = False, batch_file = '/data/doupe_lab/mimi/042511/batch_alone')
  # plot_model_selection_data(sc.io.loadmat('/data/doupe_lab/mimi/041511/autodata/batch_alonemodel_selection_run.mat'), showfig = False, savefig = True, fname = '/Users/jeffknowles/Desktop/mimi_figs/041511_alone.png')
  # plot_model_selection_data(sc.io.loadmat('/data/doupe_lab/mimi/041511/autodata/batch_dirmodel_selection_run.mat'), showfig = False, savefig = True, fname = '/Users/jeffknowles/Desktop/mimi_figs/041511_dir.png')
  # plot_model_selection_data(sc.io.loadmat('/data/doupe_lab/mimi/042211/autodata/batch_t1-477_alonemodel_selection_run.mat'), showfig = False, savefig = True, fname = '/Users/jeffknowles/Desktop/mimi_figs/042211_alone.png')
  # plot_model_selection_data(sc.io.loadmat('/data/doupe_lab/mimi/042211/autodata/batch_t1-477_dirmodel_selection_run.mat'), showfig = False, savefig = True, fname = '/Users/jeffknowles/Desktop/mimi_figs/042211_dir.png')
  # plot_model_selection_data(sc.io.loadmat('/data/doupe_lab/mimi/042511/autodata/batch_alonemodel_selection_run.mat'), showfig = False, savefig = True, fname = '/Users/jeffknowles/Desktop/mimi_figs/042511_alone.png')
  # plot_model_selection_data(sc.io.loadmat('/data/doupe_lab/mimi/042511/autodata/batch_dirmodel_selection_run.mat'), showfig = False, savefig = True, fname = '/Users/jeffknowles/Desktop/mimi_figs/042511_dir.png')

#
