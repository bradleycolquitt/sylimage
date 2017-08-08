#from sys import *
import scipy as sc
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib import mlab
from scipy import io
from scipy.io import wavfile
from scipy import ndimage
from pylab import specgram, psd
from itertools import *
from scipy import signal
from array import array
import math
import pdb
import wavio

import sylable_cluster_tools as cluster
import evsonganaly


def impwav(a):
    """Imports a wave file as an array where a[1] 
    is the sampling frequency and a[0] is the data"""
    out=[]
    #pdb.set_trace()
    #wav = sc.io.wavfile.read(a)
    wav1 = wavio.read(a)
    wav = wav1.data
    rate = wav1.rate
    wf = wav - np.mean(wav)
    out=(wf, rate)
    #pdb.set_trace()
    return out

def impcbin(a):
    """Imports a cbin as an array where a[1] is the sampling freq and a[0] is the data"""
    file=open(a,'r')
    file=file.read()
    a=np.fromstring(file[1:-1],dtype=np.int16)
    out=[a,32000]
    return out

def impraw(a):
    """Imports a cbin as an array where a[1] is the sampling freq and a[0] is the data"""
    f = open(a,'r')
    data = f.read()
    data = np.fromstring(data[1:-1], dtype = np.int16)
    # data = np.divide(data,np.max(data))
    # plt.plot(data[100000:110000]); plt.show()
    out = (data, 40000)
    return out

def impmouseraw(a):
    """Imports a cbin as an array where a[1] is the sampling freq and a[0] is the data"""
    f = open(a,'r')
    data0 = f.read()
    data = np.fromstring(data0[8:], dtype = np.int16)
    data = np.divide(data.astype('float'),np.max(data))
    # plt.plot(data[0:15000]); plt.show()
    out = (data, 363636)
    return out

def getsyls(a, min_length = 0.01, window = 2, threshold = 1):
    """takes a file red in with impwav and returns a list of sylables"""
    fs = a[1]
    #pdb.set_trace()
    fa=filtersong(a)

    #fa = a;
    t = np.divide(np.arange(0,len(a[0])), float(fs))
    a=a[0]

    rect = smoothrect(fa, window, freq = fs)
    threshold = np.mean(rect) + np.std(rect) * threshold
    objs=findobject(rect, thresh = threshold)

    sylables=[x for x in [a[y] for y in objs[0]] if float(len(x))/fs > min_length]
    times=[x for x in [(t[y][0], t[y][-1]) for y in objs[0]] if (x[1]-x[0]) > min_length]
    return sylables, times

def outwave(filename,array):
    """Exports a numpy array (or just a regular python array) 
    as a wave file. It expects an array of the following format: (sample np.sqrt(sum((np.array(a)-np.array(b))**2))freq, data)"""
    sc.io.wavfile.write(filename,array[0],array[1])

def spec1d(a, Fs=32000):
    "plots a spectrogram of a"
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.specgram(a,Fs=Fs)
    #if a[1]==None: ax.specgram(a[0])
    #else: ax.specgram(a[0],Fs=a[1])
    plt.show()

def harmonic_spectrum(a, Fs = 32000):

    n_harmonics = 2

    spectrum, faxis = psd(a, Fs=Fs, NFFT=500)    
    f_idxs = np.arange(0,len(spectrum),1)
    f_idxs = f_idxs.astype(int)
    max_freq = np.floor(float(faxis[-1])/n_harmonics)
    max_idx = f_idxs[faxis>max_freq][0]-1
    f1_idxs = np.arange(0, max_idx, 1)
    f1_idxs = f1_idxs.astype(int)
    f1 = faxis[f1_idxs]

    f2_idxs = [ f_idxs[faxis >= 2*faxis[idx]][0]  for idx in f1_idxs]
    f2 = faxis[f2_idxs]
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    plt.plot(f1, spectrum[f1_idxs])
    plt.plot(f1, spectrum[f2_idxs])
    plt.show()

def harmonic_specgram(a, Fs = 32000):

    n_harmonics = 2

    fig = plt.figure()
    (px, faxis, t, im) = specgram(a, Fs=Fs, NFFT=512, noverlap = 500)   
    px = np.log10(px)
    f_idxs = np.arange(0,len(faxis),1)
    f_idxs = f_idxs.astype(int)
    max_freq = np.floor(float(faxis[-1])/n_harmonics)
    max_idx = f_idxs[faxis>max_freq][0]-1
    f1_idxs = np.arange(0, max_idx, 1)
    f1_idxs = f1_idxs.astype(int)
    f1 = faxis[f1_idxs]
    f2_idxs = [ f_idxs[faxis >= 2*faxis[idx]][0]  for idx in f1_idxs]
    f2 = faxis[f2_idxs]
    px_h = np.add(px[f1_idxs,:],px[f2_idxs,:])
    px_h[faxis<300,:] = 0
    ff = np.argmax(px_h, axis = 0)
    ff = f1[ff]
    fig = plt.figure()
    ax = fig.add_subplot(2,1,1)
    plt.pcolormesh(t, f1, px[f1_idxs,:])

    ax = fig.add_subplot(2,1,2)
    plt.pcolormesh(t, f1, px_h)
    plt.plot(t,ff,'k')

    plt.show()


#def ryth1d(a):
#    "plots a spectrogram of a.  4_13_11 I actually have no idea what I was tyring to do here."
#    fig=plt.figure()
#    ax=fig.add_subplot(111)np.sqrt(sum((np.array(a)-np.array(b))**2))
#    ax.specgram(a,NFFT=(len(a)/2),Fs=32000)
#    #if a[1]==None: ax.specgram(a[0])
#    #else: ax.specgram(a[0],Fs=a[1])
#    plt.show()

def plot1d(a):
    "plots a"
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.plot(a)
    plt.show()

def scat1d(a):
    "scatterplots a. Expects a to be a ziped string of the two values"
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.scatter(a)
    plt.show()

def hist(a,bins):
    "plots a histogram of a"
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(a,bins)
    plt.show()

def histlog(a,bins):
    "plots a histogram of a"
    fig=plt.figure()
    ax=fig.add_subplot(111)
    ax.hist(a,bins,log=True)
    plt.show()

def threshold(a,thresh=None):
    """Returns a thresholded array of the same length as input
    with everything below a specific threshold set to 0.
    By default threshold is sigma."""
    if thresh==None: thresh = 1
    #pdb.set_trace()
    #thresh = sc.mean(a) + sc.std(a) * thresh
    out=np.where(abs(a)>thresh,a,np.zeros(a.shape))
    return out

def rthreshold(a, thresh=None):
    """Returns a reverse thresholded array of the same length as input
    with everything above a specific threshold set to 0.
    By default threshold is sigma."""
    if thresh==None: thresh = sc.std(a)
    out=np.where(abs(a)>thresh,np.zeros(a.shape),a)
    return out

def mask(a,thresh=None):
    """Returns a masnp.sqrt(sum((np.array(a)-np.array(b))**2))k array of the same length as input
    with everything below a specific threshold set to 0 and
    everything above that threshold set to 1.
    By default threshold is sigma."""
    if thresh==None: thresh = 5*np.std(a)
    out=np.where(abs(a)>thresh,np.ones(a.shape),np.zeros(a.shape))
    return out

def weinent(a):
    "returns the weiner entropy of an array"
    a=np.array(a)
    log_a=np.log(a)
    out=((np.exp(log_a.mean()))/(a.mean()))
    return out

#def window(seq, n):
#    "Returns a string of windows (of width n). 4_13_11 getting rid of this because it is a sloppy solution to this problem."
#    it = iter(seq)
#    out = tuple(islice(it, n))
#    if len(out) == n:
#     return out

def mfreqz(b,a=1):
    """Plots the frequency and phase response of a filter.""" 
    w,h = sc.signal.freqz(b,a)
    fig=plt.figure()
    h_dB = 20 * np.log10 (abs(h))
    ax=fig.add_subplot(211)
    ax.plot(w/max(w),h_dB,'k')
    ax.set_ylim(-150, 5)
    ax.set_ylabel('Magnitude (db)')
    ax.set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    ax.set_title(r'Frequency response')
    ay=fig.add_subplot(212)
    h_Phase = np.unwrap(np.arctan2(np.imag(h),np.real(h)))
    ay.plot(w/max(w),h_Phase,'k')
    ay.set_ylabel('Phase (radians)')
    ay.set_xlabel(r'Normalized Frequency (x$\pi$rad/sample)')
    ay.set_title(r'Phase response')
    fig.subplots_adjust(hspace=0.5)
    plt.show()

def filtercall(a):
    b=sc.signal.iirdesign([0.5,0.7], [0.46,0.75], 1, 60, ftype='ellip')
    out=sc.signal.filtfilt(b[0],b[1],a)
    return (out)

def filtersong(a):
    #pdb.set_trace()
    out=[]
    dat = np.asarray(a[0])
    print(len(dat))
    print(dat)
    b=sc.signal.iirdesign(wp=0.04, ws=0.02, gpass=1, gstop=60,ftype='ellip')
    out.append(sc.signal.filtfilt(b[0],b[1],x=dat))
    out.append(a[1])
    return (out)

def smoothrect(a,window=None,freq=None):
    """smooths and rectifies a song.  Expects a file format (data,samplerate).
    If you don't enter a smoothing window size it will use 2ms as a default."""
    if freq== None: freq=32000
    if window == None: window=2
    if type(a) == tuple or len(a) == 2:
        a = a[0]
    le=int(round(float(freq*window)/1000))
    h=np.ones(le)/le
    smooth= np.convolve(h,abs(a))
    offset = round((len(smooth)-len(a))/2)
    smooth=smooth[(offset):(len(a)+offset)]
    return smooth

def smooth(a,window=None,freq=None):
    """smooths a song.  Expects a file format (data,samplerate).
    If you don't enter a smoothing window size it will use 2ms as a default."""
    if window == None: window=2
    if freq== None: freq=32000
    le=int(round(freq*window/1000))
    h=np.ones(le)/le
    smooth= np.convolve(h,a)
    offset = round((len(smooth)-len(a))/2)
    smooth=smooth[(1+offset):(len(a)+offset)]
    return smooth

def specplot(a):
    fig=plt.figure()
    ay=fig.add_subplot(111)
    #ay.specgram(a[0],Fs=a[1])
    ay.plot(smoothrect(filtersong(a),20), 'k', (mask(smoothrect(filtersong(a),20),2*(np.abs(np.median(a[0]))))),'b')
    plt.show()

def songseg(a, sigma=None):
    """returns a mask of an array where everything below 2sigma is zero and anything above 2
    sigma is a number 1 through N which is the number of the syllable.
    This is really helpful for segmenting data. This expects the song to have been imported using
    impwav."""
    if sigma==None: sigma=2
    label=sc.ndimage.label(mask(smoothrect(filtersong(a),20),sigma*(np.abs(np.median(a[0])))))
    return label

def songsegrange(a, cutoff=None,smwindow=None):
    """returns a mask of an array where everything below 2sigma is zero and anything above 2
    sigma is a number 1 through N which is the number of the syllable.
    This is really helpful for segmenting data. This expects the song to have been imported using
    impwav."""
    if smwindow==None: smwindow=20
    if cutoff==None: cutoff=2
    label=sc.ndimage.label(mask(smoothrect(filtersong(a)[0],smwindow),cutoff))
    return label

def sylablelen(a):
    """returns an array with the length of all sylables in wave file a.  Expects a
    to have been created using impwav.  At some point I should probably rewrite this to be faster."""
    lab=songseg(a)
    freq=a[1]
    sylno=lab[1]
    inc=1
    out=[]
    lst=list(lab[0])
    while inc<=sylno:
        len=lst.count(inc)
        out.append(len)
        inc=inc+1
    out=out/freq
    return out

def intersyllen(a):
    """returns an array with all intersylable lengths in wave file a.  Expects a
    to have been created using impwav."""
    msk=mask(smoothrect(filtersong(a),20),2*np.median(a[0]))
    freq=a[1]
    lst=list(msk)
    lst.reverse()
    endind=lst.index(1)
    lst.reverse()
    lst=lst[lst.index(1):(len(lst)-(endind))]
    lst=np.array(lst)
    lst=np.where(lst==0,np.ones(len(lst)),np.zeros(len(lst)))
    lst=sc.ndimage.label(lst)
    out=[]
    sylno=lst[1]
    inc=1
    lst=list(lst[0])
    while inc<=sylno:
        leng=lst.count(inc)
        out.append(leng)
        inc=inc+1
    out=np.float32(out)
    out=out/(int(freq)/1000)
    return out

def sylrate(a):
    """Returns the rate of sylables in a given song in sylables/sec"""
    lab=songseg(a)
    msk=mask(smoothrect(filtersong(a),20),2*np.median(a[0]))
    lst=list(msk)
    lst.reverse()
    endind=lst.index(1)
    lst.reverse()
    lst=lst[lst.index(1):(len(lst)-(endind))]
    songleng= np.float32(len(lst))/np.float32(a[1])
    out=lab[1]/songleng
    return out

def songleng(a):
    """Returns the song length in sylables"""
    lab=songseg(a)
    return lab[1]

def ffcalc(a, freq=None):
    """Returns the fundamental frequency of an array, a. Expects raw data,  the default frequency is 32,000. This uses brute force correlation which is slow for large data sets but more accurate than fft based methods. Returns the data in wavelength"""
    if freq==None: freq=32000
    corr=sc.correlate(a,a,mode='same')
    corr=corr[(len(corr)/2):(len(corr)-len(corr)/4)]
    dat=np.diff(np.where(np.diff(corr)>0,1,0))
    out=float(freq)/float(((list(dat)).index(-1)))
    return out

def ffcalcfft(a, freq=None):
    """Returns the fundamental frequency of a string, a. Expects raw data,  the default frequency is 
    32000. This method uses ffts."""
    if freq==None: freq=32000
    fft=sc.fftpack.fft(a)
    corr=sc.fftpack.ifft(fft*fft.conjugate())
    corr=corr[:(len(corr)/4)]
    dfff=np.diff(corr)
    dat=np.diff(np.where(dfff>0,1,0))
    if -1 not in list(dat): out=freq
    else:
        first=(((list(dat)).index(-1)))
        slope=(dfff[(first+1)]-dfff[first])
        out=(slope*first-dfff[first])/slope
    out=freq/out
    return out

def ffqualitymask(a, freq=None):
    """NOT DONE! Returns the fundamental frequency of a string, a. Expects raw data,  the default frequency is 
    32000. This method uses ffts."""
    if freq==None: freq=32000
    fft=sc.fftpack.fft(a)
    corr=sc.fftpack.ifft(fft*fft.conjugate())
    corr=corr[:(len(corr)/4)]
    dfff=np.diff(corr)
    dat=np.diff(np.where(dfff>0,1,0))
    if -1 not in list(dat): out=freq
    else:
        first=(((list(dat)).index(-1)))
        slope=(dfff[(first+1)]-dfff[first])
        out=(slope*first-dfff[first])/slope
    out=freq/out
    return out

def ffcalcfftqual(a, freq=None):
    """Returns the fundamental frequency and the amplitude of the autocorrelation at the first peak, of a string, a.
    Expects raw data,  the default frequency is 32000. This method uses ffts."""
    if freq==None: freq=32000
    fft=sc.fftpack.fft(a)
    corr=sc.fftpack.ifft(fft*fft.conjugate())
    #corr=corr[:(len(corr)/4)]
    dfff=np.diff(corr)
    dat=np.diff(np.where(dfff>0,1,0))
    if -1 not in list(dat): out=(float(freq),float(0))
    else:
        first=(((list(dat)).index(-1)))
        slope=(dfff[(first+1)]-dfff[first])
        out=(slope*first-dfff[first])/slope
        out=(freq/out,corr[first])
    return out

def ffcalcfftqual2(a, freq=None):
    """Returns the fundamental frequency of a string, and the amplitude of the correlation at the best peak (of the first 4).
    the default frequency is 32000. This method uses ffts."""
    if freq==None: freq=32000
    fft=sc.fftpack.fft(a)
    corr=sc.fftpack.ifft(fft*fft.conjugate())
    corr=corr[:(len(corr)/4)]
    lags = np.divide(freq, np.arange(0,len(corr), dtype = float))
    plt.plot(lags,corr); plt.show()
    import ipdb; ipdb.set_trace(); 
    dfff=np.diff(corr)
    dat=np.diff(np.where(dfff>0,1,0))
    if -1 not in list(dat): out=(float(freq),float(0))
    else:
        dat=np.where(dat<0,1,0)
        dat=list(dat) 
        (lab, inc)=sc.ndimage.label(dat)
        
        if inc >4: inc=4
        arr=[]
        while inc!=0:
                first=[]
                pos=((list(lab)).index(inc))-1
                first.append(pos)
                first.append(corr[pos])
                arr.append(first)
                inc=inc-1
        arr.sort(key=lambda x:x[1])
        first=arr[len(arr)-1][0]
        slope=(dfff[(first+1)]-dfff[first])
        out=(slope*first-dfff[first])/slope
        out=freq/out
        out=(out,(arr[len(arr)-1][1]))
    return out


def ffcalc_jk(a, fs = 32000, fmin = 0.1e3, fmax = 8e3):
    """Returns the fundamental frequency of a string, and the amplitude of the correlation at the best peak (of the first 4).
    the default frequency is 32000. This method uses ffts."""
    corr = np.correlate(a,a,'full')
    lags = np.arange(0,len(corr), dtype = float) - len(a) + 1
    lags[len(a)-1] = 0.1
    lags_f = np.divide(fs, lags)
    active_idxs = (lags_f > fmin) & (lags_f < fmax)
    peak_idxs = sc.signal.argrelmax(corr[active_idxs])[0]
    if len(peak_idxs) > 0:
        max_peak_idx = peak_idxs[np.argmax(corr[active_idxs][peak_idxs])]
        return (lags_f[active_idxs][max_peak_idx], corr[active_idxs][max_peak_idx])
    else: 
        return (0, 0)

def ffprofilequalmask(a, fs=None, window=None):
    """returns a string of local estimates of the fundamental frequency of a string.
    It also returns the amplitude of the highest peak in the autocorrelation as a quality measure.
    By default, freq=32000 and window=256."""
    if fs==None: fs=32000
    if window==None: window=256
    out = [ffcalcfftqual(a[x:x+(window-1)], fs) for x in range(len(a))]
    freq, amp =zip(*out)
    freq = np.real(freq)
    amp = np.real(amp)
    sigma=sc.std(amp)
    freq[amp<sigma] = float('nan')
    t = np.arange(0,len(freq)) / float(fs)
    return freq, t

def ffprofile_specgram(a, Fs = 32000, percent_boundry = 30, plot = False):
    # cacluclate spectrogram
    px, faxis, t = mlab.specgram(a, Fs=Fs, NFFT=1024, noverlap = 1000, window = mlab.window_hanning)
    # calculate overall fundamental frequency to stay within
    ave_ff_freq = ffcalc_jk(a)[0]
    ave_ff_freq = np.real(ave_ff_freq)
    #px = np.log10(px)
    f_idxs = np.arange(0,len(faxis),1)
    f_idxs = f_idxs.astype(int)  
    f_idxs = f_idxs[(faxis > ave_ff_freq*float(1-float(percent_boundry)/100)) & (faxis <  ave_ff_freq*float(1+float(percent_boundry)/100))]  
    max_idxs = np.argmax(px[f_idxs,:], axis = 0)
    max_idxs = f_idxs[max_idxs]
    f_contour = faxis[max_idxs]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.specgram(sylable, Fs=Fs)
        ax.plot(t, f_contour,'k')
        ax.plot([t[0], t[-1]], [ave_ff_freq, ave_ff_freq],'-b')
        ax.plot([t[0], t[-1]], [ave_ff_freq*float(1+float(percent_boundry)/100), ave_ff_freq*float(1+float(percent_boundry)/100)],'b')
        ax.plot([t[0], t[-1]], [ave_ff_freq*float(1-float(percent_boundry)/100), ave_ff_freq*float(1-float(percent_boundry)/100)],'b')
        plt.show()

    return  (t, f_contour)

def ffprofile_corr(a, Fs = 32000, window=512, energy_per_sample_thresh = 0.02, plot = False):
    """returns a string of local estimates of the fundamental frequency of a string correcting for harmonics
    By default, freq=32000 and window=256."""
    a = np.divide(a, np.max(np.abs(a)))
    ff_est_i = ffcalc_jk(a)[0]
    ff_est = float(ff_est_i)
    f_range = ff_est * 0.1
    f_contour = np.zeros(len(a))
    energy = np.zeros(len(a))
    for x in range(len(a)):
        data = ffcalc_jk(a[x:x+window-1], fmin = ff_est - f_range, fmax = ff_est + f_range)
        f_contour[x] = data[0]
        energy[x] = data[1] / window
        if energy[x] > energy_per_sample_thresh:
            ff_est = f_contour[x]
            f_range = 100
        else:
            ff_est = float(ff_est_i)
            f_range = ff_est * 0.1

    t0= np.divide(np.arange(0, len(f_contour), dtype = float), Fs)
    idxs = np.arange(0,len(f_contour))
    idx_i = idxs[energy>energy_per_sample_thresh][0]
    idx_t = idxs[energy>energy_per_sample_thresh][-1]
    f_contour = np.interp(t0[idx_i:idx_t], t0[energy>energy_per_sample_thresh], f_contour[energy>energy_per_sample_thresh])
    t = t0[idx_i:idx_t]

    if plot:
        fig = plt.figure()
        ax = fig.add_subplot(2,1,1)
        ax.specgram(a, Fs=Fs)
        ax.plot(t, f_contour,'k')
        plt.xlim([0, t[-1]])
        ax = fig.add_subplot(2,1,2)
        plt.plot(t0, energy)
        plt.xlim([0, t[-1]])
        # ax.plot([t[0], t[-1]], [ave_ff_freq, ave_ff_freq],'-b')
        # ax.plot([t[0], t[-1]], [ave_ff_freq*float(1+float(percent_boundry)/100), ave_ff_freq*float(1+float(percent_boundry)/100)],'b')
        # ax.plot([t[0], t[-1]], [ave_ff_freq*float(1-float(percent_boundry)/100), ave_ff_freq*float(1-float(percent_boundry)/100)],'b')
        plt.show()
    return (t, f_contour)



def ffprofilequal2(a, freq=None, window=None):
    """returns a string of local estimates of the fundamental frequency of a string correcting for harmonic errors.
    It also returns the amplitude of the highest peak in the autocorrelation as a quality measure.
    By default, freq=32000 and window=256."""
    if freq==None: freq=32000
    if window==None: window=256
    out = [ffcalcfftqual2(a[x:x+(window-1)],freq) for x in range(len(a))]
    return out

def ffprofile2(a, freq=None, window=None):
    """returns a string of local estimates of the fundamental frequency of a string correcting for harmonics
    By default, freq=32000 and window=256."""
    if freq==None: freq=32000
    if window==None: window=256
    out = [ffcalcfft2(a[x:x+(window-1)],freq) for x in range(len(a))]
    return out

def ffprofilefft(a, freq=None, window=None):
    """Returns a string of local estimates of the fundamental frequency of a string."""
    if freq==None: freq=32000
    if window==None: window=256
    out = [ffcalcfft(a[x:x+(window-1)],freq) for x in range(len(a))]
    return out

def ffacorr(a):
    """Returns the autocorrelation of a. Expects raw data"""
    z=np.zeros(2*len(a))
    z[:len(a)]=a
    fft=sc.fft(z)
    out=sc.ifft(fft*sc.conj(fft))
    return (out[:len(out)/2])

def norm(a):
    """normalizes a string by it's average"""
    a=(np.array(a)-np.average(a))/np.std(a)
    return a

def maxnorm(a):
    """normalizes a string by it's max"""
    a=a/max(a)
    return a

def norment(a):
    """returns an entropy calculation for an array normalized to between 0 and 1"""
    a=a/np.average(a)
    out=-100*sum(a*np.log(a))/np.log(len(a))
    return out

def normmaxent(a):
    """returns an entropy calculation for an array normalized to between 0 and 1"""
    a=a/max(a)
    out=-100*sum(a*np.log(a))/np.log(len(a))
    return out

def entropy(a):
    """retunrs an entropy calculation for an array"""
    out=-100*sum(a*np.log(a))/np.log(len(a))
    return out

def window(func,a, window=None):
    """produces a string which is the application of func on sliding window on string a."""
    if window==None: window=256
    out = [func(a[x:x+(window-1)]) for x in range(len(a))]
    return out

def songfinder(song):
    """identifies an imported wav file as a song.  Returns True if it is a song
    and False if it isn't.  Earliest version based on mean amplitude"""
    powerspecd=plt.psd(filtersong(song)[0])
    ent=np.mean((powerspecd[0][10:100]))
    return ent>100000

def eucliddist(a,b):
    """calculates a euclidean distance between two equal length arrays"""
    return np.sqrt(sum((np.array(a)-np.array(b))**2))

def sqformdistmat(array):
    """creates a squareform distance matrix for an array.  Currently only
    uses euclidean dist"""
    out=[]
    for x in array:
        intout=[]
        for y in array:
                intout.append(eucliddist(x,y))
        out.append(intout)
    return out

def findobject(file, thresh = 1):
    #pdb.set_trace()
    thresh=threshold(file,thresh=thresh)
    label=(sc.ndimage.label(thresh)[0])
    objs=sc.ndimage.find_objects(label)
    return(objs,thresh)

def psd_fft(a,fs=None):
    if fs==None:fs=32000
    n=float(len(a))
    ffta=sc.fft(a)
    ffta=ffta[1:n/2+1]
    ffta=np.array(ffta)
    psda=(1/(fs*n))*abs(ffta)
    psda[2:-1]=2*psda[2:-1]
    return(psda)

def psdrange_fft(a,fs=None,frange=None):
    if frange==None:frange=(650,10000)
    if fs==None:fs=32000
    n=float(len(a))
    ffta=sc.fft(a)
    ffta=ffta[1:n/2+1]
    ffta=np.array(ffta)
    psda=(1/(fs*n))*abs(ffta)
    psda[2:-1]=2*psda[2:-1]
    freq=np.arange(frange[0],frange[1],fs/n)
    psda=psda[frange[0]/(fs/n):frange[1]/(fs/n)]
    return psda, freq

def psd_lomb_scargle(a,fs=None,frange=None):
    if frange==None:frange=(650,10000)
    if fs==None:fs=32000
    n=float(len(a))
    x=np.linspace(0,1/32,n)
    f=np.linspace(500,10000,10)
    psd=sc.signal.lombscargle(x,a[0],f)

def pltpsd(a,fs=None,frange=None):
    if frange==None:frange=(650,10000)
    if fs==None:fs=32000
    n=float(len(a))
    ffta=sc.fft(a)
    ffta=ffta[1:n/2+1]
    ffta=np.array(ffta)
    psda=(1/(fs*n))*abs(ffta)
    psda[2:-1]=2*psda[2:-1]
    #freq=np.arange(0,(fs/2),fs/n)
    freq=np.arange(frange[0],frange[1],fs/n)
    psda=psda[frange[0]/(fs/n)-2:frange[1]/(fs/n)]
    npsda=norm(psda)
    smnpsda=smooth(npsda,20)
    print(len(smnpsda))
    print(len(freq))
    print(len(psda))
    plt.plot(freq,(smnpsda))
    plt.title('psd using fft')
    plt.xlabel('freq (Hz)')
    plt.ylabel('power(dB)')
    plt.show()
    return(psda)

def is_sylable(a, fs = 32000):
    """Test whether the sound contained in array a is a sylable based on several criterian
    input: 
    a = an array of samples
    fs = sampling rate
    output:
    true or False"""
    # parameters
    test = True
    max_length = 0.5 # seconds
    min_length = 0.02 # seconds
    if float(len(a))/fs < min_length: test = False
    elif float(len(a))/fs > max_length: test = False
    return test

def is_stack(a, fs = 32000):
    """Test whether a sylable is a stack based on criteria below"""
    if not is_sylable(a, fs): return False
    test = False
    fft=sc.fftpack.fft(a)
    corr=sc.fftpack.ifft(fft*fft.conjugate())
    corr=corr[:(len(corr)/4)]
    dfff=np.diff(corr)
    dat=np.diff(np.where(dfff>0,1,0))
    return test

def allign_by_xcorr(array_of_wfs, fs = 32000):
    """allign wfs by xcorr each wf's envelope with the mean evelope of all wfs provided. 
        Inputs:     
                array_of_wfs - list or np array of lists or np arrays (or a square np array)

                kwargs:
                    fs = 32000
                    window_size
        Outputs:
                wfs_out - np array of zeropadded, alligned wfs
    """
    # calculate stuff about the wfs in array of wfs
    max_length = max([len(wf) for wf in array_of_wfs])
    mean_length = np.mean([len(wf) for wf in array_of_wfs])
    # create envelopes matrix
    envelopes = np.zeros((len(array_of_wfs), max_length))
    for kwf,wf in enumerate(array_of_wfs):
        blank = np.zeros(max_length)
        blank[:len(wf)] = wf
        envelopes[kwf] = smoothrect(blank)
    # loop thru envelopes xcorrelating with the mean and drop wfs into wfs out based on the laggs
    wfs_out = np.zeros((len(array_of_wfs), max_length))
    mean_envelope = np.mean(envelopes, axis = 0)
    start_idx = int(max_length/2) - round(float(mean_length)/2)
    lags = np.arange(-max_length+1, max_length, 1, dtype = int)
    offsets = np.zeros(len(array_of_wfs))
    for kwf, wf in enumerate(array_of_wfs):
        # do cross correlation, find max lag and save in offsets array
        corr = np.correlate(envelopes[kwf], mean_envelope, 'full')
        mx_idx = np.argmax(corr)
        offsets[kwf] = lags[mx_idx]
        # err there are issues here when the wf ends up outside the bounds of your output array. These issues are central to
        # alligning the syllbales since they are of different lengths.  This works but is ugly.  
        idx1 = start_idx-offsets[kwf]
        idx2 = start_idx - offsets[kwf] + len(wf)
        if idx1 >= 0 and idx2 <= wfs_out.shape[1]:
            wfs_out[kwf, idx1:idx2] = wf #smoothrect(wf)
        else:
            if idx1 < 0:
                wfidx1 = abs(idx1)
                idx1 = 0
                idx2 = len(wf) - wfidx1
                wfs_out[kwf,idx1:idx2] = wf[wfidx1:]
            elif idx2 > wfs_out.shape[1]:
                wfidx2 = len(wf) + (wfs_out.shape[1]-idx2)
                idx2 = wfs_out.shape[1] 
                wfs_out[kwf,idx1:idx2] = wf[:wfidx2]
    return wfs_out


# sandbox for jeff
if __name__ == "__main__":
    import os
    path = "/data/brainard_lab/parent_pairs_nest/N56/parent/"
    parent_song_files=[path+x for x in os.listdir(path)]
    parent_song_files = np.array(filter(lambda x: x[-4:]=='.wav', parent_song_files))
    sylables = []
    labels = []
    for sf in parent_song_files[0:5]:
        syls, labs = evsonganaly.get_ev_sylables(evsonganaly.load_ev_file(sf))
        sylables.extend(syls)
        labels.extend(labs)
    labels = np.array(labels)
    sylables = np.array(sylables)
    traces = []
    for ksyl,sylable in enumerate(sylables[labels == 'a']):
        if is_sylable(sylable):# and ksyl==10:
            print(ksyl)
            data = cluster.CalculateSpectrum(sylable, plot = True)
            import ipdb; ipdb.set_trace(); 
            # t = np.divide(np.arange(0,len(sylable), dtype = float),32000)
            # #(tfreq, ffreq) = ffprofile_specgram(sylable, plot = True)
            # (tfreq, ffreq) = ffprofile_corr(sylable, plot = False)
            # traces.append(smooth(ffreq,window = 1, freq = 32000))
            
    import ipdb; ipdb.set_trace(); 

