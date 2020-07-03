import numpy as np

def Spectrum(sig, FrameLength, FrameRate, FFT_SIZE, flag):
    Len = len(sig)
    ncols = int((Len-FrameLength)/FrameRate)
    fftspectrum = np.zeros([FFT_SIZE, ncols])
    Spectrum = np.zeros([FFT_SIZE//2+1, ncols])
    En = np.zeros([1, ncols])
    wind = np.hamming(FrameLength)
    x_seg = []
    fftspectrum = []
    yphase = []
    Spec = []
    i = 0
    for t in range(0, Len-FrameLength, FrameRate):
        x_seg.append(wind*(sig[t:(t+FrameLength)]))
        fftspectrum.append(np.fft.fft(x_seg[i], FFT_SIZE))
        yphase.append(np.angle(fftspectrum[i]))
        Spec.append(np.abs(fftspectrum[i][0:FFT_SIZE//2+1]))
        i += 1

    fftspectrum = np.array(fftspectrum)
    yphase = np.array(yphase)
    Spec = np.array(Spec)
    if flag==2:
        Spec = Spec**2
    elif flag==1:
        Spec = Spec
    else:
        Spec = fftspectrum[0:FFT_SIZE//2, :]
    return np.log10(Spec), yphase

