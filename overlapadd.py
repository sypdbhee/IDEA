import numpy as np

def overlap(X, yphase):
    windowLen = 512
    ShiftLen = 256
    [FreqRes, FrameNum] = X.shape
    
    Spec = X*np.exp(1j*yphase)

    if windowLen % 2 == 1:
        Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1:, ]))))
    else:
        Spec = np.concatenate((Spec, np.flipud(np.conj(Spec[1:-1, ]))))

    sig = np.zeros((FrameNum-1)*ShiftLen+windowLen)
    for i in range(FrameNum):
        start = i*ShiftLen
        spec = Spec[:, i]
        sig[start:start+windowLen] = sig[start:start+windowLen]+np.real(np.fft.ifft(spec, windowLen))
    return sig
