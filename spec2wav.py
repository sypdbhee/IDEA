from overlapadd import overlap
import numpy as np

def PowerSpectrum2Wave(spec, yphase):
    spec = spec.T
    yphase = yphase.T
    spec = 10**spec
    yphase = yphase[:int(np.floor(len(yphase)/2)+1), :]
    sig = overlap(np.sqrt(spec), yphase)
    return sig
