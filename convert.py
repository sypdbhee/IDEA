import os, sys, pickle
import numpy as np
from spec2wav import PowerSpectrum2Wave 
from librosa.output import write_wav as write
import scipy
import pdb

def main():
    os.system("mkdir -p "+sys.argv[1]+"wav/")
    with open(sys.argv[1]+'pred.txt', 'rb') as fp:
        data = pickle.load(fp)
    i = sys.argv[1][-2]
    if i == '0':
        i = '10'
    with open('../Data/test/noisy_'+i+'_parameter.txt', 'rb') as fp:
        parameters = pickle.load(fp)
    for i in range(len(data)):
        sig = np.asarray(data[i])
        sig = np.squeeze(sig)
        wav = PowerSpectrum2Wave(sig, parameters[i][0][5:-5, :])
        wav = wav*10
        wav = wav.astype(np.int16)
        print(parameters[i][1])
        name = parameters[i][1]
        scipy.io.wavfile.write(sys.argv[1]+'wav'+name,16000,np.int16(wav));

if __name__ == '__main__':
    main()
