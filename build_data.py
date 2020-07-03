import pickle, sys
import numpy as np
from glob import iglob
from librosa.core import load as read
from spectrum import Spectrum

def main():
    dirname = sys.argv[1]
    data = []
    parameter = []
    for filename in iglob(dirname + '/*.wav'):
        print(filename)
        sig, rate = read(filename, sr=16000)
        sig = (sig - np.mean(sig))/np.var(sig)
        sp, yphase = Spectrum(sig, 512, 256, 512, 2)
        data.append(sp)
        parameter.append([yphase, filename.replace(dirname, '')])
        with open(sys.argv[2]+'_parameter.txt', 'wb') as fp:
            pickle.dump(parameter, fp)
        with open(sys.argv[2]+'.txt', 'wb') as fp:
            pickle.dump(data, fp)

if __name__ == '__main__':
    main()
