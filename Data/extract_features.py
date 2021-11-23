import numpy as np
import scipy.io.wavfile
import scipy.signal
import os
import matplotlib.pyplot as plt


# Each wav is padded/truncated to 16000 samples and then converted in a
# 20x64 spectrogram (20 frequencies x 64 segments).  Actually,
# spectrograms of 33 frequencies are first computed.  The first 19
# frequencies are retained, and the remaining are added together.
# Finally, spectrograms are flattened in a vector.

DATA_DIR = "speech_commands_v0.02"
SAMPLES = 16000  # 1 second at 16 KHz
SEGMENTS = 80
FREQUENCIES = 20  # This keeps almost everything


def load_wav(filename):
    rate, wav = scipy.io.wavfile.read(filename)
    if wav.ndim == 2:
        wav = wav.mean(1)
    wav = wav[:SAMPLES]
    if wav.shape[0] < SAMPLES:
        wav = np.pad(wav, (0, SAMPLES - wav.shape[0]))
    return wav


def spectrogram(x):
    s = scipy.signal.spectrogram(x, nperseg=SAMPLES // SEGMENTS, noverlap=0)
    freqs, segs, y = s
    y[FREQUENCIES - 1, :] = y[FREQUENCIES - 1:, :].sum(0)
    y = y[:FREQUENCIES, :].reshape(-1)
    return y


def get_classes():
    classes = [
        c for c in os.listdir(DATA_DIR)
        if os.path.isdir(os.path.join(DATA_DIR, c))
        if not c.startswith("_")
    ]
    classes.sort()
    return classes


def load_data():
    sgrams = []
    labels = []
    progressive = []
    names = []
    classes = get_classes()
    for label, class_ in enumerate(classes):
        print(label, class_)
        path = os.path.join(DATA_DIR, class_)
        files = [f for f in os.listdir(path) if f.endswith(".wav")]
        for f in files:
            fname = os.path.join(path, f)
            x = load_wav(fname)
            s = spectrogram(x)
            sgrams.append(s)
            # plt.imsave(class_ + ".png", s.reshape(FREQUENCIES, -1), cmap="hot")
            # break
            labels.append(label)
            progr = int(f.partition("_")[0], 16)
            progressive.append(progr)
            names.append(os.path.join(class_, f))
    X = np.stack(sgrams, 0)
    Y = np.array(labels)
    P = np.array(progressive)
    N = np.array(names)
    return X, Y, P, N


if __name__ == "__main__":
    X, Y, P, names = load_data()
    P = P % 10
    test_indices = (P == 0).nonzero()[0]
    validation_indices = (P == 1).nonzero()[0]
    train_indices = (P > 1).nonzero()[0]
    np.random.shuffle(train_indices)
    smalltrain_indices = train_indices[:10000]
    
    with open("classes.txt", "w") as f:
        for c in get_classes():
            print(c, file=f)
    
    for name, idx in [("test", test_indices),
                      ("validation", validation_indices),
                      ("train", train_indices),
                      ("smalltrain", smalltrain_indices)]:
        np.savez_compressed(name, X[idx, :], Y[idx])
        np.savetxt(name + "-names.txt", names[idx], fmt="%s")
