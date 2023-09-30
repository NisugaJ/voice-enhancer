import noisereduce as nr
from scipy.io import wavfile

def noise_reduce():
    # load data
    rate, data = wavfile.read("../../../AppData/tempStorage/convertedAudio/out.wav")
    # select section of data that is noise
    noisy_part = data[10000:15000]
    # perform noise reduction
    reduced_noise = nr.reduce_noise(audio_clip=data, noise_clip=noisy_part, verbose=True)

# place marker according to the user = 'user_focused'
# noise_reduce()