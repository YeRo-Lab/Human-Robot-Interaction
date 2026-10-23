import numpy as np
import wave
from scipy.io import wavfile
import noisereduce as nr

def extract_mono_channel(input_wav, output_wav, channel_index=2):
    wf = wave.open(input_wav, 'rb')
    n_channels = wf.getnchannels()
    framerate = wf.getframerate()
    raw = wf.readframes(wf.getnframes())
    wf.close()

    audio = np.frombuffer(raw, dtype=np.int16)
    audio = audio.reshape(-1, n_channels)
    mono = audio[:, channel_index]

    wf_out = wave.open(output_wav, 'wb')
    wf_out.setnchannels(1)
    wf_out.setsampwidth(2)
    wf_out.setframerate(framerate)
    wf_out.writeframes(mono.tobytes())
    wf_out.close()
    print("Extracted mono channel to:", output_wav)

def denoise_wav(input_wav, output_wav):
    rate, data = wavfile.read(input_wav)
    reduced_noise = nr.reduce_noise(
        y=data,
        sr=rate,
        prop_decrease=0.7,        # how aggressively to reduce (1.0 = full, lower = gentler)
        stationary=True,          # try this - see note below
        n_std_thresh_stationary=2.5  # higher = less aggressive gating (default is 1.5)
    )
    wavfile.write(output_wav, rate, reduced_noise)
    print("Saved denoised audio to:", output_wav)

if __name__ == "__main__":
    extract_mono_channel("1captured_audio.wav", "1captured_audio_mono.wav")
    denoise_wav("1captured_audio_mono.wav", "1captured_audio_denoised.wav")
    for i in range(4):
        extract_mono_channel("1captured_audio.wav", "channel_%d.wav" % i, channel_index=i)