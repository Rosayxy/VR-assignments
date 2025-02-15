import wave
import contextlib
import numpy as np
import argparse


class BinauralSound():
    def __init__(self, hrir_path):
        self.hrir_l, self.hrir_r = self.load_hrir(hrir_path)

    def load_hrir(self, hrir_path):
        """
        Load HRIR data from given file path.
        HRIR data for each ear is a NumPy array of shape (25, 50, 200).
        The first dimension is the azimuth from -80 to 80, and the second dimension is the elevation from -45 to -45 + 49 * 5.625
        """
        hrir_lr = np.load(hrir_path)
        return hrir_lr['hrir_l'], hrir_lr['hrir_r']

    def load_wav(self, path):
        """
        Load 1-channel 16-bit uncompressed WAV given file path.
        Returns sample rate and audio sample data. The audio sample data is a 1-D NumPy array of type np.int16.
        """
        with contextlib.closing(wave.open(path,'rb')) as f:
            if f.getcomptype() != 'NONE':
                raise ValueError('The input audio must not be compressed.')
            n_channels = f.getnchannels()
            if n_channels != 1:
                raise ValueError('The input audio must be 1-channel WAV.')
            sample_width = f.getsampwidth()
            if sample_width != 2:
                raise ValueError('The input audio must be 16-bit WAV.')
            sample_rate = f.getframerate()
            n_samples = f.getnframes()
            dat = f.readframes(n_samples * n_channels)
        audio = np.frombuffer(dat, dtype=np.int16)
        print(f'Loaded audio from {path}.')
        return sample_rate, audio

    def save_wav(self, path, sample_rate, samples):
        """
        Save 2-channel 16-bit WAV to the given path.
        """
        if isinstance(samples, np.ndarray) and samples.dtype == np.int16 and samples.ndim == 2 and samples.shape[1] == 2:
            with contextlib.closing(wave.open(path, 'w')) as f:
                f.setparams((2, 2, sample_rate, samples.shape[0], 'NONE', 'not compressed'))
                f.writeframes(samples.tobytes('C'))
        else:
            raise TypeError('The audio data must be numpy array with dtype int16 and shape (N, 2).')
        print(f'Saved audio to {path}.')
    
    def convolve_fft(self, data, filter):
        """
        data * filter => ifft(fft(data) x filter(data)), where * denotes convolution (卷积) and x denotes multiplication.
        可以用于每一个小片段的环绕效果
        """
        L = len(data) + len(filter) - 1
        return np.fft.ifft(np.fft.fft(data, L) * np.fft.fft(filter, L)).real[:len(data)]

    def gen_demo(self, input_audio_path):
        sample_rate, audio = self.load_wav(input_audio_path)
        audio = audio.astype(np.float32)
        audio_l, audio_r = self.convolve_fft(audio, self.hrir_l[0, 8]), self.convolve_fft(audio, self.hrir_r[0, 8])
        audio_lr = np.stack((audio_l, audio_r), axis=1)
        # normalize the amplitude to avoid overflow
        i16min, i16max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
        audio_lr = (audio_lr * i16max / np.abs(audio_lr).max()).clip(i16min, i16max).astype(np.int16) # [-32768, 32767]
        self.save_wav('left-side.wav', sample_rate, audio_lr) 

    def gen_360(self, input_audio_path):
        """
        Implement this function to generate binaural audio with time-varying space locations.
        For example, to generate an audio with the sound source rotating around.
        You could also try other audios for fun.
        This function generates a 2-channel 16-bit WAV file that projects the sound source to horizontal (-80,80) degrees around the listener.
        It's design mainly devides the audio into small parts, and convolve each part with the corresponding HRIR.
        Besides, because the fft convolution might have unexpected effect at the borders of the devided parts, thus causing small pop noises, we need to add some padding to avoid the edge effect.
        """
        
        sample_rate, audio = self.load_wav(input_audio_path)
        audio = audio.astype(np.float32)
        position_num = self.hrir_l.shape[0]
        duration = len(audio) / sample_rate
        audio_lr = np.zeros((len(audio), 2), dtype=np.float32)
        unit_len = int(len(audio)//position_num)
        print(f'unit_len: {unit_len}, duration: {duration}, position_num: {position_num}, len(audio): {len(audio)} remainder: {len(audio)%position_num}')
        for i in range(position_num):
            left_boarder = max(0,unit_len*i-100)
            left_start = unit_len*i-left_boarder
            right_end = min(len(audio),unit_len*(i+1)+100)
            
            audio_l, audio_r = self.convolve_fft(audio[left_boarder:right_end], self.hrir_l[i, 8]), self.convolve_fft(audio[left_boarder:right_end], self.hrir_r[i, 8])
            audio_lr[unit_len*i:unit_len*(i+1), 0] = audio_l[left_start:left_start+unit_len]
            audio_lr[unit_len*i:unit_len*(i+1), 1] = audio_r[left_start:left_start+unit_len]

        # do remainder
        audio_l, audio_r = self.convolve_fft(audio[unit_len*position_num-100:], self.hrir_l[position_num-1, 8]), self.convolve_fft(audio[unit_len*position_num-100:], self.hrir_r[position_num-1, 8])
        audio_lr[unit_len*position_num:, 0] = audio_l[100:]
        audio_lr[unit_len*position_num:, 1] = audio_r[100:]
        
        # normalize the amplitude to avoid overflow
        i16min, i16max = np.iinfo(np.int16).min, np.iinfo(np.int16).max
        audio_lr = (audio_lr * i16max / np.abs(audio_lr).max()).clip(i16min, i16max).astype(np.int16) # [-32768, 32767]
        self.save_wav('360.wav', sample_rate, audio_lr)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio', type=str, default='assets/nokia.wav')
    parser.add_argument('--demo', action='store_true', help='Run demo program.')
    args = parser.parse_args()

    bs = BinauralSound('assets/hrir.npz')

    if args.demo:
        bs.gen_demo(args.audio)
    else:
        bs.gen_360(args.audio)
