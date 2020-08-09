import torch
from torchaudio.transforms import Spectrogram, TimeStretch, Resample
from torchaudio import functional as F

class PitchShift(torch.nn.Module):
    r"""Shift the pitch of a waveform by n_steps semitones.
    Args:
        sample_rate(int): Waveform sampling rate.
        n_steps(int, optional): How many (fractional) half-steps to shift waveform. (Default: ``4``)
        bins_per_octave(int, optional): How many steps per octave. (Default: ``12``)
        n_fft (int, optional): Size of FFT, creates ``n_fft // 2 + 1`` bins. (Default: ``400``)
        win_length (int, optional): Window size. (Default: ``n_fft``)
        hop_length (int, optional): Length of hop between STFT windows. (Default: ``win_length // 2``)
    """

    def __init__(self,
                 sample_rate,
                 n_steps=4,
                 bins_per_octave=12,
                 n_fft=400,
                 win_length=None,
                 hop_length=None,
                 ):
        super(PitchShift, self).__init__()

        self.sample_rate = sample_rate
        self.n_steps = n_steps
        self.bins_per_octave = bins_per_octave
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length

        if bins_per_octave < 1:
            raise ValueError('bins_per_octave must be a positive integer.')

        if not self.win_length:
            self.win_length = self.n_fft
        if not self.hop_length:
            self.hop_length = self.win_length // 2

        self.Spectrogram = Spectrogram(power=None,
                                       n_fft=self.n_fft,
                                       win_length=self.win_length,
                                       hop_length=self.hop_length).to("cuda")

        n_freq = n_fft // 2 + 1
        self.rate = 2.0 ** (-n_steps / self.bins_per_octave)

        self.TimeStretch = TimeStretch(hop_length=self.hop_length, n_freq=n_freq, fixed_rate=self.rate).to("cuda")
        self.Resample = Resample(self.sample_rate / self.rate, self.sample_rate).to("cuda")

    def forward(self, waveform):
        # type: (Tensor) -> Tensor
        r"""
        Args:
            waveform (torch.Tensor): Tensor of audio of dimension (..., time).
        Returns:
            torch.Tensor: Tensor of audio of dimension (..., time).
        """

        complex_specgrams = self.Spectrogram(waveform)
        complex_specgrams_stretch = self.TimeStretch(complex_specgrams)
        waveform_stretch = F.istft(complex_specgrams_stretch,
                                   n_fft=self.n_fft,
                                   hop_length=self.hop_length,
                                   win_length=self.win_length)

        waveform_shift = self.Resample(waveform_stretch)

        waveform_length = waveform.size(-1)
        waveform_shift_length = waveform_shift.size(-1)

        if waveform_length < waveform_shift_length:
            return waveform_shift[..., :waveform_length]

        return torch.nn.functional.pad(waveform_shift, [0, waveform_length - waveform_shift_length])