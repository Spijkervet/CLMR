import torchaudio

def tensor_to_audio(fn, t, sr):
    torchaudio.save(fn, t.cpu(), sample_rate=sr)