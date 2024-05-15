import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import config
import random

def mel_to_db(mel_spec, floor=1e-6):
    return 10 * torch.log10(mel_spec + floor)

mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=config.SAMPLE_RATE,
        n_fft=config.N_FFT,
        win_length=config.WIN_LENGTH,
        hop_length=config.HOP_LENGTH,
        center=config.CENTER,
        pad_mode=config.PAD_MODE,
        power=config.POWER,
        norm=config.NORM,
        n_mels=config.N_MELS,
        mel_scale=config.MEL_SCALE,
    ).to(config.DEVICE)


class MTATDataset(Dataset):
    def __init__(
        self,
        device,
        data_dir,
        annotations_file,
        tags_file,
        target_sample_rate,
        num_samples,
        transformation,
        target_image_size,
    ):
        self.data_dir = data_dir
        tags = pd.read_csv(self.data_dir + tags_file, header=None)
        tags = pd.concat(
            [pd.DataFrame([["clip_id"], ["mp3_path"]]), tags], ignore_index=True
        )
        self.tags = tags[0].values.tolist()

        self.annotations = pd.read_csv(
            self.data_dir + annotations_file, delimiter="\t", header=0
        )
        self.annotations = self.annotations.loc[:, "clip_id":"mp3_path"]
        self.annotations = self.annotations[self.annotations.eq(1).any(axis=1)]
        self.annotations = self.annotations.loc[:, self.tags]
        self.device = device
        self.transformation = transformation.to(self.device)
        self.target_sample_rate = target_sample_rate
        self.num_samples = num_samples
        self.target_image_size = target_image_size

    def __len__(self):
        return len(self.annotations) * 15

    def __getitem__(self, index):
        audio_sample_path = self.data_dir + self.annotations.iloc[index // 15, 1]
        label = self._get_audio_sample_label(index // 15).to(self.device)

        signal, sr = torchaudio.load(audio_sample_path,normalize=True)
        signal = signal.to(torch.float32)
        signal = signal.to(self.device)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self._cut_if_necessary(signal)
        signal = self._right_pad_if_necessary(signal)
        signal = signal[0]

        signals_cutted = signal.unfold(0, self.num_samples // 8, self.num_samples // 16)
        signal = self.transformation(signals_cutted[index % 15])
        signal = signal.unsqueeze(0).unsqueeze(0)
        signal = F.interpolate(
            signal, size=self.target_image_size, mode="bilinear", align_corners=False
        )
        signal = signal.squeeze()
        signal_db = mel_to_db(signal)
        return signal_db.unsqueeze(0), label

    def _cut_if_necessary(self, signal):
        if signal.shape[1] > self.num_samples:
            signal = signal[:, : self.num_samples]
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal
    def _right_pad_if_necessary(self, signal):
        length_signal = signal.shape[1]
        if length_signal < self.num_samples:
            num_missing_samples = self.num_samples - length_signal
            last_dim_padding = (0, num_missing_samples)
            signal = torch.nn.functional.pad(signal, last_dim_padding).to(self.device)
        return signal

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_label(self, index):
        label = self.annotations.iloc[index, 2:].values.astype(np.float32)
        return torch.tensor(label).to(torch.float32)



if __name__ == "__main__":
    MTAT = MTATDataset(
        device=config.DEVICE,
        data_dir=config.DATA_DIR,
        annotations_file=config.ANNOTATIONS_FILE,
        tags_file=config.TAGS_FILE,
        transformation=mel_spectrogram,
        target_sample_rate=config.SAMPLE_RATE,
        num_samples=config.NUM_SAMPLES,
        target_image_size=config.IMAGE_SIZE,
    )
    print(f"There are {len(MTAT)} samples in the dataset.")
    index = random.randint(0, len(MTAT))
    signal, label = MTAT[index]
    print(signal.shape)
    import matplotlib.pyplot as plt
    signal_cpu = signal.to('cpu')
    plt.imshow(signal_cpu.squeeze(), cmap='hot', origin='lower')
    plt.colorbar()
    plt.show()
