import torch
import torch.nn.functional as F
import torchaudio
from hardmodel import ResCnn
import config
import pandas as pd
import torch
import torch.nn.functional as F
import torchaudio
import config
from dataset import mel_to_db
from dataset import mel_spectrogram 

def predict(model, audio_path):
        tags = pd.read_csv(config.DATA_DIR + config.TAGS_FILE, header=None)
        tags = tags[0].values.tolist()
        model.eval()
        with torch.no_grad():
            mel_spectrograms = predprocess(audio_path)
            output = model(mel_spectrograms)
            output = torch.special.expit(output)
            threshold = config.THRESHOLD
            binary_tensor = (output >= threshold).float()
            sum_tensor = torch.sum(binary_tensor, dim=0)
            result = [tags[i] for i in range(len(tags)) if sum_tensor[i] >= 1]
            print(result)
        return torch.special.expit(output)
        
def predprocess(audio_path):
    signal, sample_rate = torchaudio.load(audio_path)
    signal = signal.to(config.DEVICE)
    
    if sample_rate != config.SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sample_rate, config.SAMPLE_RATE).to(
                config.DEVICE
        )
        signal = resampler(signal)
        
    if signal.shape[0] > 1:
        signal = torch.mean(signal, dim=0, keepdim=True)
    
    signals_cutted = signal.unfold(1, config.NUM_SAMPLES // 8, config.NUM_SAMPLES // 16)
    
    mel_spectrograms = mel_spectrogram(signals_cutted)   
    mel_spectrograms = mel_to_db(mel_spectrograms)
    mel_spectrograms = F.interpolate(
            mel_spectrograms, size=(config.IMAGE_SIZE), mode="bilinear", align_corners=False
    )
    return torch.permute(mel_spectrograms,(1,0,2,3))
    
if __name__ == '__main__':
    model = ResCnn(
        in_channels=config.IN_CHANNELS,
        classes_num=config.CLASSES_NUM,
        kernel_size=config.KERNEL_SIZE,
        stride=config.STRIDE,
        image_size=config.IMAGE_SIZE
    ).to(config.DEVICE)
    #TODO: choose track for main
    model.load_state_dict(torch.load('../models/version1/model_20240513_131650_6'))
    predict(model,'hiphop.wav')
    
