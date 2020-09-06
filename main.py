import os
import torch
import librosa
import librosa.display
import numpy as np
from scipy.io.wavfile import write
import matplotlib.pyplot as plt

from unet import UNet

torch.backends.cudnn.deterministic = True
seed = 1234
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_psnr(img1, img2, min_value=0, max_value=1):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = max_value - min_value
    return 10 * torch.log10((PIXEL_MAX ** 2) / mse)

def save_spectrogram(data, file_name):
    # librosa.display.specshow(librosa.amplitude_to_db(data, ref = np.max), y_axis='log', x_axis='time')
    # plt.tight_layout()
    magnitude = np.absolute(data)
    plt.imshow(magnitude, cmap='magma')
    plt.savefig(file_name)

def save_wav(stft, sample_rate, file_name):
    data = librosa.istft(stft, center=False)
    write(file_name, sample_rate, data)

class Trainer(object):
    def __init__(self, wav_name, conv_type='regular'):
        self.wav_name = wav_name
        self.make_toy_data(wav_name)
        stft, noise_stft, self.sample_rate, normalize_data = self.prepare_data(wav_name)
        self.stft_real_max, self.stft_real_min, self.stft_imag_max, self.stft_imag_min = normalize_data

        self.clean_data = self.make_torch_data(stft)
        # self.corrupted_data = self.make_torch_data(noise_stft)
        self.corrupted_data = self.make_torch_data(stft + np.random.randn(*stft.shape)*0.1)
        # self.corrupted_data = self.clean_data.clone() + torch.randn(*self.clean_data.shape) * 0.1
        # self.fixed_input = torch.randn(*self.clean_data.shape) * 0.1

        self.conv_type = conv_type
        self.model = UNet(conv_type=self.conv_type)
        self.output_dir = "toy_experiment_output"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        save_spectrogram(self.torch_to_stft(self.clean_data),
                         os.path.join(self.output_dir, '{}_{}.png'.format(self.conv_type, 'clean')))
        save_spectrogram(self.torch_to_stft(self.corrupted_data),
                         os.path.join(self.output_dir, '{}_{}.png'.format(self.conv_type, 'corrupted')))

    def make_toy_data(self, wav_name):
        sample_rate = 44100
        duration = 5
        t = np.linspace(0., duration, sample_rate * duration)
        amplitude = np.iinfo(np.int16).max
        data = amplitude * (
                    np.sin(2. * np.pi * 1000 * t) + np.sin(2. * np.pi * 2000 * t) + np.sin(2. * np.pi * 3000 * t))
        write(wav_name, sample_rate, data)

    def prepare_data(self, file_name):
        duration = 3
        data, sample_rate = librosa.load(file_name)
        data = data[:sample_rate * duration]
        data = data[:(len(data)//2**4) * 2**4]
        win_size = int(round(sample_rate * 0.032)) # 1 frame = 32ns
        
        stft = librosa.stft(data, n_fft=win_size+4, win_length=win_size,center=False)
        noise_stft = librosa.stft(data + np.random.randn(*data.shape) * 0.1, n_fft=win_size+4, win_length=win_size,center=False)
        
        stft_real_max, stft_real_min = np.max(stft.real), np.min(stft.real)
        stft_imag_max, stft_imag_min = np.max(stft.imag), np.min(stft.imag)

        return stft, noise_stft, sample_rate, (stft_real_max, stft_real_min, stft_imag_max, stft_imag_min)

    def make_torch_data(self, stft):
        # we treat complex STFT coefficients as two separate real-valued channels.
        torch_data = torch.zeros(2, stft.shape[0], stft.shape[1])
        torch_data[0, :, :] = torch.FloatTensor((stft.real - self.stft_real_min) / (self.stft_real_max - self.stft_real_min))
        torch_data[1, :, :] = torch.FloatTensor((stft.imag - self.stft_imag_min) / (self.stft_imag_max - self.stft_imag_min))
        return torch_data

    def torch_to_stft(self, data):
        data = data.clone().detach().cpu().numpy()
        out = np.zeros((data.shape[1], data.shape[2]), dtype=np.complex64)
        out.real = data[0, :, :] * (self.stft_real_max - self.stft_real_min) + self.stft_real_min
        out.imag = data[1, :, :] * (self.stft_imag_max - self.stft_imag_min) + self.stft_imag_min
        return out

    def train(self):
        lr = 0.001
        epochs = 1000
        saving_epochs = [50, 200, 1000]

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()

        self.model = self.model.cuda()
        self.clean_data = self.clean_data.cuda()
        self.corrupted_data = self.corrupted_data.cuda()

        loss_avg_meter = AverageMeter()
        loss_log = []
        for i in range(epochs):
            optimizer.zero_grad()
            out = self.model(self.corrupted_data)
            loss = criterion(out, self.clean_data)
            loss.backward()
            optimizer.step()

            loss_avg_meter.update(loss.item())
            loss_log.append(loss_avg_meter.avg)
            if (i+1) % 10 == 0:
                print("[{:4d}/{:4d}] loss : {:.4f}, PSNR : {:.4f}".format(i+1, epochs, loss_avg_meter.avg, get_psnr(out, self.clean_data)))
            if i+1 in saving_epochs:
                save_spectrogram(self.torch_to_stft(self.corrupted_data),
                                 os.path.join(self.output_dir, '{}_{:04d}.png'.format(self.conv_type, i+1)))
                save_wav(self.torch_to_stft(self.corrupted_data), self.sample_rate,
                         os.path.join(self.output_dir, '{}_{:04d}.wav'.format(self.conv_type, i+1)))

if __name__ == '__main__':
    trainer = Trainer('toy_data.wav', conv_type='regular')
    trainer.train()



