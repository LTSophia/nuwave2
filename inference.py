from .lightning_model import NuWave2
from omegaconf import OmegaConf as OC
import os
import argparse
import torch
import librosa as rosa
from scipy.io.wavfile import write as swrite
import numpy as np
from scipy.signal import resample_poly

nuwave2_dir = os.path.dirname(os.path.realpath(__file__))

def main(checkpoint, wavarg, sr, steps=None, device='cuda', result_dir=None):
    hparams = OC.load(os.path.join(nuwave2_dir, 'hparameter.yaml'))
    if result_dir is None:
        result_dir = hparams.log.test_result_dir
    if steps is None or steps == 8:
        steps = 8
        noise_schedule = eval(hparams.dpm.infer_schedule)
    else:
        noise_schedule = None
    model = NuWave2(hparams).to(device)
    model.eval()
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'] if not('EMA' in checkpoint) else ckpt)

    highcut = sr // 2
    nyq = 0.5 * hparams.audio.sampling_rate
    hi = highcut / nyq

    wav, _ = rosa.load(wavarg, sr=sr, mono=True)
    wav /= np.max(np.abs(wav))

    # upsample to the original sampling rate
    wav_l = resample_poly(wav, hparams.audio.sampling_rate, sr)
    wav_l = wav_l[:len(wav_l) - len(wav_l) % hparams.audio.hop_length]

    fft_size = hparams.audio.filter_length // 2 + 1
    band = torch.zeros(fft_size, dtype=torch.int64)
    band[:int(hi * fft_size)] = 1

    wav_l = torch.from_numpy(wav_l.copy()).float().unsqueeze(0).to(device)
    band = band.unsqueeze(0).to(device)

    wav_recon, wav_list = model.inference(wav_l, band, steps, noise_schedule)

    wavname = os.path.basename(wavarg)

    wav_recon = torch.clamp(wav_recon, min=-1, max=1 - torch.finfo(torch.float16).eps)

    swrite(os.path.join(result_dir, wavname),
           hparams.audio.sampling_rate, wav_recon[0].detach().cpu().numpy())
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--checkpoint',
                        type=str,
                        required=True,
                        help="Checkpoint path")
    parser.add_argument('-i',
                        '--wav',
                        type=str,
                        default=None,
                        help="audio")
    parser.add_argument('--sr',
                        type=int,
                        required=True,
                        help="Sampling rate of input audio")
    parser.add_argument('--steps',
                        type=int,
                        required=False,
                        help="Steps for sampling")
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        required=False,
                        help="Device, 'cuda' or 'cpu'")

    args = parser.parse_args()
    main(args.checkpoint, args.wav, args.sr, steps=args.steps, device=args.device)
