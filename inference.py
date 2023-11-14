from .lightning_model import NuWave2
from omegaconf import OmegaConf as OC
import os
import argparse
import torch
import librosa as rosa
from scipy.io.wavfile import write as swrite
import numpy as np
from scipy.signal import resample_poly
import ffmpeg
from shutil import copy as shcopy

nuwave2_dir = os.path.dirname(os.path.realpath(__file__))

def main(checkpoint, wav_files, device='cuda', result_dir=None):
    hparams = OC.load(os.path.join(nuwave2_dir, 'hparameter.yaml'))
    if result_dir is None:
        result_dir = hparams.log.test_result_dir
    output_fol = os.path.realpath(result_dir)
    os.makedirs(output_fol, exist_ok=True)
    steps = 8
    noise_schedule = eval(hparams.dpm.infer_schedule)

    model = NuWave2(hparams).to(device)
    model.eval()
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['state_dict'] if not('EMA' in checkpoint) else ckpt)

    nyq = 0.5 * hparams.audio.sampling_rate
    fft_size = hparams.audio.filter_length // 2 + 1
    total_files = len(wav_files)
    i = 0
    progress_bar(0, total_files)
    for wav_file in wav_files:
        i += 1
        wav_name = os.path.basename(wav_file)
        print_wav = os.path.splitext(wav_name)[0]
        progress_bar(i, total_files, file_name=print_wav)
        sr = int(ffmpeg.probe(wav_file)['streams'][0]['sample_rate'])
        if (sr >= hparams.audio.sampling_rate):
            shcopy(wav_file, os.path.join(output_fol, wav_name))
            continue
        highcut = sr // 2
        hi = highcut / nyq

        wav, _ = rosa.load(wav_file, sr=sr, mono=True)
        wav /= np.max(np.abs(wav))

        # upsample to the original sampling rate
        wav_l = resample_poly(wav, hparams.audio.sampling_rate, sr)
        wav_l = wav_l[:len(wav_l) - len(wav_l) % hparams.audio.hop_length]

        band = torch.zeros(fft_size, dtype=torch.int64)
        band[:int(hi * fft_size)] = 1

        wav_l = torch.from_numpy(wav_l.copy()).float().unsqueeze(0).to(device)
        band = band.unsqueeze(0).to(device)

        wav_recon = model.inference(wav_l, band, steps, noise_schedule)

        wav_recon = torch.clamp(wav_recon, min=-1, max=1 - torch.finfo(torch.float16).eps)

        swrite(os.path.join(output_fol, wav_name),
            hparams.audio.sampling_rate, wav_recon[0].detach().cpu().numpy())

def progress_bar(current, total, file_name='\t', fill='#'):
    length = 50
    total_digits = max(len(str(total)), 2)
    ws = ' ' * (8 - total_digits * 2)
    print_current = str(current).zfill(total_digits)
    print_total = str(total).zfill(total_digits)
    progress = f'{print_current}|{print_total}'
    filledLength = int(length * (current / total))
    bar = fill * filledLength + '-' * (length - filledLength)
    end_pad = ' ' * (37 - len(file_name))
    print(f'\r   <{bar}>   [ {progress} ] Completed{ws}  ({file_name}){end_pad}', end='\r')
    # Print New Line on Complete
    if current == total:
        print()

def paths_to_wavs(paths):
    wavs = []
    for path in paths:
        full_path = os.path.realpath(path)
        if os.path.isdir(full_path):
            wavs.extend(list(map(lambda x : os.path.join(full_path, x), os.listdir(full_path))))
        elif os.path.isfile(full_path):
            wavs.append(full_path)
    return wavs

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c',
                        '--checkpoint',
                        type=str,
                        required=True,
                        help="Path to checkpoint file.")
    parser.add_argument('-i',
                        '--input',
                        action="extend",
                        nargs="+",
                        type=str,
                        help="Paths of files or folders to upscale.")
    parser.add_argument('--device',
                        type=str,
                        default='cuda',
                        required=False,
                        help="Device to use with pytorch, 'cuda' or 'cpu'.")
    parser.add_argument('-o',
                        '--out',
                        type=str,
                        required=False,
                        help="Folder to output upscaled WAVs.")
    args = parser.parse_args()
    wav_files = paths_to_wavs(args.input)
    main(args.checkpoint, wav_files, device=args.device, result_dir=args.out)
