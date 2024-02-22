# ASR
from deepgram import DeepgramClient, PrerecordedOptions
# go here and get an API key https://console.deepgram.com/signup
# The API key we created in step 3
DEEPGRAM_API_KEY = '9025aaf8281bd33ea44bc02d093247c2949d0b00'

# Hosted sample file
file_path_out = '/home/bari/workspace/spring_winter_school/speaker_extraction_workshop/outputs/y_ckpt.wav'
file_mix='/home/bari/workspace/spring_winter_school/speaker_extraction_workshop/speaker_extraction/outputs/mono_s1/mix.wav'
deepgram = DeepgramClient(DEEPGRAM_API_KEY)

def transcribe_wav(wav_path):
    with open(wav_path, 'rb') as buffer_data:
        payload = { 'buffer': buffer_data }

        options = PrerecordedOptions(
            smart_format=True, model="base", language="en-US"
        )
        response = deepgram.listen.prerecorded.v('1').transcribe_file(payload, options)
        trns = response.results.channels[0].alternatives[0].transcript
    return trns

import torch,torchaudio
import IPython.display as ipd

def display_audio(samples,sr):
    if torch.is_tensor(samples):
        samples= samples.detach().numpy()
    ipd.display(ipd.Audio(samples,rate=sr))

import numpy as np

def pad_to_length(x,length):
    if torch.is_tensor(x):
        x = norm(torch.cat((x.squeeze(),torch.zeros(length-x.shape[-1]))))
    elif isinstance(x,(np.ndarray,np.generic)):
        x = norm(np.concatenate((x,np.zeros(length-x.shape[-1]))))
    return x

def norm(samples):
    return 0.9*samples/max(abs(samples))

def mix(wav1,wav2,sir=0): #mixes two audio signals with sir
    #check lengths
    max_len = max(wav1.shape[-1],wav2.shape[-1])
    wav1,wav2 = norm(pad_to_length(wav1,max_len)),norm(pad_to_length(wav2,max_len))
    if torch.is_tensor(wav1) and torch.is_tensor(wav2):
        G =torch.sqrt(10 ** (-sir / 10) * torch.std(wav1) ** 2 / torch.std(wav2) ** 2)
    elif isinstance(wav1,(np.ndarray,np.generic)) and isinstance(wav2,(np.ndarray,np.generic)):
        G =np.sqrt(10 ** (-sir / 10) * np.std(wav1) ** 2 / np.std(wav2) ** 2)
    wav1 += G*wav2
    return norm(wav1).squeeze()