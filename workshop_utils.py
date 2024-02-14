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