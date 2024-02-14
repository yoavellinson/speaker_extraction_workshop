from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))
import os
from data import  CreateFeatures_specific_sig,CreateFeatures_specific_sig_vec
# from utils import save_wave
import torch
from torch.utils.data import DataLoader
from model_def import Extraction_Model

from omegaconf import OmegaConf

class Extractor:
    def __init__(self,parent_dir=Path(__file__).parent.parent) -> None:
        self.parent_dir = Path(parent_dir)
        self.ckpt_path = self.parent_dir/'epoch=374,val_loss=-12.62.pth'
        self.save_dir = self.parent_dir.parent/'outputs' # dir of the signal
        if not self.save_dir.is_dir():
            self.save_dir.mkdir()
        self.hp = OmegaConf.load(str(self.parent_dir/'extraction_model/config.yaml'))
        self.load_model()

    def load_model(self):
        self.model = Extraction_Model(self.hp)
        self.model.load_state_dict(torch.load(self.ckpt_path))
        self.model.eval()

    @torch.no_grad()
    def extract_embedding(self,path_ref):
        self.model.hp.return_emb = True
        test_set = CreateFeatures_specific_sig(
                        self.hp,path_ref,path_ref, 1, train_mode=False)
        testloader = DataLoader(test_set, batch_size=1, shuffle=False,
                                num_workers=self.hp.dataloader.num_workers, pin_memory=self.hp.dataloader.pin_memory)
        
        for (mixs,  ref1) in testloader: # mix: list[0-5,5-10,10-15,...]  ref: tensor
            i=0
            for mix in mixs:
                # mix/ref1 dims [1,2,129,-1]  -> [1,real-imaginary,frequncey,frames]
                embeds = self.model.forward(mix, ref1) #mix and ref1 same size
        self.model.hp.return_emb = False
        return embeds
    
    @torch.no_grad()        
    def extract_wave(self,path_mix,path_ref,save_dir=''):
        if save_dir == '':
            save_dir=self.save_dir
        test_set = CreateFeatures_specific_sig(
                        self.hp,path_mix,path_ref, 1, train_mode=False)
        testloader = DataLoader(test_set, batch_size=1, shuffle=False,
                                num_workers=self.hp.dataloader.num_workers, pin_memory=self.hp.dataloader.pin_memory)
        # return testloader
        for (mixs,  ref1) in testloader: # mix: list[0-5,5-10,10-15,...]  ref: tensor
            i=0
            for mix in mixs:
                # mix/ref1 dims [1,2,129,-1]  -> [1,real-imaginary,frequncey,frames]
                Y_outputs,_, _, _ = self.model.forward(mix, ref1) #mix and ref1 same size
                y1_curr =  self.post_processing(Y_outputs)
        
                y1 = y1_curr if i==0 else torch.cat((y1,y1_curr),0)
                i +=1

            # ======== save results ========= # 
            return y1.detach()
            
    @torch.no_grad()    
    def extract_vec(self,mix,sr_mix,ref,sr_ref):
        test_set = CreateFeatures_specific_sig_vec(
                        self.hp,mix,ref,sr_mix,sr_ref ,1,train_mode=False)
        testloader = DataLoader(test_set, batch_size=1, shuffle=False,
                                num_workers=self.hp.dataloader.num_workers, pin_memory=self.hp.dataloader.pin_memory)
        # return testloader
        out_shape = int(mix.shape[-1]*(8000/sr_mix))
        for (mixs,  ref1) in testloader: # mix: list[0-5,5-10,10-15,...]  ref: tensor
            i=0
            for mix in mixs:
                # mix/ref1 dims [1,2,129,-1]  -> [1,real-imaginary,frequncey,frames]
                Y_outputs,_, _, _ = self.model.forward(mix, ref1) #mix and ref1 same size
                y1_curr =  self.post_processing(Y_outputs)
        
                y1 = y1_curr if i==0 else torch.cat((y1,y1_curr),0)
                i +=1

            return y1.detach()[:out_shape]
        
    @torch.no_grad()
    def post_processing(self,Y_outputs):
        Y_output  = Y_outputs[-1]
        Y_com1 = Y_output[0,:,:] + 1j*Y_output[1,:,:]
        y1_curr = torch.istft(Y_com1, n_fft=self.hp.stft.fft_length,hop_length=self.hp.stft.fft_hop,window=torch.hamming_window(self.hp.stft.fft_length))
        return y1_curr

# if __name__ == "__main__":
#     import torchaudio
#     import numpy as np
#     def norm(samples):
#         return 0.9*samples/max(abs(samples))

#     def mix(wav1,wav2,sir=0): #mixes two audio signals with sir
#         #check lengths
#         max_len = max(wav1.shape[-1],wav2.shape[-1])

#         if torch.is_tensor(wav1) and torch.is_tensor(wav2):
#             wav1 = torch.cat((wav1.squeeze(),torch.zeros(max_len-wav1.shape[-1])))
#             wav2 = torch.cat((wav2.squeeze(),torch.zeros(max_len-wav2.shape[-1])))
#             G =torch.sqrt(10 ** (-sir / 10) * torch.std(wav1) ** 2 / torch.std(wav2) ** 2)

#         #np
#         elif isinstance(wav1,(np.ndarray,np.generic)) and isinstance(wav2,(np.ndarray,np.generic)):
#             wav1 = np.concatenate((wav1,np.zeros(max_len-wav1.shape[-1])))
#             wav2 = np.concatenate((wav2,np.zeros(max_len-wav2.shape[-1])))
#             G =np.sqrt(10 ** (-sir / 10) * np.std(wav1) ** 2 / np.std(wav2) ** 2)

#         wav1 += G*wav2
#         return norm(wav1).squeeze()
#     e = Extractor()
#     f_1,sr = torchaudio.load('/home/bari/workspace/spring_winter_school/speaker_extraction_workshop/audio_samples/females/29095/4970-29095-0034.wav')
#     m1,sr = torchaudio.load('/home/bari/workspace/spring_winter_school/speaker_extraction_workshop/audio_samples/males/70968/61-70968-0027.wav')
#     mixed = mix(f_1,m1)
#     ref_female,sr = torchaudio.load('/home/bari/workspace/spring_winter_school/speaker_extraction_workshop/audio_samples/males/70968/61-70968-0027.wav')
#     y_hat = e.extract_vec(mix =mixed.unsqueeze(0),sr_mix=sr,ref=ref_female,sr_ref=sr)