import torch
from torch.utils.data import DataLoader, Dataset
import torchaudio
import sys
import torchaudio.functional as F
eps = torch.exp(torch.tensor(-6))

def mix_process(self, mix):
    Mix = torch.stft(torch.squeeze(mix),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, return_complex=True,window=torch.ones(self.hp.stft.fft_length))
    Mix[0:2, :] = Mix[0:2, :] * 0.001
    ######################normalize mix and target#####################################################
    mx_mix = torch.max(torch.max(torch.abs(torch.real(Mix)) , torch.max(torch.abs(torch.imag(Mix)) )))
    Mix = Mix/mx_mix

    Mix_input = Mix
    Mix_input = torch.unsqueeze(Mix_input,0)
    Mix_input = torch.cat( (torch.real(Mix_input),torch.imag(Mix_input)),0)

    return Mix_input

def ref_process(self,ref):
    
    Ref = torch.stft(torch.squeeze(ref),n_fft=self.hp.stft.fft_length, hop_length=self.hp.stft.fft_hop, return_complex=True,window=torch.ones(self.hp.stft.fft_length))
    Ref[0:2, :] = Ref[0:2, :] * 0.001
    Ref = Ref[:,5:-5]


            ######
    mn, mx = min(Ref.shape[1], 313), max(Ref.shape[1], 313)
        
    if mn == Ref.shape[1]:
                    repeat = int(mx//mn)
                    remain = mx % mn
                    Ref_padded = Ref.repeat(1, repeat)
                    Ref = torch.cat(
                        (Ref_padded, Ref[:, :remain]), dim=1)
    else:
                    Ref = Ref[:, :mn]

        
    mx_ref = torch.max(torch.max(torch.abs(torch.real(Ref)) , torch.max(torch.abs(torch.imag(Ref))) ))
    Ref = Ref/mx_ref
    Ref_input = Ref
    Ref_input = torch.unsqueeze(Ref_input,0)
    Ref_input = torch.cat( (torch.real(Ref_input),torch.imag(Ref_input)),0)

    return Ref_input
     



class CreateFeatures_specific_sig(Dataset):
    def __init__(self, hp, path_mix,path_ref, batch_size, train_mode):
        super().__init__()
        self.hp = hp
        self.path_mix = path_mix
        self.path_ref = path_ref
        self.batch_size = batch_size
        # self.listdir = [data_dir]


    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
       
        mix, fs_m = torchaudio.load(self.path_mix)
        ref, fs_r = torchaudio.load(self.path_ref)

        if mix.shape[0] > 1:
            raise Exception("Sorry, no multi channel allowed") 
       

          
        mix,ref = mix.squeeze(),ref.squeeze()

        if fs_m != 8000:
            mix = F.resample(mix, fs_m, 8000)
        if fs_r != 8000:
            ref = F.resample(ref, fs_r, 8000)
   
        tens = 8000 * 5 #seconds. equal to 313 frames in the STFT domain

        Mix_input = []

        if mix.shape[-1] > tens:
            repeat = int(mix.shape[-1]//tens)
            remain = tens - (mix.shape[-1] % tens)


            for i in range(repeat+1):
                mix_curr = mix[i*tens:(i+1)*tens]
                if i==repeat:
                    mix_curr = torch.cat((mix_curr,torch.zeros(remain)),0)


                if i==0:
                    Mix_inputi = mix_process(self,mix_curr)
                    Ref_input1  =  ref_process(self,ref)
                else:
                    Mix_inputi = mix_process(self,mix_curr)
                Mix_input.append(Mix_inputi)

        else:
            remain = tens - (mix.shape[-1] % tens)
            mix = torch.cat((mix,torch.zeros(remain)),0)

            Mix_inputi = mix_process(self,mix)
            Ref_input1  =  ref_process(self,ref)
            Mix_input.append(Mix_inputi)

        return Mix_input,  Ref_input1
    
    
class CreateFeatures_specific_sig_vec(Dataset):
    def __init__(self, hp, mix,ref,sr_mix,sr_ref,batch_size, train_mode):
        super().__init__()
        self.hp = hp
        self.mix = mix
        self.ref = ref
        self.batch_size = batch_size
        self.sr_mix = sr_mix
        self.sr_ref = sr_ref
        # self.listdir = [data_dir]


    def __len__(self):
        return 1
    
    def __getitem__(self, idx):

        mix = self.mix
        ref = self.ref
        if mix.shape[0] > 1:
            raise Exception("Sorry, no multi channel allowed") 
       

          
        mix,ref = mix.squeeze(),ref.squeeze()

        if self.sr_mix != 8000:
            mix = F.resample(mix, self.sr_mix, 8000)
        if self.sr_ref != 8000:
            ref = F.resample(ref, self.sr_ref, 8000)
        tens = 8000 * 5 #seconds. equal to 313 frames in the STFT domain

        Mix_input = []

        if mix.shape[-1] > tens:
            repeat = int(mix.shape[-1]//tens)
            remain = tens - (mix.shape[-1] % tens)

            for i in range(repeat+1):
                mix_curr = mix[i*tens:(i+1)*tens]
                if i==repeat:
                    mix_curr = torch.cat((mix_curr,torch.zeros(remain)),0)


                if i==0:
                    Mix_inputi = mix_process(self,mix_curr)
                    Ref_input1  =  ref_process(self,ref)
                else:
                    Mix_inputi = mix_process(self,mix_curr)
                Mix_input.append(Mix_inputi)

        else:
            remain = tens - (mix.shape[-1] % tens)
            mix = torch.cat((mix,torch.zeros(remain)),0)

            Mix_inputi = mix_process(self,mix)
            Ref_input1  =  ref_process(self,ref)
            Mix_input.append(Mix_inputi)

        return Mix_input,  Ref_input1
