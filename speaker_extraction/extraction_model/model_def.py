from utils_classes import Encoder ,  Bottleneck,Encoder_ref,Decoder_ayal
import torch.nn as nn
import torch
import logging
import pytorch_lightning as pl
logging.getLogger('lightning').setLevel(logging.WARNING)


eps = torch.exp(torch.tensor(-6))


class Pl_module(pl.LightningModule):
    def __init__(self, model,hp):
        super().__init__()
        self.model = model
        self.hp = hp
      
    def forward(self, mix,ref=None,tar=None,separation=False):
        y = self.model(mix,ref) 
        return y
 

### =================== Three_Stages ==================== ###
class Extraction_Model(nn.Module): 
    def __init__(self, hp):
        super().__init__()
        self.hp_sep_conf = hp.copy() 
        self.hp_sep_conf.model_def_name = 'First_Stage'
        self.hp_sep_conf.insert_ref,self.hp_sep_conf.insert_mix , self.hp_sep_conf.get_full_emb_ref= False,False,False
        
        model_def = First_Stage(self.hp_sep_conf)
        self.model_mag = Pl_module(model_def,self.hp_sep_conf)

        self.hp = hp
        
        kernel_size=(4,3)
        stride=(2,1)

        self.ic = 2  
        oc = 2  
        ngf = hp.unet.num_filters
        d_model=512

        self.encoder = Encoder(hp, self.ic, ngf,kernel_size,stride=stride)
        
        self.bottleneck_proccess = Bottleneck(hp,d_model,nhead=8,isref=hp.insert_ref,ismix = hp.insert_mix)
        
        decoder_ic = 1024 if (hp.insert_ref and hp.bottleneck_op=='concat') else 512
        self.decoder = Decoder_ayal(hp,  kernel_size,stride=stride,ic=decoder_ic)

        encoder_layer = nn.TransformerEncoderLayer(d_model=258, nhead=6,batch_first=True)
        num_sa_layers_decoder = 6 if hp.full_decoder_self_att else 1
        self.self_attention_out = nn.TransformerEncoder(encoder_layer,num_layers=num_sa_layers_decoder)
        self.postconv = nn.Conv2d(2, oc, kernel_size=3, stride=1, padding=1)

    def forward(self, mix, ref,batch_idx=None):
        if not self.hp.triplet_loss:
            outputs_model_mag,emb_ref_first_stage = self.model_mag(mix,ref)
        else:
            outputs_model_mag,emb_ref_first_stage,emb_output_first_stage = self.model_mag(mix,ref)

        if self.hp.return_emb:
            return emb_output_first_stage
        
        output_model_mag = outputs_model_mag[0]
        if self.hp.test or mix.shape[0]==1:
            output_model_mag = output_model_mag.unsqueeze(0)


        outputs = []
        for i in range(self.hp.iterations_num_phase2):

            input_encoder = output_model_mag if i==0 else output

            conv1feature, conv2feature, conv3feature, conv4feature, emb  = self.encoder(input_encoder)
            if not self.hp.insert_ref and not  self.hp.insert_mix:
                bottleneck,_ = self.bottleneck_proccess(emb)
            elif self.hp.insert_ref and not  self.hp.insert_mix:
                if not self.hp.get_full_emb_ref:
                    _, _, _, _, emb_ref  = self.encoder(ref)
                else:
                    emb_ref = emb_ref_first_stage
                bottleneck,_ = self.bottleneck_proccess(emb,emb_ref)
            elif  self.hp.insert_ref and   self.hp.insert_mix:
                _, _, _, _, emb_mix  = self.encoder(mix)
                if not self.hp.get_full_emb_ref:
                    _, _, _, _, emb_ref  = self.encoder(ref)
                else:
                    emb_ref = emb_ref_first_stage
                bottleneck,_ = self.bottleneck_proccess(emb,emb_ref,emb_mix)
    
                    
            # decoder
            output = self.decoder(bottleneck=bottleneck, conv1feature=conv1feature, conv2feature=conv2feature,
                                            conv3feature=conv3feature,conv4feature=conv4feature)

            # self attention


            output = torch.flatten(output,1,2).permute(0,2,1) #[B,T,Features]
            self.self_attention_out(output)
            output_sa = output.permute(0,2,1).view(output.shape[0],2,129,output.shape[1])
            output = self.postconv(output_sa)

            outputs.append(torch.squeeze(output))
        if not self.hp.triplet_loss:
            return outputs,outputs_model_mag
        else:
            return outputs,outputs_model_mag, torch.squeeze(emb_ref_first_stage),torch.squeeze(emb_output_first_stage)


### =================== First Stage==================== ###
class First_Stage(nn.Module): 
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        
        kernel_size=(4,3)
        stride=(2,1)
        self.ic = 2  if  hp.features=='real_imag' else 1
        self.oc = 2  if   hp.features=='real_imag' else 1
        ngf = hp.unet.num_filters
        d_model=512

        self.encoder = Encoder(hp, self.ic, ngf,kernel_size,stride=stride)
        if not hp.same_encoder:
            ic_ref =  2  if   hp.ref_features=='real_imag' else 1
            self.encoder_ref = Encoder(hp, ic_ref, ngf,kernel_size,stride=stride) if not self.hp.small_encoder_ref else Encoder_ref(hp, ic_ref, kernel_size,stride=stride)
        self.bottleneck_proccess = Bottleneck(hp,d_model,nhead=8,isref= True)
        ic_decoder = 1024 if self.hp.bottleneck_op=='concat' else 512
        self.decoder = Decoder_ayal(hp,  kernel_size,stride=stride,ic=ic_decoder)

        d_model_out=258 
        nhead_out = 6
     
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model_out, nhead=nhead_out,batch_first=True)
        num_sa_layers_decoder = 6 if hp.full_decoder_self_att else 1
        self.self_attention_out = nn.TransformerEncoder(encoder_layer,num_layers=num_sa_layers_decoder)
        self.postconv = nn.Conv2d(self.oc, self.oc, kernel_size=3, stride=1, padding=1)
    
    def forward(self, mix, ref=None):
        outputs = []
        for i in range(self.hp.iterations_num):
            
            input_encoder = mix if i==0 else output
            # output0 =torch.tensor(0) if i==0 else output

            # encoder
            conv1feature, conv2feature, conv3feature, conv4feature, emb  = self.encoder(input_encoder)
            if self.hp.same_encoder:
               conv1feature_ref, conv2feature_ref, conv3feature_ref, conv4feature_ref,  emb_ref = self.encoder(ref) 
            elif not self.hp.same_encoder and not self.hp.small_encoder_ref:
                conv1feature_ref, conv2feature_ref, conv3feature_ref, conv4feature_ref,  emb_ref = self.encoder_ref(ref)  
            elif not self.hp.same_encoder and  self.hp.small_encoder_ref:
                emb_ref = self.encoder_ref(ref)  
                
            # bottleneck
            bottleneck,emb_ref_final = self.bottleneck_proccess(emb,emb_ref)

            if self.hp.ref_skip_co:
                conv1feature, conv2feature, conv3feature, conv4feature = conv1feature*conv1feature_ref, conv2feature*conv2feature_ref, conv3feature*conv3feature_ref, conv4feature*conv4feature_ref
            # decoder
            output = self.decoder(bottleneck=bottleneck, conv1feature=conv1feature, conv2feature=conv2feature,
                                    conv3feature=conv3feature,conv4feature=conv4feature)

            # post net
            output = torch.flatten(output,1,2).permute(0,2,1) #[B,T,Features]
        
            self.self_attention_out(output)

            output = output.permute(0,2,1).view(output.shape[0],self.oc,self.hp.stft.fft_length//2+1,output.shape[1])
            output = self.postconv(output)

            outputs.append(torch.squeeze(output))
        if not self.hp.triplet_loss:
            return outputs, torch.squeeze(emb_ref_final)
        else:
            _, _, _, _,  emb_output = self.encoder(output)
            emb_output_final = self.bottleneck_proccess(None,emb_output)
            return outputs, torch.squeeze(emb_ref_final) , torch.squeeze(emb_output_final)
      
