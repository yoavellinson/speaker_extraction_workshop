# from turtle import forward
import torch.nn as nn
import torch
from utils import padding_if_needed

eps = torch.exp(torch.tensor(-6))

### =================== encoder ==================== ###
class Encoder(nn.Module):
    def __init__(self, hp, ic, ngf, kernel_size, stride=2):
        super().__init__()

        self.hp = hp
        self.convlayer1 = self.unet_downconv(ic, ngf, kernel_size=kernel_size, stride=1,norm=hp.norm)
        self.convlayer2 = self.unet_downconv(ngf, ngf * 2, kernel_size=kernel_size, stride=stride,norm=hp.norm)
        self.convlayer3 = self.unet_downconv(ngf * 2, ngf * 4, kernel_size=kernel_size, stride=stride,norm=hp.norm)
        self.convlayer4 = self.unet_downconv(ngf * 4, ngf * 8, kernel_size=kernel_size, stride=stride,norm=hp.norm)
        self.convlayer5 = self.unet_downconv(ngf * 8, ngf * 8, kernel_size=kernel_size, stride=stride,norm=hp.norm)

    def forward(self, x):
      
        conv1feature = self.convlayer1(x)
        conv2feature = self.convlayer2(conv1feature)
        conv3feature = self.convlayer3(conv2feature)
        conv4feature = self.convlayer4(conv3feature)
        conv5feature = self.convlayer5(conv4feature)
        
   
        return conv1feature, conv2feature, conv3feature, conv4feature, conv5feature

    def unet_downconv(self, nc, output_nc, kernel_size, stride,padding=1,norm='layer',transformer_norm_first=False):
        '''
        output size: [(in_size-kernel_size+2*padding)/stride]+1
        '''
        downconv = nn.Conv2d(nc, output_nc, kernel_size, stride=stride,padding=padding, bias=False)
        if not transformer_norm_first: # if True apply layernorm before self attention
            if norm =='batch':
                downnorm = nn.BatchNorm2d(output_nc)
            elif norm =='layer':
                downnorm = GlobLN(output_nc)
        if self.hp.act_fun=='relu':
            act_fun = nn.ReLU() 
        elif self.hp.act_fun=='prelu':
            act_fun = nn.PReLU()
        return nn.Sequential(*[downconv, downnorm, act_fun])
    
### =================== decoder ==================== ###
class Decoder(nn.Module):
    def __init__(self, hp, ngf, kernel_size,stride=2, dim=1):
        super().__init__()

        self.hp = hp
        self.dim = dim
        self.oc =  2
        self.model_def_name = hp.model_def_name

        in_ch1 =  ngf * 16
        in_ch2 = ngf * 16 
        in_ch3 = ngf * 16 
        in_ch4 = ngf * 16 
        in_ch5 = ngf * 8  
        in_ch6 = ngf * 4  
        in_ch7 = ngf * 2  
        self.upconvlayer1 = self.unet_upconv(in_ch1, ngf * 8, kernel_size,stride,norm=hp.norm)
        self.upconvlayer2 = self.unet_upconv(in_ch2, ngf * 8, kernel_size,stride,norm=hp.norm)
        self.upconvlayer3 = self.unet_upconv(in_ch3, ngf * 8, kernel_size,stride,norm=hp.norm)
        self.upconvlayer4 = self.unet_upconv(in_ch4, ngf * 4, kernel_size,stride,norm=hp.norm)
        self.upconvlayer5 = self.unet_upconv(in_ch5, ngf * 2, kernel_size,stride,norm=hp.norm)
        self.upconvlayer6 = self.unet_upconv(in_ch6, ngf, kernel_size,stride,norm=hp.norm)
        self.upconvlayer7 = self.unet_upconv(in_ch7, 2,(5,3) ,stride,norm=hp.norm)  #(kernel_size+1, kernel_size)
        self.postunet = self.last_layer(2, self.oc)

     

    def forward(self, **inputs):
        bottleneck, conv1feature, conv2feature, conv3feature, conv4feature, conv5feature, conv6feature = inputs['bottleneck'], inputs[
                'conv1feature'], inputs['conv2feature'], inputs['conv3feature'], inputs['conv4feature'], inputs['conv5feature'], inputs['conv6feature']

   
        upconv1feature = self.upconvlayer1(bottleneck)

        in_layer2 =  torch.cat((padding_if_needed(upconv1feature, conv6feature)), self.dim) 
        upconv2feature = self.upconvlayer2(in_layer2)
        in_layer3 =  torch.cat((padding_if_needed(upconv2feature, conv5feature)), self.dim)
        upconv3feature = self.upconvlayer3(in_layer3)
        in_layer4 =   torch.cat((padding_if_needed(upconv3feature, conv4feature)), self.dim)
        upconv4feature = self.upconvlayer4(in_layer4)
        in_layer5 =  torch.cat((padding_if_needed(upconv4feature, conv3feature)), self.dim) 
        upconv5feature = self.upconvlayer5(in_layer5)
        in_layer6 =  torch.cat((padding_if_needed(upconv5feature, conv2feature)), self.dim)
        upconv6feature = self.upconvlayer6(in_layer6)
        in_layer7 =  torch.cat((padding_if_needed(upconv6feature, conv1feature)), self.dim)
        output = self.upconvlayer7(in_layer7)

        output = self.postunet(output)

        return output



    def unet_upconv(self, nc, output_nc, kernel_size, stride=2, padding=1,norm ='batch'):
        '''
        output_size: (in_size-1)*stride-2*padding+kernel_size
        '''
   
        upconv = nn.ConvTranspose2d(nc, output_nc, kernel_size, stride=stride, padding=padding)
        if norm =='batch':
            upnorm = nn.BatchNorm2d(output_nc)
        elif norm =='layer':
            upnorm = GlobLN(output_nc)
        return nn.Sequential(*[upconv, upnorm, nn.ReLU()])


    def last_layer(self, nc, output_nc, kernel_size=3):
        postconv1 = nn.Conv2d(
            nc, output_nc, kernel_size=kernel_size, stride=1, padding=1)

        return nn.Sequential(*[postconv1])

### =================== decoder ayal==================== ###
class Decoder_ayal(nn.Module):
    def __init__(self, hp, kernel_size,stride=2, dim=1,sa=False,ic=1024):
        super().__init__()

        self.hp = hp
        self.dim = dim
        self.model_def_name = hp.model_def_name
        self.sa=sa
        
        # ic = 1024  if (not ( hp.model_def_name=='Two_Stages' or hp.model_def_name=='Three_Stages' or hp.model_def_name=='First_Stage_Separation' ) or hp.use_emb_ref) else 512
        oc = 2 #if not hp.separation else 4
        
        self.upconvlayer1 = self.unet_upconv(ic, 512, kernel_size,stride,norm=hp.norm)
        self.upconvlayer2 = self.unet_upconv(512*2, 256, kernel_size,stride,norm=hp.norm)
        self.upconvlayer3 = self.unet_upconv(512, 128, kernel_size,stride,norm=hp.norm)
        self.upconvlayer4 = self.unet_upconv(256, 64, kernel_size,stride,norm=hp.norm)
        self.upconvlayer5 = self.unet_upconv(128,  16, (4,3),stride=1,norm=hp.norm)
        self.postunet = self.last_layer(16,oc)

  
    def forward(self, **inputs):
        bottleneck, conv1feature, conv2feature, conv3feature, conv4feature, = inputs['bottleneck'], inputs[
                'conv1feature'], inputs['conv2feature'], inputs['conv3feature'], inputs['conv4feature']

        upconv1feature = self.upconvlayer1(bottleneck)

        in_layer2 =  torch.cat((upconv1feature, conv4feature), self.dim) 
        upconv2feature = self.upconvlayer2(in_layer2)
 
        in_layer3 =  torch.cat((upconv2feature, conv3feature), self.dim)
        upconv3feature = self.upconvlayer3(in_layer3)
 
        in_layer4 =   torch.cat((upconv3feature, conv2feature), self.dim)
        upconv4feature = self.upconvlayer4(in_layer4)
   
        in_layer5 =   torch.cat((upconv4feature, conv1feature), self.dim)
        output = self.upconvlayer5(in_layer5)
     

        output = self.postunet(output)

        return output



    def unet_upconv(self, nc, output_nc, kernel_size, stride=2, padding=1,norm ='batch'):
        '''
        output_size: (in_size-1)*stride-2*padding+kernel_size
        '''
   
        upconv = nn.ConvTranspose2d(
                nc, output_nc, kernel_size, stride=stride, padding=padding)
        if norm =='batch':
            upnorm = nn.BatchNorm2d(output_nc)
        elif norm =='layer':
            upnorm = GlobLN(output_nc)
        return nn.Sequential(*[upconv, upnorm, nn.ReLU()])


    def last_layer(self, nc, output_nc, kernel_size=3):
        postconv1 = nn.Conv2d(
            nc, output_nc, kernel_size=kernel_size, stride=1, padding=1)

        return nn.Sequential(*[postconv1])


### =================== Bottleneck==================== ###
class Bottleneck(nn.Module):
    def __init__(self,hp,d_model,nhead,isref=True,ismix=False):
        super().__init__()
        self.hp=hp
        self.isref = isref
        self.ismix = ismix
        assert not ismix or isref
       
        self.fc1 = nn.Linear(8*d_model,d_model)
        self.self_attention_mix =  nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,  batch_first=True)
        if isref and  not hp.get_full_emb_ref:
            self.self_attention_ref =  nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,  batch_first=True)
            if not hp.same_encoder:
                self.fc1_ref = nn.Linear(8*d_model,d_model)

        if ismix:
            self.self_attention_mix =  nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,  batch_first=True)

        in_size = 8*d_model*2 if  (isref and self.hp.bottleneck_op=='concat') else 8*d_model
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,batch_first=True)
        self.self_attention_bn = nn.TransformerEncoder(encoder_layer,num_layers=4)
        self.fc_full_encoder_self_att2 = nn.Linear(d_model,in_size)
        
        # else:
        self.fc2 = nn.Linear(d_model,8*d_model)
  


    def forward(self,emb,emb_ref=None,emb_mix = None):
        if not emb==None:
            b,c,k,f = emb.shape
            # attention
            emb = torch.flatten(emb,1,2).permute(0,2,1) #[B,T,Features]
            emb = self.fc1(emb)
            emb = self.self_attention_mix(emb)

            if self.isref:
                if not self.hp.get_full_emb_ref:
                    emb_ref = torch.flatten(emb_ref,1,2).permute(0,2,1) 
                    emb_ref = self.fc1(emb_ref) if  self.hp.same_encoder else  self.fc1_ref(emb_ref)
                    emb_ref = self.self_attention_ref(emb_ref)
                    if  self.hp.mean_ref:
                        emb_ref = emb_ref.mean(1).unsqueeze(1)
                else:
                    emb_ref = emb_ref.unsqueeze(1) if emb_ref.ndim==2 else emb_ref
                if self.hp.bottleneck_op=='concat':
                    bottleneck =  torch.cat((emb, emb_ref), 1)
                elif self.hp.bottleneck_op=='mult':
                    bottleneck =  emb * emb_ref
                elif self.hp.bottleneck_op=='add':
                    bottleneck =  emb + emb_ref

                if self.ismix: # there is no ismix without isref
                    emb_mix = torch.flatten(emb_mix,1,2).permute(0,2,1) #[B,T,Features]
                    emb_mix = self.fc1(emb_mix) 
                    emb_mix = self.self_attention_mix(emb_mix)
                   
                    if self.hp.bottleneck_op=='concat':
                        bottleneck =  torch.cat((bottleneck, emb_mix), 1)
                    elif self.hp.bottleneck_op=='mult':
                        bottleneck =  bottleneck * emb_mix
                    elif self.hp.bottleneck_op=='add':
                        bottleneck =  bottleneck + emb_mix
            else:
                bottleneck = emb


            bottleneck =  self.self_attention_bn(bottleneck)
            bottleneck =  self.fc_full_encoder_self_att2(bottleneck)
            bottleneck = bottleneck.permute(0,2,1).reshape(b,c,k,f)

            return bottleneck,emb_ref
        else:
            emb_ref = torch.flatten(emb_ref,1,2).permute(0,2,1) 
            emb_ref = self.fc1(emb_ref) if  self.hp.same_encoder else  self.fc1_ref(emb_ref)
            emb_ref = self.self_attention_ref(emb_ref)
           
            emb_ref = emb_ref.mean(1).unsqueeze(1)
            return emb_ref
        
### =================== encoder_ref ==================== ###
class Encoder_ref(nn.Module):
    def __init__(self, hp, ic,  kernel_size,stride):
        super().__init__()

        self.hp = hp
        self.convlayer1 = self.unet_downconv(ic, 32, kernel_size=kernel_size, stride=1,norm=hp.norm)
        self.convlayer2 = self.unet_downconv(32, 64, kernel_size=kernel_size, stride=stride,norm=hp.norm)


    def forward(self, x):
      
        conv1feature = self.convlayer1(x)
        conv2feature = self.convlayer2(conv1feature)
       
        return conv2feature

    def unet_downconv(self, nc, output_nc, kernel_size, stride,padding=1,norm='layer',transformer_norm_first=False):
        '''
        output size: [(in_size-kernel_size+2*padding)/stride]+1
        '''
        downconv = nn.Conv2d(nc, output_nc, kernel_size, stride=stride,padding=padding, bias=False)
        if not transformer_norm_first: # if True apply layernorm before self attention
            if norm =='batch':
                downnorm = nn.BatchNorm2d(output_nc)
            elif norm =='layer':
                downnorm = GlobLN(output_nc)
        if self.hp.act_fun=='relu':
            act_fun = nn.ReLU() 
        elif self.hp.act_fun=='prelu':
            act_fun = nn.PReLU()
        return nn.Sequential(*[downconv, downnorm, act_fun])


############# utils #############

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class _LayerNorm(nn.Module):
    """Layer Normalization base class."""

    def __init__(self, channel_size):
        super(_LayerNorm, self).__init__()
        self.channel_size = channel_size
        self.gamma = nn.Parameter(torch.ones(channel_size),
                                  requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(channel_size),
                                 requires_grad=True)

    def apply_gain_and_bias(self, normed_x):
        """ Assumes input of size `[batch, chanel, *]`. """
        return (self.gamma * normed_x.transpose(1, -1) +
                self.beta).transpose(1, -1)


class GlobLN(_LayerNorm):
    """Global Layer Normalization (globLN)."""

    def forward(self, x):
        """ Applies forward pass.

        Works for any input size > 2D.

        Args:
            x (:class:`torch.Tensor`): Shape `[batch, chan, *]`

        Returns:
            :class:`torch.Tensor`: gLN_x `[batch, chan, *]`
        """
        dims = list(range(1, len(x.shape)))
        mean = x.mean(dim=dims, keepdim=True)
        var = torch.pow(x - mean, 2).mean(dim=dims, keepdim=True)
        return self.apply_gain_and_bias((x - mean) / (var + 1e-8).sqrt())


