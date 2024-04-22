from transformers import ViTConfig, ViTModel, TrOCRForCausalLM, TrOCRConfig, DebertaConfig
from transformers import VisionEncoderDecoderModel, RobertaConfig, EncoderDecoderConfig, EncoderDecoderModel
from transformers import ConvNextConfig, ConvNextModel, BertConfig, VisionEncoderDecoderConfig
from transformers import GPT2Config, GPT2LMHeadModel
import torch
import torch.nn as nn
from torch import Tensor
import math
# %%
import torch
from torch import nn
import torch.nn.functional as F


class GlobalContext(nn.Module):
    def __init__(self, filters):
        super(GlobalContext, self).__init__()
        self.linear_squeeze = nn.Linear(filters, filters//8)
        self.linear_excite = nn.Linear(filters//8, filters)

    def forward(self, data):
        pool = F.avg_pool1d(data, data.shape[-1]).squeeze(-1)
        pool = F.gelu(self.linear_squeeze(pool))
        pool = torch.sigmoid(self.linear_excite(pool)).unsqueeze(-1)
        final = torch.multiply(data,pool)
        return final


class EasterUnit(nn.Module):
    def __init__(self, filters, kernel, stride, dropouts, inchannel):
        super(EasterUnit, self).__init__()
        self.conv1 = nn.Conv1d(inchannel, filters, kernel_size=1, stride=1, padding='same')
        self.batchnorm1 = nn.BatchNorm1d(filters, eps=1e-5, momentum=0.997)
        self.conv2 = nn.Conv1d(inchannel, filters, kernel_size=1, stride=1, padding='same')
        self.batchnorm2 = nn.BatchNorm1d(filters, eps=1e-5, momentum=0.997)
        # First Block
        self.conv3 = nn.Conv1d(inchannel, filters, kernel_size=kernel, stride=stride, padding='same')
        self.batchnorm3 = nn.BatchNorm1d(filters, eps=1e-5, momentum=0.997)
        # Second Block
        self.conv4 = nn.Conv1d(filters, filters, kernel_size=kernel, stride=stride, padding='same')
        self.batchnorm4 = nn.BatchNorm1d(filters, eps=1e-5, momentum=0.997)
        # Third Block
        self.conv5 = nn.Conv1d(filters, filters, kernel_size=kernel, stride=stride, padding='same')
        self.batchnorm5 = nn.BatchNorm1d(filters, eps=1e-5, momentum=0.997)
        self.global_context = GlobalContext(filters)
        self.dropout = dropouts
        self.dropout_layer = nn.Dropout(self.dropout)
    
    def forward(self, old, data):

        old = self.batchnorm1(self.conv1(old))
        this = self.batchnorm2(self.conv2(data))
        old = torch.add(old, this)
        data = self.dropout_layer(F.gelu(self.batchnorm3(self.conv3(data))))
        data = self.dropout_layer(F.gelu(self.batchnorm4(self.conv4(data))))
        data = self.batchnorm5(self.conv5(data))
        data = self.global_context(data)
        final = torch.add(old, data)
        data = self.dropout_layer(F.gelu(final))
        return data, old


class Easter2(nn.Module):
    def __init__(self, inchannel, vocab_size):
        super(Easter2, self).__init__()
        self.conv1 = nn.Conv1d(inchannel, 128, kernel_size=3, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.997)


        self.conv2 = nn.Conv1d(128, 128, kernel_size=3, stride=2, padding=1)
        self.batchnorm2 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.997)

        self.easter_unit1 = EasterUnit(256, 5, 1, 0.2, 128)
        self.easter_unit2 = EasterUnit(256, 7, 1, 0.2, 256)
        self.easter_unit3 = EasterUnit(256, 9, 1, 0.3, 256)

        self.conv3 = nn.Conv1d(256, 512, kernel_size=11, stride=1, padding='same', dilation=2)
        self.batchnorm3 = nn.BatchNorm1d(512, eps=1e-5, momentum=0.997)

        self.conv4 = nn.Conv1d(512, 512, kernel_size=1, stride=1, padding='same')
        self.batchnorm4 = nn.BatchNorm1d(512, eps=1e-5, momentum=0.997)

        self.conv5 = nn.Conv1d(512, vocab_size, kernel_size=1, stride=1, padding='same')
        self.dropout1d_first = nn.Dropout(0.2)
        self.dropout1d = nn.Dropout(0.4)
    def forward(self, input_data):
        data = self.dropout1d_first(F.gelu(self.batchnorm1(self.conv1(input_data))))
        data = self.dropout1d_first(F.gelu(self.batchnorm2(self.conv2(data))))
        
        old = data

        data, old = self.easter_unit1(old, data)
        data, old = self.easter_unit2(old, data)
        data, old = self.easter_unit3(old, data)

        data = self.dropout1d(F.gelu(self.batchnorm3(self.conv3(data))))
        data = self.dropout1d(F.gelu(self.batchnorm4(self.conv4(data))))
        data = self.conv5(data)
        data = F.max_pool1d(data, 2, 2).permute(0, 2, 1)

        return data



def model_ved(vocab_size):
    enc_config = ViTConfig(hidden_size=256,
                           num_hidden_layers=3,
                           num_attention_heads=8,
                           intermediate_size=1024,
                           hidden_dropout_prob=0.1)
    dec_config = RobertaConfig(vocab_size=vocab_size, 
                               num_hidden_layers=3, 
                               hidden_size=256, 
                               num_attention_heads=8,
                               intermediate_size=1024,
                               hidden_dropout_prob=0.1)

    config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    model = VisionEncoderDecoderModel(config=config)
    return model



def model_conv_transformer(inchannel, vocab_size):

    class Conv(nn.Module):
        def __init__(self):
            super(Conv, self).__init__()            # 512 * 64

            self.conv1 = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1),
                nn.BatchNorm2d(16),
                nn.GELU(),
                nn.MaxPool2d(2, 2)      #    256 * 32
                # nn.MaxPool2d((2, 4), (2, 4)),   # 128 * 8
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.BatchNorm2d(32),
                nn.GELU(),
                nn.MaxPool2d(2, 2)    # 128 * 16
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(32, 48, 3, padding=1),
                nn.BatchNorm2d(48),
                nn.GELU(),
                nn.Conv2d(48, 64, 3, padding=1),
                nn.BatchNorm2d(64),
                nn.GELU(),
                nn.MaxPool2d((1, 2), (1, 2)),   # 128 * 8
                nn.Dropout2d(0.2),
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(64, 96, 3, padding=1),
                nn.BatchNorm2d(96),
                nn.GELU(),
                nn.Conv2d(96, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.GELU(),
                nn.MaxPool2d((1, 2), (1, 2)),    # 128 * 4
                nn.Dropout2d(0.2),
            )
            self.conv5 = nn.Sequential(
                nn.Conv2d(128, 256, 4),
                nn.BatchNorm2d(256),
                nn.GELU(),
            )  

        def forward(self,
                src: Tensor,
               ):

            src = self.conv1(src)
            # print(x.shape)                                 # (*, 16, 32, 256)
            src = self.conv2(src)
            # print(x.shape)                                 # (*, 32, 16, 128)
            src = self.conv3(src)
            # print(x.shape)                                 # (*, 64, 8, 128)
            src = self.conv4(src)
            # print(x.shape)                                 # (*, 128, 4, 128)        
            src = self.conv5(src)
            # print(x.shape)                                 # (*, 256, 1, 125)
            src = src.squeeze(-1)
            src = src.permute((0, 2, 1)).contiguous()        # (*, 125, 256)

            return src
        

    model_conv = Conv()
    # model_conv = Easter2(inchannel, 256)
    

    dec = {'vocab_size':vocab_size,
           'n_positions':512,
           'n_embd':256,
           'n_head':4,
           'n_layer':2
           }

    enc = {'vocab_size':vocab_size,
           'num_hidden_layers':2,
           'hidden_size':256,
           'num_attention_heads':4,
           'intermediate_size':1024,
           'hidden_act':'gelu'
           }

    enc_config = RobertaConfig(**enc)

    # dec_config = RobertaConfig(**enc)
    dec_config = GPT2Config(**dec)

    # dec_config = RobertaConfig(vocab_size=vocab_size, hidden_size=256, num_attention_heads=8)
    # enc_config = RobertaConfig(vocab_size=vocab_size, hidden_size=256, num_attention_heads=8)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    model_transformer = EncoderDecoderModel(config=config)

    return model_conv, model_transformer



def model_convnext(vocab_size):

    class Conv(nn.Module):
        def __init__(self):
            super(Conv, self).__init__()            # 48 * 48

            self.conv1 = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.gelu(),
                nn.MaxPool2d((1, 2), (1, 2))      #    48 * 24
            )
            self.conv2 = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.gelu(),
                nn.MaxPool2d((1, 2), (1, 2))    # 48 * 12
            )
            self.conv3 = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.gelu(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.gelu(),
                nn.MaxPool2d((1, 2), (1, 2)),   # 48 * 6
                nn.Dropout2d(0.2),
            )
            self.conv4 = nn.Sequential(
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.gelu(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.gelu(),
                nn.MaxPool2d((1, 2), (1, 2)),    # 48 * 3
                nn.Dropout2d(0.2),
            )
            self.conv5 = nn.Sequential(
                nn.Conv2d(256, 256, 3),
                nn.BatchNorm2d(256),
                nn.gelu(),
            )  

        def forward(self,
                src: Tensor,
               ):

            src = self.conv1(src)
            # print(x.shape)                                 # (*, 16, 32, 650)
            src = self.conv2(src)
            # print(x.shape)                                 # (*, 32, 16, 325)
            src = self.conv3(src)
            # print(x.shape)                                 # (*, 64, 8, 325)
            src = self.conv4(src)
            # print(x.shape)                                 # (*, 128, 4, 325)        
            src = self.conv5(src)
            # print(x.shape)                                 # (*, 256, 1, 322)
            src = src.squeeze(-1)
            src = src.permute((0, 2, 1)).contiguous()

            return src


        

    conv_config = ConvNextConfig(num_stages=2, hidden_sizes=[96,256])
    model_conv = ConvNextModel(conv_config)


    enc_config = RobertaConfig(vocab_size=vocab_size, num_hidden_layers=3, hidden_size=256, num_attention_heads=8)
    dec_config = RobertaConfig(vocab_size=vocab_size, num_hidden_layers=3, hidden_size=256, num_attention_heads=8)

    # dec_config = RobertaConfig(vocab_size=vocab_size, hidden_size=256, num_attention_heads=8)
    # enc_config = RobertaConfig(vocab_size=vocab_size, hidden_size=256, num_attention_heads=8)
    config = EncoderDecoderConfig.from_encoder_decoder_configs(enc_config, dec_config)
    model_transformer = EncoderDecoderModel(config=config)

    return model_conv, model_transformer



    


