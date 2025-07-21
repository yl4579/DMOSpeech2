# part of the code is borrowed from https://github.com/lawlict/ECAPA-TDNN

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as trans
from ctcmodel import ConformerCTC
# from ctcmodel_nopool import ConformerCTC as ConformerCTCNoPool
from pathlib import Path

''' Res2Conv1d + BatchNorm1d + ReLU
'''


class Res2Conv1dReluBn(nn.Module):
    '''
    in_channels == out_channels == channels
    '''

    def __init__(self, channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True, scale=4):
        super().__init__()
        assert channels % scale == 0, "{} % {} != 0".format(channels, scale)
        self.scale = scale
        self.width = channels // scale
        self.nums = scale if scale == 1 else scale - 1

        self.convs = []
        self.bns = []
        for i in range(self.nums):
            self.convs.append(nn.Conv1d(self.width, self.width, kernel_size, stride, padding, dilation, bias=bias))
            self.bns.append(nn.BatchNorm1d(self.width))
        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)

    def forward(self, x):
        out = []
        spx = torch.split(x, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            # Order: conv -> relu -> bn
            sp = self.convs[i](sp)
            sp = self.bns[i](F.relu(sp))
            out.append(sp)
        if self.scale != 1:
            out.append(spx[self.nums])
        out = torch.cat(out, dim=1)

        return out


''' Conv1d + BatchNorm1d + ReLU
'''


class Conv1dReluBn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.relu(self.conv(x)))


''' The SE connection of 1D case.
'''


class SE_Connect(nn.Module):
    def __init__(self, channels, se_bottleneck_dim=128):
        super().__init__()
        self.linear1 = nn.Linear(channels, se_bottleneck_dim)
        self.linear2 = nn.Linear(se_bottleneck_dim, channels)

    def forward(self, x):
        out = x.mean(dim=2)
        out = F.relu(self.linear1(out))
        out = torch.sigmoid(self.linear2(out))
        out = x * out.unsqueeze(2)

        return out


''' SE-Res2Block of the ECAPA-TDNN architecture.
'''

class SE_Res2Block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, scale, se_bottleneck_dim):
        super().__init__()
        self.Conv1dReluBn1 = Conv1dReluBn(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.Res2Conv1dReluBn = Res2Conv1dReluBn(out_channels, kernel_size, stride, padding, dilation, scale=scale)
        self.Conv1dReluBn2 = Conv1dReluBn(out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.SE_Connect = SE_Connect(out_channels, se_bottleneck_dim)

        self.shortcut = None
        if in_channels != out_channels:
            self.shortcut = nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
            )

    def forward(self, x):
        residual = x
        if self.shortcut:
            residual = self.shortcut(x)

        x = self.Conv1dReluBn1(x)
        x = self.Res2Conv1dReluBn(x)
        x = self.Conv1dReluBn2(x)
        x = self.SE_Connect(x)

        return x + residual


''' Attentive weighted mean and standard deviation pooling.
'''

class AttentiveStatsPool(nn.Module):
    def __init__(self, in_dim, attention_channels=128, global_context_att=False):
        super().__init__()
        self.global_context_att = global_context_att

        # Use Conv1d with stride == 1 rather than Linear, then we don't need to transpose inputs.
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, attention_channels, kernel_size=1)  # equals W and b in the paper
        else:
            self.linear1 = nn.Conv1d(in_dim, attention_channels, kernel_size=1)  # equals W and b in the paper
        self.linear2 = nn.Conv1d(attention_channels, in_dim, kernel_size=1)  # equals V and k in the paper

    def forward(self, x):

        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x

        # DON'T use ReLU here! In experiments, I find ReLU hard to converge.
        alpha = torch.tanh(self.linear1(x_in))
        # alpha = F.relu(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        residuals = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(residuals.clamp(min=1e-9))
        return torch.cat([mean, std], dim=1)


class ECAPA_TDNN(nn.Module):
    def __init__(self, channels=512, emb_dim=512, 
        global_context_att=False, use_fp16=True,
        ctc_cls=ConformerCTC,
        ctc_path='/data4/F5TTS/ckpts/F5TTS_norm_ASR_vocos_pinyin_Emilia_ZH_EN/model_last.pt',
        ctc_args={'vocab_size': 2545, 'mel_dim': 100, 'num_heads': 8, 'd_hid': 512, 'nlayers': 6},
        ctc_no_grad=False
    ):
        super().__init__()
        if ctc_path != None:
            ctc_path = Path(ctc_path)
            model = ctc_cls(**ctc_args)
            state_dict = torch.load(ctc_path, map_location='cpu')
            model.load_state_dict(state_dict['model_state_dict'])
            print(f"Initialized pretrained ConformerCTC backbone from {ctc_path}.")
        else:
            raise ValueError(ctc_path)

        self.ctc_model = model
        self.ctc_model.out.requires_grad_(False)
    
        if ctc_cls == ConformerCTC:
            self.feat_num = ctc_args['nlayers'] + 2 + 1
        # elif ctc_cls == ConformerCTCNoPool:
        #     self.feat_num = ctc_args['nlayers'] + 1
        else:
            raise ValueError(ctc_cls)
        feat_dim = ctc_args['d_hid']

        self.emb_dim = emb_dim
        
        self.feature_weight = nn.Parameter(torch.zeros(self.feat_num))
        self.instance_norm = nn.InstanceNorm1d(feat_dim)

        # self.channels = [channels] * 4 + [channels * 3]
        self.channels = [channels] * 4 + [1536]

        self.layer1 = Conv1dReluBn(feat_dim, self.channels[0], kernel_size=5, padding=2)
        self.layer2 = SE_Res2Block(self.channels[0], self.channels[1], kernel_size=3, stride=1, padding=2, dilation=2, scale=8, se_bottleneck_dim=128)
        self.layer3 = SE_Res2Block(self.channels[1], self.channels[2], kernel_size=3, stride=1, padding=3, dilation=3, scale=8, se_bottleneck_dim=128)
        self.layer4 = SE_Res2Block(self.channels[2], self.channels[3], kernel_size=3, stride=1, padding=4, dilation=4, scale=8, se_bottleneck_dim=128)

        # self.conv = nn.Conv1d(self.channels[-1], self.channels[-1], kernel_size=1)
        cat_channels = channels * 3
        self.conv = nn.Conv1d(cat_channels, self.channels[-1], kernel_size=1)
        self.pooling = AttentiveStatsPool(self.channels[-1], attention_channels=128, global_context_att=global_context_att)
        self.bn = nn.BatchNorm1d(self.channels[-1] * 2)
        self.linear = nn.Linear(self.channels[-1] * 2, emb_dim)

        if ctc_no_grad:
            for param in self.ctc_model.parameters():
                param.requires_grad = False
            self.ctc_model = self.ctc_model.eval()
        else:
            self.ctc_model = self.ctc_model.train()
        self.ctc_no_grad = ctc_no_grad
        print('ctc_no_grad: ', self.ctc_no_grad)

    def forward(self, latent, input_lengths,  return_asr=False):
        if self.ctc_no_grad:
            with torch.no_grad():
                asr, h = self.ctc_model(latent, input_lengths)
        else:
            asr, h = self.ctc_model(latent, input_lengths)
        
        x = torch.stack(h, dim=0)
        norm_weights = F.softmax(self.feature_weight, dim=-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        x = (norm_weights * x).sum(dim=0)
        x = x + 1e-6
        # x = torch.transpose(x, 1, 2) + 1e-6
            
        x = self.instance_norm(x)
        # x = torch.transpose(x, 1, 2)

        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        out3 = self.layer3(out2)
        out4 = self.layer4(out3)

        out = torch.cat([out2, out3, out4], dim=1)
        out = F.relu(self.conv(out))
        out = self.bn(self.pooling(out))
        out = self.linear(out)

        if return_asr:
            return out, asr
        return out

if __name__ == "__main__":
    from diffspeech.ldm.model import DiT
    from diffspeech.data.collate import get_mask_from_lengths
    from diffspeech.tools.text.vocab import IPA

    bsz = 3

    # Sample ipa
    ipa_lens = torch.randint(10, 50, (bsz,)).cuda()
    ipa_mask = get_mask_from_lengths(ipa_lens).cuda()
    ipa = torch.randint(0, len(IPA.vocab), (bsz, ipa_mask.size(-1))).cuda()

    # Sample latent
    latent_lens = torch.randint(50, 250, (bsz,)).cuda()
    latent_mask = get_mask_from_lengths(latent_lens).cuda()
    latent = torch.randn(bsz, latent_mask.size(-1), 64).cuda()

    # Sample prompt
    prompt_mask = get_mask_from_lengths(
        (latent_lens * 0.25).long(), max_len=latent_mask.size(-1)
    ).cuda()
    prompt_latent = latent * prompt_mask.unsqueeze(-1)

    model = ECAPA_TDNN(emb_dim=512).cuda()

    emb = model(latent, latent_mask.sum(axis=-1))

    print(emb.shape)