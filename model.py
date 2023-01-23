import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet50_Weights
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights

import math

#============classes===================

class Sims(nn.Module):
    def __init__(self):
        super(Sims, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        '''(N, S, E)  --> (N, 1, S, S)'''
        f = x.shape[1]
        
        I = torch.ones(f).to(self.device)
        xr = torch.einsum('bfe,h->bhfe', (x, I))   #[x, x, x, x ....]  =>  xr[:,0,:,:] == x
        xc = torch.einsum('bfe,h->bfhe', (x, I))   #[x x x x ....]     =>  xc[:,:,0,:] == x
        diff = xr - xc
        out = torch.einsum('bfge,bfge->bfg', (diff, diff))
        out = out.unsqueeze(1)
        #out = self.bn(out)
        out = F.softmax(-out/13.544, dim = -1)  # temperature T=13.544
        return out

#---------------------------------------------------------------------------

class ResNet50Bottom(nn.Module):
    def __init__(self):
        super(ResNet50Bottom, self).__init__()
        self.original_model = torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2, progress=True)
        self.activation = {}
        h = self.original_model.layer3[2].register_forward_hook(self.getActivation('comp'))
        
    def getActivation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook

    def forward(self, x):
        self.original_model(x)
        output = self.activation['comp']
        return output

class KeyPointResnet50Bottom(nn.Module):
    def __init__(self, scale='2', trainable_weights=False):
        super(KeyPointResnet50Bottom, self).__init__()
        self.original_model = torchvision.models.detection.keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.COCO_V1, progress=True, trainable_backbone_layers=None if trainable_weights else 0)
        self.activation = {}
        self.scale = scale
        self.backbone = self.original_model.backbone  # https://github.com/pytorch/vision/blob/main/torchvision/models/detection/backbone_utils.py
        self.featmap_sizes = {'0': 28, '2': 7}

    def forward(self, x):
        output = self.backbone(x)[self.scale]
        return output

#---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        x = self.dropout(x)
        return x 

#----------------------------------------------------------------------------

class TransEncoder(nn.Module):
    def __init__(self, d_model, n_head, dim_ff, dropout=0.0, num_layers = 1):
        super(TransEncoder, self).__init__()
        self.pos_encoder = PositionalEncoding(d_model, 0.1, 64)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model = d_model,
                                                    nhead = n_head,
                                                    dim_feedforward = dim_ff,
                                                    dropout = dropout,
                                                    activation = 'relu')
        encoder_norm = nn.LayerNorm(d_model)
        self.trans_encoder = nn.TransformerEncoder(encoder_layer, num_layers, encoder_norm)
                
    def forward(self, src):
        src = self.pos_encoder(src)
        e_op = self.trans_encoder(src)
        return e_op


#=============Model====================


class RepNet(nn.Module):
    def __init__(self, num_frames, img_size=112, use_mha=True, backbone='resnet', backbone_scale='2', trainable_backbone=False):
        super(RepNet, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        self.num_frames = num_frames
        self.img_size = img_size
        self.use_mha = use_mha
        self.featmap_size = 7
        if backbone == 'resnet':
            self.resnetBase = ResNet50Bottom()
        elif backbone == 'keypoint':
            self.resnetBase = KeyPointResnet50Bottom(scale=backbone_scale, trainable_weights=trainable_backbone)
            self.featmap_size = int(self.img_size / (7 * 2 ** (int(backbone_scale))))
            self.featmap_size = 13
            self.featmap_size = 49
            self.featmap_size = 57
            print(self.img_size, backbone_scale, self.featmap_size)
            #self.featmap_size = self.resnetBase.featmap_sizes[backbone_scale]
        
        
        self.conv3D = nn.Conv3d(in_channels = 1024 if backbone == 'resnet' else 256,
                                out_channels = 512,
                                kernel_size = 3,
                                padding = (3,1,1),
                                dilation = (3,1,1))
        self.bn1 = nn.BatchNorm3d(512)
        self.pool = nn.MaxPool3d(kernel_size = (1, self.featmap_size, self.featmap_size))
        
        self.sims = Sims()
        if self.use_mha:
            self.mha_sim = nn.MultiheadAttention(embed_dim=512, num_heads=4)

        
        self.conv3x3 = nn.Conv2d(in_channels = 2 if self.use_mha else 1,
                                 out_channels = 32,
                                 kernel_size = 3,
                                 padding = 1)
        
        self.bn2 = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout(0.25)
        self.input_projection = nn.Linear(self.num_frames * 32, 512)
        self.ln1 = nn.LayerNorm(512)
        
        self.transEncoder1 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers = 1)
        #self.transEncoder2 = TransEncoder(d_model=512, n_head=4, dropout = 0.2, dim_ff=512, num_layers = 1)
        
        #period length prediction
        self.fc1_1 = nn.Linear(512, 512)
        self.ln1_2 = nn.LayerNorm(512)
        self.fc1_2 = nn.Linear(512, self.num_frames//2)
        self.fc1_3 = nn.Linear(self.num_frames//2, 1)
        self.period_len_layers = [self.fc1_1, self.fc1_2, self.fc1_3]


        #periodicity prediction
        self.fc2_1 = nn.Linear(512, 512)
        self.ln2_2 = nn.LayerNorm(512)
        self.fc2_2 = nn.Linear(512, self.num_frames//2)
        self.fc2_3 = nn.Linear(self.num_frames//2, 1)
        self.periodicity_layers = [self.fc2_1, self.fc2_2, self.fc2_3]

    def forward(self, x, ret_sims = False, ret_embs = False):
        batch_size, _, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.resnetBase(x)

        #print(x.shape)
        
        x = x.view(batch_size, self.num_frames, x.shape[1],  x.shape[2],  x.shape[3])
        x = x.transpose(1, 2)
        x = F.relu(self.bn1(self.conv3D(x)))
                        
        x = x.view(batch_size, 512, self.num_frames, self.featmap_size, self.featmap_size)
        x = self.pool(x).squeeze(3).squeeze(3)
        x = x.transpose(1, 2)                           #batch, num_frame, 512
        x = x.reshape(batch_size, self.num_frames, -1)
        embs = x
        x1 = F.relu(self.sims(x))
        
        
        x = x.transpose(0, 1)
        if self.use_mha:
            _, x2 = self.mha_sim(x, x, x)
            x2 = F.relu(x2.unsqueeze(1))
            x = torch.cat([x1, x2], dim = 1)
        else:
            x = x1
        xret = x
        
        x = F.relu(self.bn2(self.conv3x3(x)))     #batch, 32, num_frame, num_frame
        #print(x.shape)
        x = self.dropout1(x)

        x = x.permute(0, 2, 3, 1)
        x = x.reshape(batch_size, self.num_frames, -1)  #batch, num_frame, 32*num_frame
        x = self.ln1(F.relu(self.input_projection(x)))  #batch, num_frame, d_model=512
        
        x = x.transpose(0, 1)                          #num_frame, batch, d_model=512
        
        #period
        x1 = self.transEncoder1(x)
        x1 = x1.transpose(0, 1)
        y1 = F.relu(self.ln1_2(self.fc1_1(x1)))
        y1 = F.relu(self.fc1_2(y1))
        y1 = F.relu(self.fc1_3(y1))

        #periodicity
        #x2 = self.transEncoder2(x)
        #y2 = x2.transpose(0, 1)
        y2 = F.relu(self.ln2_2(self.fc2_1(x1)))
        y2 = F.relu(self.fc2_2(y2))
        y2 = self.fc2_3(y2)
        
        #y1 = y1.transpose(1, 2)                         #Cross enropy wants (minbatch*classes*dimensions)
        if ret_sims and ret_embs:
            return y1, y2, xret, embs
        elif ret_embs:
            return y1, y2, embs
        elif ret_sims:
            return y1, y2, xret
        return y1, y2
