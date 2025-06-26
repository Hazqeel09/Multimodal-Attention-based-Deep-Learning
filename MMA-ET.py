import torch
from torch import nn
import torch.nn.functional as F
from pytorch_tabnet.tab_network import TabNetEncoder
from vit_orig import ViTOrig
    
class MMA_ET(torch.nn.Module):
    def __init__(self, inp_dim, unsupervised_model, att_dropout = 0., ff_dropout = 0., drop_path  =  0.):
        super(MMA_ET, self).__init__()
        self.enc_tabnet = TabNetEncoder(
            input_dim=inp_dim, output_dim=64, n_d=64, n_a=64
        )
        self.vit = ViTOrig(
            image_size=8,
            patch_size=8,
            num_classes=3,
            dim=1024,
            depth=8,
            heads=128,
            mlp_dim=2048,
            channels=13,
            att_dropout = att_dropout,
            ff_dropout = ff_dropout,
            drop_path = drop_path,
        )
        self.pre_train_weight = unsupervised_model.network.encoder.state_dict()
        self.enc_tabnet.load_state_dict(self.pre_train_weight)

    def forward(self, batch_size, vital_data, textual_data):
        enc_output, enc_loss = self.enc_tabnet(vital_data)

        layers_sentence_vectors = torch.reshape(textual_data, (batch_size, 12, 8, 8))
        layer_vital_signs = torch.reshape(
            (enc_output[0] + enc_output[1] + enc_output[2]), (batch_size, 1, 8, 8)
        )
        concatenated_layers = torch.cat((layer_vital_signs, layers_sentence_vectors), 1)

        output = self.vit(concatenated_layers)

        return output, enc_loss