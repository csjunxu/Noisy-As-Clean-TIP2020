from .rcan_model.rcan import RCAN
import torch
import torch.nn as nn

class Rcan(nn.Module):


    def __init__(self, input_depth, n_resgroups, n_resblocks, n_feats, reduction):
        super(Rcan, self).__init__()
        # rcan_args = {
        #     'n_resgroups'   : 5,
        #     'n_resblocks'   : 10,
        #     'n_feats'       : 64,
        #     'reduction'     : 16,
        #     'scale'         : [1],
        #     'rgb_range'     : 255,
        #     'n_colors'      : 3,
        #     'res_scale'     : 1,
        # }
        rcan_args = {
            'n_resgroups'   : n_resgroups,
            'n_resblocks'   : n_resblocks,
            'n_feats'       : n_feats,
            'reduction'     : reduction,
            'scale'         : [1],
            'rgb_range'     : 1,
            'n_colors'      : input_depth,
            'res_scale'     : 1,
        }




        self.model = make_model(rcan_args)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)

def make_model(args):
  return RCAN(args).to(torch.device('cuda'))