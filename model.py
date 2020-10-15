from functools import partial

import torch.nn as nn
import torch.nn.functional as F

NORM_DICT ={
    "bn":nn.BatchNorm2d
}
ACT_DICT = {
    "relu":nn.ReLU
}

def get_model(main_params, 
              feature={}, aggregator={}, mlp={}, with_softmax=False):
    device = main_params["device"]
    in_channel = main_params["in_channel"]
    num_classes = main_params["num_classes"]

    last_act = partial(nn.Softmax, dim=1) if with_softmax else None
    model = Model(in_channel, num_classes, feature=feature, aggregator=aggregator, mlp=mlp, last_act=last_act).to(device)
    return model


class Model(nn.Module):
    def __init__(self, in_channel, num_classes, feature={}, aggregator={}, mlp={}, last_act=None):
        super().__init__()
        self.feature, num_last_channels = get_feature(in_channel, **feature)
        self.aggregator, num_features = get_aggregator(num_last_channels, **aggregator)
        self.mlp = get_mlp(num_features, num_classes, **mlp)

        self.last_act = last_act() if last_act is not None else None

    def forward(self, x):
        x = self.feature(x)        
        x = self.aggregator(x)        
        o = self.mlp(x)    
        if self.last_act is not None:
            o = self.last_act(o)
        return o

def get_feature(in_channel, mod_type="conv", c0=32, num_layers=3, norm=None, act="relu"):
    norm_func = NORM_DICT.get(norm, None)
    act_func = ACT_DICT.get(act, None)

    if mod_type == "conv":
        cin = in_channel
        cout = c0
        seqs = []
        for n in range(num_layers):
            seqs.append(ConvMod(cin, cout, 3, padding=1, stride=2, norm_func=norm_func, act_func=act_func))
            cin, cout = cout, cout * 2
        num_last_channels = cout // 2
    else:
        raise NotImplementedError(mod_type)
    return nn.Sequential(*seqs), num_last_channels

def get_aggregator(num_last_channels, agr_type="gap"):
    if agr_type == "gap":
        aggregator = GapAggregator()
        num_features = num_last_channels
    else:
        raise NotImplementedError(agr_type)

    return aggregator, num_features

def get_mlp(num_features, num_classes, layer_units=[], act="relu"):
    act_func = ACT_DICT.get(act, None)
    seqs = []
    cin = num_features
    for layer_unit in layer_units:
        seqs.append(nn.Linear(cin, layer_unit))
        seqs.append(act_func())
        cin = layer_unit 
    seqs.append(nn.Linear(cin, num_classes))
    return nn.Sequential(*seqs)


class ConvMod(nn.Sequential):
  def __init__(self, in_channel, out_channel, kernel_size, padding=1, stride=2, norm_func=None, act_func=None):
    seqs = []
    seqs.append(nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride))
    if norm_func is not None:
      seqs.append(norm_func(out_channel))
    if act_func is not None:
      seqs.append(act_func())
    super().__init__(*seqs)


class GapAggregator(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool(x).squeeze(3).squeeze(2)
        return x


if __name__ == "__main__":
    main_params = {
        "device":"cpu",
        "in_channel": 1,
        "num_classes": 10

    }
    model = get_model(main_params)
    print(model)