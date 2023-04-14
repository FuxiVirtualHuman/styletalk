from torch import nn


class ADAIN(nn.Module):
    def __init__(self, content_nc, condition_nc, hidden_nc):
        super().__init__()

        self.param_free_norm = nn.InstanceNorm1d(content_nc, affine=False)

        use_bias = True

        self.mlp_shared = nn.Sequential(
            nn.Linear(condition_nc, hidden_nc, bias=use_bias),
            nn.ReLU(),
        )
        self.mlp_gamma = nn.Linear(hidden_nc, content_nc, bias=use_bias)
        self.mlp_beta = nn.Linear(hidden_nc, content_nc, bias=use_bias)

    def forward(self, content, condition):

        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(content)

        # Part 2. produce scaling and bias conditioned on feature
        actv = self.mlp_shared(condition)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        out = normalized * (1 + gamma) + beta
        return out


class ConvNormRelu(nn.Module):
    def __init__(
        self,
        conv_type="1d",
        in_channels=3,
        out_channels=64,
        downsample=False,
        kernel_size=None,
        stride=None,
        padding=None,
        norm="IN",
        leaky=False,
        adain_condition_nc=None,
        adain_hidden_nc=None,
    ):
        super().__init__()
        if kernel_size is None:
            if downsample:
                kernel_size, stride, padding = 4, 2, 1
            else:
                kernel_size, stride, padding = 3, 1, 1

        if conv_type == "1d":
            self.conv = nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            )
            if norm == "IN":
                self.norm = nn.InstanceNorm1d(out_channels, affine=True)
            elif norm == "ADAIN":
                self.norm = ADAIN(out_channels, adain_condition_nc, adain_hidden_nc)
            elif norm == "NONE":
                self.norm = nn.Identity()
            else:
                raise NotImplementedError
        nn.init.kaiming_normal_(self.conv.weight)

        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True) if leaky else nn.ReLU(inplace=True)

    def forward(self, x, condition=None):
        """

        Args:
            x (_type_): (B, C, L)
            condition (_type_, optional): (B, C)

        Returns:
            _type_: _description_
        """
        x = self.conv(x)
        if isinstance(self.norm, ADAIN):
            x = self.norm(x, condition)
        else:
            x = self.norm(x)
        x = self.act(x)
        return x


class MyConv1d(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size, stride, padding),
            nn.BatchNorm1d(cout),
        )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)
