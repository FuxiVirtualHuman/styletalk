import torch.nn as nn

from core.networks import get_network


class StyleTalk(nn.Module):
    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg

        content_encoder_class = get_network(cfg.CONTENT_ENCODER_TYPE)
        self.content_encoder = content_encoder_class(**cfg.CONTENT_ENCODER)

        style_encoder_class = get_network(cfg.STYLE_ENCODER_TYPE)
        cfg.defrost()
        cfg.STYLE_ENCODER.input_dim = cfg.DATASET.FACE3D_DIM
        cfg.freeze()
        self.style_encoder = style_encoder_class(**cfg.STYLE_ENCODER)

        decoder_class = get_network(cfg.DECODER_TYPE)
        cfg.defrost()
        cfg.DECODER.output_dim = cfg.DATASET.FACE3D_DIM
        cfg.freeze()
        self.decoder = decoder_class(**cfg.DECODER)
