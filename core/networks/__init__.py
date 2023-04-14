from core.networks.generator import ContentEncoder, StyleEncoder, Decoder
from core.networks.disentangle_decoder import DisentangleDecoder

def get_network(name: str):
    obj = globals().get(name)
    if obj is None:
        raise KeyError("Unknown Network: %s" % name)
    else:
        return obj
