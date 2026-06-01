import ml_collections
import numpy as np

from models.encoders import make_encoder
from models.networks import Identity


def build_encoders(cfg: ml_collections.ConfigDict, observations) -> dict:
    return {"encoder": make_observation_encoder(cfg, observations)}

def should_use_image_encoder(cfg: ml_collections.ConfigDict, observations: np.ndarray) -> bool:
    if cfg.encoder.lower() == "none":
        return False
    if observations.ndim < 4:
        return False
    return True


def make_image_encoder(cfg: ml_collections.ConfigDict):
    encoder_kwargs = {
        "hidden_dims": (cfg.encoder_feature_dim,),
    }

    if cfg.encoder.lower() == "drqv2":
        encoder_kwargs["num_features"] = cfg.encoder_num_features

    return make_encoder(cfg.encoder.lower(), **encoder_kwargs)


def make_observation_encoder(cfg: ml_collections.ConfigDict, observations: np.ndarray):
    if should_use_image_encoder(cfg, observations):
        return make_image_encoder(cfg)
    return Identity()
