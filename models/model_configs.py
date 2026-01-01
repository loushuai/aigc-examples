from typing import Union

from models.unet import UNetModel


MODEL_CONFIGS = {
    "ddpm": {
        "in_channels": 3,
        "model_channels": 64,
        "out_channels": 3,
        "num_res_blocks": 2,
        "attention_resolutions": [2],
        "dropout": 0.0,
        "channel_mult": [2, 4, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "flow_matching": {
        "in_channels": 3,
        "model_channels": 128,
        "out_channels": 3,
        "num_res_blocks": 4,
        "attention_resolutions": [2],
        "dropout": 0.3,
        "channel_mult": [2, 2, 2],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
    "score_matching": {
        "in_channels": 3,
        "model_channels": 64,
        "out_channels": 3,
        "num_res_blocks": 2,
        "attention_resolutions": [2],
        "dropout": 0.0,
        "channel_mult": [2, 4, 4],
        "conv_resample": False,
        "dims": 2,
        "num_classes": None,
        "use_checkpoint": False,
        "num_heads": 1,
        "num_head_channels": -1,
        "num_heads_upsample": -1,
        "use_scale_shift_norm": True,
        "resblock_updown": False,
        "use_new_attention_order": True,
        "with_fourier_features": False,
    },
}


def instantiate_model(architechture: str) -> UNetModel:
    if architechture in MODEL_CONFIGS:
        model_config = MODEL_CONFIGS[architechture]
        return UNetModel(**model_config)
    else:
        raise ValueError(f"Unknown architecture: {architechture}")


# test the function
if __name__ == "__main__":
    model = instantiate_model("ddpm")
    print(model)
