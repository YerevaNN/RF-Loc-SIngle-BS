#!/bin/bash
# Inference
python run.py \
    --config-name="inference" \
    split=test \
    algorithm=mlp_unet \
    network=mlp_unet \
    network.u_input_channels=3 \
    network.mlp_input_dim=11 \
    datamodule=wair_d_images_sequence \
    datamodule.batch_size=2 \
    datamodule.num_workers=2 \
    datamodule.input_kernel_size=null \
    datamodule.output_kernel_size=3 \
    datamodule.kernel_size_decay=1 \
    datamodule.use_channels_seq=[0,1,2,3,4] \
    datamodule.los_ratio=1 \
    datamodule.n_links=3 \
    gpu=0 \

