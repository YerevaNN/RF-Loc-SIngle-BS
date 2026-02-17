#!/bin/bash
# Training with MLP-UNet
python run.py \
    datamodule=wair_d_images \
    network=mlp_unet \
    network.u_input_channels=2 \
    algorithm=mlp_unet \
    datamodule.batch_size=2 \
    datamodule.num_workers=4 \
    trainer.max_epochs=15 \
