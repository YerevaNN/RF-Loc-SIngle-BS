#!/bin/bash
# Training with UNet
python run.py \
    datamodule=wair_d_images \
    network=unet \
    network.input_channels=2 \
    algorithm=unet_erm \
    datamodule.batch_size=2 \
    datamodule.num_workers=4 \
    trainer.max_epochs=15