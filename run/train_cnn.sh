#!/bin/bash
# Training with CNN
python run.py \
    datamodule=wair_d_sequence \
    network=cnn \
    network.p=15 \
    algorithm=erm \
    datamodule.batch_size=2 \
    datamodule.num_workers=4 \
    trainer.max_epochs=15
