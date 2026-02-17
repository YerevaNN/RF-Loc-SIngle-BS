#!/bin/bash
# Training with MLP (WAIR-D Original)
python run.py \
    datamodule=wair_d_sequence \
    network=wair_d_original \
    network.input_dim=67 \
    algorithm=erm \
    datamodule.batch_size=2 \
    datamodule.num_workers=4 \
    trainer.max_epochs=15
