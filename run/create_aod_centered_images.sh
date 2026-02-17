#!/bin/bash
# Creating AoD centered images
python run.py \
    --config-name=prepare_data \
    center_aod=True \
    prepared_data_dir=/nfs/dgx/raid/iot/data/adhoc_images
