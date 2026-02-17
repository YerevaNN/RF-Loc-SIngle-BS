#!/bin/bash
# Visualization
python -m streamlit run run.py \
    --server.address=0.0.0.0 \
    -- \
    --config-name="validation_visualize"
