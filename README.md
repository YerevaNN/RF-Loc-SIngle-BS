# RF-Loc-Single-BS

Code for the paper: [Deep learning with synthetic data for wireless NLOS positioning with a single base station](https://doi.org/10.1016/j.adhoc.2024.103696)

RF-based localization using a single base station on the [WAIR-D dataset](https://www.mobileai-dataset.com/html/default/yingwen/DateSet/1590994253188792322.html?index=1).

## Setup

### 1. Create the conda environment

```bash
conda env create
conda activate rf_loc_single_bs
```

### 2. Download the WAIR-D dataset

Download the Wireless AI Research Dataset from:
https://www.mobileai-dataset.com/html/default/yingwen/DateSet/1590994253188792322.html?index=1

The dataset contains:
- **Part 1 (Sparse Deployment):** 10,000 maps with 5 BS and 30 UE locations each
- **Part 2 (Dense Deployment):** 100 maps with 1 BS and 10,000 UE locations each

### 3. Configure environment variables

Copy the example environment file and fill in the paths:

```bash
cp .env.example .env
```

Edit `.env` with your local paths:

| Variable | Description |
|---|---|
| `RAW_DATA_DIR` | Path to the raw WAIR-D dataset |
| `PREPARED_DATA_DIR` | Path to store prepared image data |
| `IMAGE_DATA_PATH` | Path to the prepared image dataset |
| `SEQUENCE_DATA_PATH` | Path to the prepared sequence dataset |
| `NOISE_DIR` | Path to store generated noise data |
| `PREDICTIONS_PATH` | Path to store inference predictions |
| `OUTPUT_DIR` | Path for training outputs (checkpoints, logs) |
| `AIM_REPO` | Path to the Aim experiment tracking repository |

## Run Scripts

All run scripts are located in the `run/` directory.

### Data Preparation

| Script | Description |
|---|---|
| `run/create_images.sh` | Prepare image data from raw WAIR-D dataset |
| `run/create_noisy_images.sh` | Prepare image data with added angle noise |
| `run/create_aod_centered_images.sh` | Prepare images with AoD centered at the image center |
| `run/create_noisy_aod_centered_images.sh` | Prepare AoD-centered images with added angle noise |
| `run/create_noise.sh` | Generate angle noise data (AoD/AoA) |
| `run/create_sequences.sh` | Prepare sequence data from raw WAIR-D dataset |
| `run/create_noisy_sequences.sh` | Prepare sequence data with added angle noise |

### Training

| Script | Description |
|---|---|
| `run/train_unet.sh` | Train a UNet model on image data |
| `run/train_mlp_unet.sh` | Train an MLP-UNet model on image data |
| `run/train_cnn.sh` | Train a CNN model on sequence data |
| `run/train_mlp.sh` | Train the WAIR-D original MLP model on sequence data |

### Inference and Evaluation

| Script | Description |
|---|---|
| `run/inference.sh` | Run inference with a trained model |
| `run/evaluate.sh` | Evaluate predictions (RMSE, accuracy at distance thresholds) |
| `run/visualize.sh` | Launch a Streamlit app for visualizing results |
