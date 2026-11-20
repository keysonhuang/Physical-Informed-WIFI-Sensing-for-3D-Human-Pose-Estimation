# WiFi-Based 3D Human Pose Estimation

A PyTorch implementation for estimating 3D human joints from pose-aligned WiFi channel state information (CSI). The project includes CSI sanitization, temporal static/dynamic decomposition, physics-guided multipath attention, action-conditioned pose decoding, cross-environment feature alignment, environment adapters, training, and evaluation.

## Features

- **CSI sanitization:** cyclic conjugate products across three receive antennas, packet-level amplitude normalization, phase unwrapping, and least-squares phase-slope removal.
- **Temporal decomposition:** a centered moving average separates the static channel component from the motion-related residual.
- **Physics-guided encoder:** antenna/frequency positional encoding, a learnable multipath-coherence prior, and three frequency-restricted attention branches.
- **Action-conditioned pose estimation:** action probabilities are inferred from CSI and used to generate dynamic convolution kernels, query spatial features, and modulate joint-specific decoders.
- **Cross-environment adaptation:** a supervised contrastive objective aligns same-action dynamic features across environments; lightweight adapters use static CSI for calibration.
- **Evaluation:** MPJPE, similarity-aligned PA-MPJPE, torso-normalized PCK@20/PCK@50, and action accuracy.
- **Model size:** approximately 14.0 million trainable parameters with the default MM-Fi configuration.

## Repository layout

    .
    ├── README.md
    ├── scene1-4actions-bright.mp4
    ├── scene2-5actions-weaklight.mp4
    └── code_3D-WIFI-HPE/
        ├── configs/mmfi.yaml
        ├── tests/
        ├── wifi_pose/
        │   ├── data/       # CSI processing and MM-Fi dataset
        │   ├── models/     # PGMA encoder and pose network
        │   ├── config.py
        │   ├── engine.py
        │   ├── losses.py
        │   └── metrics.py
        ├── evaluate.py
        ├── train.py
        └── requirements.txt

## Requirements

- Python 3.10 or newer
- PyTorch 2.0 or newer
- NumPy, SciPy, and PyYAML
- CUDA is optional; CPU execution is supported

Install dependencies:

    cd code_3D-WIFI-HPE
    python -m venv .venv
    source .venv/bin/activate        # Windows: .venv\Scripts\activate
    python -m pip install --upgrade pip
    pip install -r requirements.txt

## Dataset

The loader expects the MM-Fi directory layout below. Dataset files are not included in this repository.

    MMFi_Dataset/Dataset/
    ├── E01/
    │   └── S01/
    │       └── A01/
    │           ├── ground_truth.npy
    │           └── wifi-csi/
    │               ├── frame001.mat
    │               ├── frame002.mat
    │               └── ...
    ├── E02/
    ├── E03/
    └── E04/

Each CSI MAT file must contain:

- CSIamp: amplitude array shaped [3, 114, packets].
- CSIphase: phase array shaped [3, 114, packets], in radians.

Two-dimensional [3, 114] arrays are also accepted and treated as one packet. Each ground_truth.npy file must contain 17 three-dimensional joints per aligned frame, shaped [frames, 17, 3]. Coordinates are evaluated in millimeters. The loader aligns the available CSI frames and pose annotations by their common sequence length.

The default signal configuration describes one transmitter, three receive antennas, 114 active subcarriers, a 40 MHz channel, and a 5 GHz center frequency. Antenna spacing defaults to half a wavelength. If the recording hardware differs, update the signal geometry and frequency fields in configs/mmfi.yaml and ensure the CSI loader receives the corresponding antenna/subcarrier layout.

### CSI processing

For each pose-aligned CSI frame, the loader:

1. Reconstructs complex CSI from amplitude and phase.
2. Forms cyclic conjugate products between each receive antenna and the next antenna in the receiver array.
3. Uses the square root of the pair-product magnitude as the pair amplitude and normalizes packet-level gain.
4. Unwraps each pair phase across subcarriers and removes its least-squares linear frequency slope.
5. Averages packet observations and stacks amplitude and phase into a [342, 2] feature tensor.
6. Computes the static component as the mean over a centered 10-frame window; the dynamic component is the current frame minus that mean.

At sequence boundaries, the temporal window is padded by repeating the first or last available frame. For an even 10-frame window, this implementation uses four preceding frames, the current frame, and five future frames; this preserves the five-frame look-ahead used by the centered filter.

## Data protocols and splits

Set data.protocol and data.split in configs/mmfi.yaml.

| Setting | Meaning |
|---|---|
| P1 | Daily activities: A02, A03, A04, A05, A13, A14, A17, A18, A19, A20, A21, A22, A23, A27 |
| P2 | Rehabilitation activities: A01, A06, A07, A08, A09, A10, A11, A12, A15, A16, A24, A25, A26 |
| P3 | All 27 activities |
| random | Deterministic 80/20 split by complete subject-action recordings; frames from one recording are not split across sets |
| cross_subject | Subject-disjoint split using train_subjects and test_subjects in the configuration |
| cross_environment | Train on E01-E03 and evaluate on E04 by default |

The default subject-disjoint lists hold out two subjects from each environment. Edit the subject lists to use a different subject split. For cross-environment evaluation, the held-out environment is treated as unknown and its static calibration residual is the mean output of the source-environment adapters.

## Training

From the code_3D-WIFI-HPE directory:

    python train.py --config configs/mmfi.yaml --data-root ../MMFi_Dataset/Dataset

Select a device or checkpoint directory when needed:

    python train.py \
      --config configs/mmfi.yaml \
      --data-root ../MMFi_Dataset/Dataset \
      --device cuda:0 \
      --checkpoint-dir ./checkpoints

Default optimization settings:

| Parameter | Value |
|---|---:|
| Optimizer | Adam, betas (0.9, 0.999) |
| Initial learning rate | 1e-4 |
| Learning-rate schedule | Multiply by 0.5 every 30 epochs |
| Batch size | 32 |
| Epochs | 100 |
| Action loss weight | 0.5 |
| Cross-environment loss weight | 0.2 |
| Contrastive temperature | 0.07 |
| Static/dynamic window | 10 aligned frames |

The total objective combines squared Euclidean joint regression, action cross-entropy, and supervised contrastive alignment of samples that share an action but come from different environments. The action label is used only as a training target and to form contrastive positive pairs; the pose model predicts action probabilities from CSI during both training and inference.

The best checkpoint is selected by validation MPJPE and saved as checkpoints/best.pt. The latest checkpoint is saved as checkpoints/last.pt when training.save_last is enabled.

## Evaluation

Evaluate the best checkpoint with the split and configuration stored in the checkpoint:

    python evaluate.py \
      --checkpoint ./checkpoints/best.pt \
      --data-root ../MMFi_Dataset/Dataset \
      --device cuda:0

Reported metrics:

- **MPJPE (mm):** mean Euclidean joint error in the original coordinate frame.
- **PA-MPJPE (mm):** MPJPE after per-sample similarity alignment, including rotation, translation, and scale.
- **PCK@20 / PCK@50:** percentage of joints within 20% / 50% of the ground-truth torso length. The torso length uses the right-shoulder and left-hip indices configured in metrics.
- **Action accuracy:** accuracy of the CSI-derived action prediction.

For a cross-environment split, evaluation does not select an adapter using the held-out environment identity. It averages the source adapters for each test sample.

## Tests

Run the automated tests from the project directory:

    pytest -q

The tests cover CSI sanitization, MAT loading, temporal-window dataset construction, model forward/backward execution, unknown-environment adapter inference, contrastive objectives, and pose metrics. They use synthetic data and do not require the full MM-Fi dataset.

## Checkpoints and reproducibility

- Random seeds are configured at the top level of configs/mmfi.yaml.
- The complete configuration is stored in each checkpoint.
- Checkpoints and generated outputs are excluded from version control.
- The full dataset and trained weights must be obtained and stored separately.

## Scope

The included data pipeline is configured for MM-Fi CSI recordings with three receive antennas, 114 subcarriers, and 17-joint annotations. Other hardware layouts or skeleton definitions require corresponding dataset and geometry configuration changes.
