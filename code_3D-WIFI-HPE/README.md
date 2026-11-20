# WiFi 3D Human Pose Estimation

A PyTorch training and evaluation codebase with selectable data adapters for MM-Fi, Person-in-WiFi-3D, and a custom real-world collection. Each adapter converts its source into one model interface:

- dynamic: [batch, spatial_channels * subcarriers, 2]
- static: [batch, spatial_channels * subcarriers, 2]
- pose: [batch, joints, 3]
- action: integer class ID
- environment: integer domain ID (-1 means unknown)

Select a dataset with data.dataset in the YAML configuration. No source dataset is copied into this repository.

## 1. Installation

Python 3.10+ is recommended.

~~~bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows PowerShell
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

## 2. Select a dataset

| Dataset | Config | Spatial CSI channels × subcarriers | Joints |
|---|---|---:|---:|
| MM-Fi | configs/mmfi.yaml | 3 × 114 = 342 | 17 |
| Person-in-WiFi-3D | configs/person_in_wifi_3d.yaml | 9 × 30 = 270 | 14 |
| Custom collection | configs/real_world.yaml | 9 × 30 = 270 by default | 14 by default |

The model derives its input grid and output joint count from the selected config. The PGMA encoder accepts a configurable number of spatial channels. For nine distributed links, it does not invent link coordinates: the spatial part of the coherence prior is neutral unless signal.channel_positions_m is supplied.

### MM-Fi

Expected layout:

~~~text
MMFi_Dataset/Dataset/
  E01/S01/A01/
    ground_truth.npy
    wifi-csi/frame001.mat
    ...
~~~

Each MAT file contains CSIamp and CSIphase, normally [3, 114, packets]. Pose annotations are [frames, 17, 3]. The MM-Fi adapter retains the P1/P2/P3 action protocols and random, cross-subject, and cross-environment splits.

~~~bash
python train.py --config configs/mmfi.yaml --data-root ../MMFi_Dataset/Dataset
python evaluate.py --checkpoint checkpoints/best.pt --data-root ../MMFi_Dataset/Dataset
~~~

### Person-in-WiFi-3D

Use the official single-person split layout:

~~~text
data/person_in_wifi_3d/
  train_data/
    csi/                  # raw HDF5 MAT files
    keypoint/             # one .npy annotation per sample
    train_data_list.txt
    csi_ap/               # optional preprocessed .npy files
  test_data/
    csi/
    keypoint/
    test_data_list.txt
    csi_ap/
~~~

The raw CSI MAT file follows the official csi_out/real and csi_out/imag HDF5 layout. The loader converts [packets, subcarriers, TX, RX] into nine spatial links and 30 subcarriers, applies phase sanitization, and averages the packet dimension. If csi_ap/{sample}.npy exists, it can instead read the official preprocessed [3, 180, packets] amplitude/phase tensor. Keypoint files may be [14, 3] or [1, 14, 3].

The pose regressor is single-person. Samples with more than one annotated person are skipped; this matches the single-person benchmark subset used by this project. The official split lists determine train and validation/test samples.

The official list files do not provide action and environment labels in a separate standardized column. Therefore the example config disables action and domain losses and action-accuracy reporting by default. If labels are available, populate action_map and environment_map with keys equal to a sample name or its sequence key (the first two underscore-separated name components), then set action_labels_available: true and enable the corresponding loss weights.

~~~bash
python train.py --config configs/person_in_wifi_3d.yaml --data-root ./data/person_in_wifi_3d
python evaluate.py --checkpoint checkpoints/person_in_wifi_3d/best.pt
~~~

### Custom real-world collection

The custom adapter uses a CSV manifest so the loader does not assume undocumented raw filenames or synchronization conventions.

Recommended directory:

~~~text
data/real_world/
  metadata/manifest.csv
  csi/
    frame000001.npz
    frame000002.npz
  poses/
    frame000001.npy
    frame000002.npy
~~~

Required manifest columns:

| Column | Meaning |
|---|---|
| csi_path | CSI file path, relative to data.root or absolute |
| pose_path | Pose .npy/.npz/.mat path |
| split | train or val/test |
| sample_id | Optional stable sample name |
| sequence_id | Optional recording/clip ID used to build temporal windows |
| frame_index | Optional integer frame order within the sequence |
| action_id | Action class in [0, model.num_actions) |
| environment_id | Environment/domain ID in [0, model.num_environments) |

The example configs/real_world.yaml expects an amplitude/phase NPZ or MAT file with keys CSIamp and CSIphase. Each array can be [9, 30, packets] or [3, 3, 30, packets]. Pose files are [14, 3] (or [1, 14, 3]) in millimeters. Change amplitude_key, phase_key, and pose_key if your saved field names differ.

Supported data.csi_format values:

- amplitude_phase: separate amplitude and phase arrays in NPZ/MAT files.
- complex: a complex CSI array in NPY/NPZ/MAT.
- features: precomputed [270, 2] features.
- person_in_wifi_hdf5: the same csi_out/real and csi_out/imag HDF5 MAT layout as the public Person-in-WiFi-3D dataset.

The real-world recording profile supported by the example config is three links with nine TX/RX antenna pairs, 30 subcarriers, 20 CSI packets per pose frame, and a 14-joint pose. Update signal.num_spatial_channels, signal.num_subcarriers, model.num_joints, and the CSI format/keys if your exported files use a different shape. The CSI packet-to-pose alignment must be done before writing the manifest; sequence_id and frame_index then define the pose-frame temporal order used by the centered static/dynamic window.

The example config requires action_id and environment_id labels. If those labels are not available, set require_action_environment_labels: false, set the relevant loss weights to zero, and set action_labels_available: false. Configure the right-shoulder and left-hip joint indices to enable PCK metrics; otherwise MPJPE and PA-MPJPE are reported.

~~~bash
python train.py --config configs/real_world.yaml --data-root ./data/real_world
python evaluate.py --checkpoint checkpoints/real_world/best.pt
~~~

## 3. Temporal processing

All adapters produce one pose-aligned CSI feature matrix per sample. A centered window_size over pose frames is used to compute:

- static: mean CSI feature across the temporal window.
- dynamic: current-frame feature minus the static mean.

The Person-in-WiFi-3D adapter first reduces the 20 packet observations belonging to one video frame to one frame feature. The custom adapter expects each manifest row to already represent one pose-aligned frame. For the described real-world sampling rate (600 Hz CSI and 30 fps pose), that means each frame-level CSI file contains its corresponding 20 packets.

## 4. Tests

Run from this directory:

~~~bash
pytest -q
~~~

The synthetic tests cover the original MM-Fi path, the official Person-in-WiFi-3D HDF5 and preprocessed CSI layouts, the custom manifest adapter, and the 9-link/14-joint model configuration. They do not require downloading the full datasets.

## 5. Notes

- The custom adapter is intentionally schema-driven. It does not guess the binary layout of private capture files; configure the field names or export each aligned frame into the manifest format above.
- The 9-link spatial coherence prior is left geometry-neutral unless calibrated channel coordinates are provided.
- Keep raw data and checkpoints outside version control.
