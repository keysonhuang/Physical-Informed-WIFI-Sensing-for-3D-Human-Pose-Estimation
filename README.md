# WiFi-Based 3D Human Pose Estimation

A PyTorch implementation for estimating 3D human pose from WiFi channel state information (CSI). The training code supports three selectable data sources through a common dataset interface:

- **MM-Fi** — 3 receive-antenna channels × 114 subcarriers; 17-joint annotations.
- **Person-in-WiFi-3D** — 9 spatial links × 30 subcarriers; single-person 14-joint annotations.
- **Our real-world collection** — manifest-driven adapter for the 9-link, 30-subcarrier, 14-joint collection profile.

## Code

The complete training, evaluation, data adapters, configs, and tests are in [code_3D-WIFI-HPE/](code_3D-WIFI-HPE/).

See [code_3D-WIFI-HPE/README.md](code_3D-WIFI-HPE/README.md) for installation, expected directory structures, CSI formats, manifest columns, and commands for each dataset.

## Quick start

~~~bash
cd code_3D-WIFI-HPE
pip install -r requirements.txt

# Select one dataset:
python train.py --config configs/mmfi.yaml
python train.py --config configs/person_in_wifi_3d.yaml
python train.py --config configs/real_world.yaml

pytest -q
~~~

Dataset files and trained checkpoints are not included in this repository. Keep them outside version control.
