import csv

import h5py
import numpy as np
import torch

from wifi_pose.data.custom_manifest import ManifestPoseDataset
from wifi_pose.data.person_in_wifi_3d import PersonInWiFi3DPoseDataset
from wifi_pose.models import build_model


def make_person_wifi_sample(split_dir, name, seed=1):
    rng = np.random.default_rng(seed)
    (split_dir / "csi").mkdir(parents=True, exist_ok=True)
    (split_dir / "keypoint").mkdir(parents=True, exist_ok=True)
    raw = (rng.normal(size=(20, 30, 3, 3)) + 1j * rng.normal(size=(20, 30, 3, 3))).astype(np.complex64)
    with h5py.File(split_dir / "csi" / f"{name}.mat", "w") as handle:
        group = handle.create_group("csi_out")
        group.create_dataset("real", data=raw.real)
        group.create_dataset("imag", data=raw.imag)
    np.save(split_dir / "keypoint" / f"{name}.npy", rng.normal(size=(1, 14, 3)).astype(np.float32))


def test_person_in_wifi_3d_official_layout_and_raw_csi(tmp_path):
    train_dir = tmp_path / "train_data"
    test_dir = tmp_path / "test_data"
    make_person_wifi_sample(train_dir, "p1_1_0", 1)
    make_person_wifi_sample(train_dir, "p1_1_1", 2)
    make_person_wifi_sample(test_dir, "p1_2_0", 3)
    (train_dir / "train_data_list.txt").write_text("p1_1_0\np1_1_1\n")
    (test_dir / "test_data_list.txt").write_text("p1_2_0\n")
    config = {
        "window_size": 3,
        "signal": {"subcarrier_spacing_hz": 625000.0},
        "action_map": {"p1_1": 2},
        "environment_map": {"p1_1": 1},
    }
    dataset = PersonInWiFi3DPoseDataset(tmp_path, config, "train")
    sample = dataset[0]
    assert sample["dynamic"].shape == (270, 2)
    assert sample["static"].shape == (270, 2)
    assert sample["pose"].shape == (14, 3)
    assert sample["action"] == 2
    assert sample["environment"] == 1
    assert torch.isfinite(sample["dynamic"]).all()


def test_person_in_wifi_preprocessed_csi_ap_layout(tmp_path):
    train_dir = tmp_path / "train_data"
    test_dir = tmp_path / "test_data"
    for directory, name in ((train_dir, "p1_1_0"), (test_dir, "p1_2_0")):
        (directory / "csi_ap").mkdir(parents=True, exist_ok=True)
        (directory / "keypoint").mkdir(parents=True, exist_ok=True)
        np.save(directory / "csi_ap" / f"{name}.npy", np.random.default_rng(4).normal(size=(3, 180, 20)).astype(np.float32))
        np.save(directory / "keypoint" / f"{name}.npy", np.zeros((1, 14, 3), dtype=np.float32))
    (train_dir / "train_data_list.txt").write_text("p1_1_0\n")
    (test_dir / "test_data_list.txt").write_text("p1_2_0\n")
    dataset = PersonInWiFi3DPoseDataset(tmp_path, {"window_size": 1}, "train")
    assert dataset[0]["dynamic"].shape == (270, 2)


def test_custom_manifest_amplitude_phase_adapter(tmp_path):
    (tmp_path / "csi").mkdir()
    (tmp_path / "poses").mkdir()
    rng = np.random.default_rng(10)
    for index, split in enumerate(("train", "val")):
        np.savez(
            tmp_path / "csi" / f"sample{index}.npz",
            CSIamp=rng.uniform(0.1, 2.0, size=(9, 30, 20)).astype(np.float32),
            CSIphase=rng.uniform(-np.pi, np.pi, size=(9, 30, 20)).astype(np.float32),
        )
        np.save(tmp_path / "poses" / f"sample{index}.npy", rng.normal(size=(14, 3)).astype(np.float32))
    with (tmp_path / "manifest.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["sample_id", "csi_path", "pose_path", "split", "sequence_id", "frame_index", "action_id", "environment_id"],
        )
        writer.writeheader()
        writer.writerow({"sample_id": "s0", "csi_path": "csi/sample0.npz", "pose_path": "poses/sample0.npy", "split": "train", "sequence_id": "seq0", "frame_index": 0, "action_id": 1, "environment_id": 0})
        writer.writerow({"sample_id": "s1", "csi_path": "csi/sample1.npz", "pose_path": "poses/sample1.npy", "split": "val", "sequence_id": "seq1", "frame_index": 0, "action_id": 1, "environment_id": 1})
    config = {
        "manifest": "manifest.csv",
        "window_size": 1,
        "csi_format": "amplitude_phase",
        "amplitude_key": "CSIamp",
        "phase_key": "CSIphase",
        "require_action_environment_labels": True,
        "signal": {"subcarrier_spacing_hz": 625000.0},
    }
    dataset = ManifestPoseDataset(tmp_path, config, "train")
    sample = dataset[0]
    assert sample["dynamic"].shape == (270, 2)
    assert sample["pose"].shape == (14, 3)
    assert sample["action"] == 1
    assert sample["environment"] == 0


def test_model_accepts_nine_links_and_fourteen_joints():
    config = {
        "data": {
            "split": "fixed",
            "signal": {
                "num_spatial_channels": 9,
                "num_subcarriers": 30,
                "center_frequency_hz": 5.32e9,
                "subcarrier_spacing_hz": 625000.0,
            },
        },
        "model": {
            "num_actions": 8,
            "num_joints": 14,
            "d_model": 16,
            "num_heads": 4,
            "encoder_layers": 1,
            "feedforward_dim": 32,
            "action_embedding_dim": 16,
            "num_frequency_bands": 3,
            "dropout": 0.0,
            "dynamic_kernel_size": 3,
            "environment_hidden_dim": 16,
            "num_environments": 4,
        },
    }
    model = build_model(config)
    dynamic = torch.randn(2, 270, 2)
    static = torch.randn(2, 270, 2)
    output = model(dynamic, static, torch.tensor([0, -1]))
    assert output["pose"].shape == (2, 14, 3)
    assert output["action_logits"].shape == (2, 8)
    assert torch.isfinite(output["pose"]).all()
