import inspect

import numpy as np
import torch
from scipy.io import savemat

from wifi_pose.data.csi import clear_csi_cache, load_sanitized_csi, preprocess_csi_arrays
from wifi_pose.data.mmfi import MMFiPoseDataset
from wifi_pose.losses import total_objective
from wifi_pose.metrics import mpjpe, pa_mpjpe, pck
from wifi_pose.models import build_model


def tiny_config():
    return {
        "data": {
            "signal": {
                "num_receive_antennas": 3,
                "num_subcarriers": 12,
                "center_frequency_hz": 5.0e9,
                "subcarrier_spacing_hz": 312500.0,
                "antenna_spacing_m": None,
                "sigma_antenna_m": None,
                "sigma_frequency_hz": 6.0e6,
            },
            "split": "cross_environment",
            "source_environment_ids": [0, 1, 2],
        },
        "model": {
            "num_actions": 5,
            "num_joints": 17,
            "d_model": 16,
            "num_heads": 4,
            "encoder_layers": 1,
            "feedforward_dim": 32,
            "action_embedding_dim": 32,
            "num_frequency_bands": 3,
            "dropout": 0.0,
            "dynamic_kernel_size": 3,
            "environment_hidden_dim": 16,
            "num_environments": 4,
        },
        "training": {
            "action_loss_weight": 0.5,
            "domain_loss_weight": 0.2,
            "contrastive_temperature": 0.07,
        },
    }


def test_preprocess_csi_arrays_returns_finite_pair_features():
    rng = np.random.default_rng(7)
    amplitude = rng.uniform(0.2, 2.0, size=(3, 24, 8))
    phase = rng.uniform(-np.pi, np.pi, size=(3, 24, 8))
    amplitude[0, 0, 0] = np.nan
    phase[1, 2, 1] = np.inf
    features = preprocess_csi_arrays(amplitude, phase)
    assert features.shape == (3 * 24, 2)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    assert np.all(features[:, 0] >= 0.0)


def test_load_sanitized_csi_reads_mat_file(tmp_path):
    rng = np.random.default_rng(8)
    path = tmp_path / "frame001.mat"
    savemat(path, {
        "CSIamp": rng.uniform(0.5, 1.5, size=(3, 18, 4)),
        "CSIphase": rng.uniform(-np.pi, np.pi, size=(3, 18, 4)),
    })
    clear_csi_cache()
    features = load_sanitized_csi(path)
    assert features.shape == (54, 2)
    assert np.isfinite(features).all()
    clear_csi_cache()


def create_record(root, scene, subject, action="A01", frames=4):
    action_dir = root / scene / subject / action
    csi_dir = action_dir / "wifi-csi"
    csi_dir.mkdir(parents=True)
    rng = np.random.default_rng(sum(map(ord, scene + subject)))
    for index in range(1, frames + 1):
        savemat(csi_dir / f"frame{index:03d}.mat", {
            "CSIamp": rng.uniform(0.5, 1.5, size=(3, 12, 3)),
            "CSIphase": rng.uniform(-np.pi, np.pi, size=(3, 12, 3)),
        })
    np.save(action_dir / "ground_truth.npy", rng.normal(size=(frames, 17, 3)).astype(np.float32))


def test_cross_environment_split_and_temporal_sample(tmp_path):
    create_record(tmp_path, "E01", "S01")
    create_record(tmp_path, "E02", "S11")
    create_record(tmp_path, "E03", "S21")
    create_record(tmp_path, "E04", "S31")
    config = {
        "protocol": "P3",
        "split": "cross_environment",
        "window_size": 3,
        "signal": {"subcarrier_spacing_hz": 312500.0},
        "train_environments": ["E01", "E02", "E03"],
        "test_environments": ["E04"],
    }
    train_dataset = MMFiPoseDataset(tmp_path, config, "train")
    test_dataset = MMFiPoseDataset(tmp_path, config, "val")
    assert len(train_dataset.records) == 3
    assert len(test_dataset.records) == 1
    sample = test_dataset[1]
    assert sample["dynamic"].shape == (36, 2)
    assert sample["static"].shape == (36, 2)
    assert sample["pose"].shape == (17, 3)
    assert sample["environment"] == 3


def test_pa_mpjpe_removes_rotation_translation_and_scale():
    rng = np.random.default_rng(3)
    target = rng.normal(size=(2, 17, 3))
    angle = 0.6
    rotation = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    prediction = 2.3 * (target @ rotation) + np.array([4.0, -3.0, 2.0])
    assert mpjpe(prediction, target) > 1.0
    assert pa_mpjpe(prediction, target) < 1e-8


def test_pck_uses_torso_normalized_threshold():
    target = np.zeros((1, 17, 3), dtype=np.float64)
    target[0, 5] = [0.0, 0.0, 0.0]
    target[0, 12] = [0.0, 100.0, 0.0]
    prediction = target.copy()
    prediction[0, 0] = [0.0, 10.0, 0.0]
    prediction[0, 1] = [0.0, 30.0, 0.0]
    assert pck(prediction, target, 20) == (16 / 17)
    assert pck(prediction, target, 50) == 1.0


def test_model_forward_shapes_and_unknown_environment_adapter():
    torch.manual_seed(1)
    model = build_model(tiny_config())
    dynamic = torch.randn(4, 36, 2)
    static = torch.randn(4, 36, 2)
    environments = torch.tensor([0, 1, -1, -1])
    outputs = model(dynamic, static, environments)
    assert outputs["pose"].shape == (4, 17, 3)
    assert outputs["action_logits"].shape == (4, 5)
    assert outputs["dynamic_feature"].shape == (4, 16)
    assert torch.isfinite(outputs["pose"]).all()


def test_joint_objective_backpropagates():
    torch.manual_seed(2)
    model = build_model(tiny_config())
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    dynamic = torch.randn(4, 36, 2)
    static = torch.randn(4, 36, 2)
    target_pose = torch.randn(4, 17, 3)
    target_action = torch.tensor([0, 0, 1, 1])
    target_environment = torch.tensor([0, 1, 0, 1])
    outputs = model(dynamic, static, target_environment)
    loss, components = total_objective(
        outputs, target_pose, target_action, target_environment,
        action_weight=0.5, domain_weight=0.2, temperature=0.07,
    )
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert set(components) == {"pose", "action", "domain", "total"}
    assert model.kernel_hypernetwork.weight.grad is not None


def test_forward_does_not_accept_ground_truth_action_labels():
    parameters = inspect.signature(build_model(tiny_config()).forward).parameters
    assert "action_label" not in parameters
