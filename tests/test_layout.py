import pytest
import torch

from vqvae_latent_actions.layout import UnifiedLayout


def test_from_yaml_matches_the_fork_layout(layout):
    assert layout.total_dim == 119 and layout.num_groups == 17
    iv = layout.intervals()
    assert iv["left_arm.joints"] == (0, 8)
    assert iv["left_arm.cartesian_position"] == (8, 11)
    assert iv["left_arm.cartesian_rotation"] == (11, 17)      # rotation_6d -> 6 dims
    assert iv["left_arm.gripper"] == (17, 18)
    assert iv["left_hand.joints"] == (18, 42)
    assert iv["right_arm.joints"] == (42, 50)
    assert iv["right_arm.gripper"] == (59, 60)
    assert iv["right_hand.joints"] == (60, 84)
    assert iv["head.joints"] == (84, 87)
    assert iv["torso.joints"] == (87, 92)
    assert iv["base.velocity"] == (92, 98)
    assert iv["base.wheel_phase"] == (98, 104)
    assert iv["left_leg.joints"] == (104, 110)
    assert iv["right_leg.joints"] == (110, 116)
    assert iv["base.wheel_joints"] == (116, 119)


def test_group_index_and_membership(layout):
    gi = layout.group_index()
    m = layout.membership()
    assert gi.shape == (119,) and gi.dtype == torch.long
    assert int(gi[0]) == 0 and int(gi[18]) == 4 and int(gi[118]) == 16
    assert m.shape == (17, 119) and m.dtype == torch.bool
    assert int(m.sum()) == 119                                  # every dim belongs to exactly one group
    assert int(m.sum(dim=1)[4]) == 24                            # left_hand.joints
    assert bool(m[gi, torch.arange(119)].all())                  # membership agrees with group_index


def test_dim_weights(layout):
    w = layout.dim_weights({"left_arm.gripper": 5.0, "base.velocity": 2.0})
    assert w.shape == (119,)
    assert float(w[17]) == 5.0 and float(w[92]) == 2.0 and float(w[0]) == 1.0
    assert torch.equal(layout.dim_weights(None), torch.ones(119))


def test_from_intervals_rejects_gaps_and_overlaps():
    with pytest.raises(ValueError):
        UnifiedLayout.from_intervals({"a": (0, 2), "b": (3, 5)})
    with pytest.raises(ValueError):
        UnifiedLayout.from_intervals({"a": (0, 3), "b": (2, 5)})
    with pytest.raises(ValueError):
        UnifiedLayout.from_intervals({"a": (0, 0)})


def test_dict_roundtrip(layout):
    again = UnifiedLayout.from_dict(layout.to_dict())
    assert again == layout and again.intervals() == layout.intervals()
