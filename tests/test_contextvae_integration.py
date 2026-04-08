import math
import unittest

from nusc_scene_agent.contextvae_integration import _case_file_name, _contextvae_local_to_global


class ContextVAEIntegrationTest(unittest.TestCase):
    def test_case_file_name_uses_rollout_anchor(self) -> None:
        name = _case_file_name(
            {
                "instance_token": "instance-123",
                "rollout_anchor_sample_token": "sample-456",
            }
        )
        self.assertEqual(name, "instance-123_sample-456")

    def test_contextvae_local_to_global_identity(self) -> None:
        restored = _contextvae_local_to_global(
            trajectory=[[10.0, 5.0], [12.0, 7.0]],
            origin_x=10.0,
            origin_y=5.0,
            heading_rad=0.0,
        )
        self.assertEqual(restored, [[10.0, 5.0], [12.0, 7.0]])

    def test_contextvae_local_to_global_rotates_back_to_global_frame(self) -> None:
        restored = _contextvae_local_to_global(
            trajectory=[[11.0, 5.0], [12.0, 5.0]],
            origin_x=10.0,
            origin_y=5.0,
            heading_rad=math.pi / 2.0,
        )
        self.assertEqual(restored, [[10.0, 6.0], [10.0, 7.0]])


if __name__ == "__main__":
    unittest.main()
