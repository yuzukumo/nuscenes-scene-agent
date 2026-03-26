import unittest

import numpy as np

from nusc_scene_agent.geometry import ego_xy_to_global, global_xy_to_anchor_ego


class GeometryRoundTripTest(unittest.TestCase):
    def test_ego_to_global_and_back(self) -> None:
        ego_xy = np.asarray([10.0, -4.0], dtype=float)
        ego_yaw = np.deg2rad(30.0)
        local_point = np.asarray([3.0, 1.5], dtype=float)

        global_point = ego_xy_to_global(local_point, ego_xy, ego_yaw)
        recovered = global_xy_to_anchor_ego(np.asarray([global_point]), ego_xy, ego_yaw)[0]

        self.assertTrue(np.allclose(recovered, local_point, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
