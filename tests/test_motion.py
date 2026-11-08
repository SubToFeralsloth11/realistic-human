"""Regression tests for the pose engine, the ragdoll hand-off and the wrist blend.

python3 -m unittest discover tests      (no Blender, no build/ artifacts needed; ~30 s)
"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
np.seterr(all="ignore")  # Accelerate's arm64 matmul raises spurious FP warnings

from rh import motions, rig  # noqa: E402
from rh.pose import Pose, Skeleton  # noqa: E402


def skeleton():
    bones = rig.all_bones()
    return Skeleton({k: {"head": v[0].tolist(), "tail": v[1].tolist(), "parent": v[2], "z": v[3].tolist()}
                     for k, v in bones.items()})


SKEL = skeleton()


def roll_about_own_axis(s, W, bone):
    """Signed roll (deg) of a posed bone about its own long axis relative to its rest frame."""
    R = s.A[bone].T @ W[bone]
    return np.degrees(np.arctan2(R[0, 2] - R[2, 0], R[0, 0] + R[2, 2]))


class TwoBoneIK(unittest.TestCase):
    def test_standing_legs_and_arms_do_not_roll(self):
        # the hinge used to be taken as each bone's local Z, which is forward for the legs:
        # thighs and calves came out rolled ~90 degrees
        p = Pose(SKEL).root((0, 0, motions.LIFT), 0)
        motions.stand(SKEL, p)
        motions.hang_arms(p, (0, 0), 0.0)
        _, _, W, _ = p.solve()
        for b in ("thigh_l", "calf_l", "thigh_r", "calf_r", "upperarm_l", "lowerarm_r"):
            self.assertLess(abs(roll_about_own_axis(SKEL, W, b)), 15.0, b)

    def test_leg_reaches_target_with_knee_toward_pole(self):
        p = Pose(SKEL).root((0, 0, 0), 0)
        target = SKEL.head["foot_l"] + np.array([0.0, -0.15, 0.20])
        p.leg("l", target, SKEL.head["calf_l"] + np.array([0.0, -0.5, 0.0]))
        _, _, _, H = p.solve()
        self.assertLess(np.linalg.norm(H["foot_l"] - target), 1e-3)
        self.assertLess(H["calf_l"][1], SKEL.head["calf_l"][1])      # knee bends forward (-Y)


class RagdollHandOff(unittest.TestCase):
    def check_floor(self, clip, limit=0.015):
        frames, _ = motions.CLIPS[clip](SKEL)
        worst, where = 0.0, None
        for i, f in enumerate(frames):
            for b, h in f.H.items():
                if b == "root":                       # the rig root follows the pelvis down; not a body part
                    continue
                tail = h + f.W[b][:, 1] * SKEL.length[b]
                z = min(h[2], tail[2])
                if z < worst:
                    worst, where = z, (b, i)
        self.assertGreater(worst, -limit, f"{clip}: {where} reaches z={worst:.3f}")

    def test_fall_keeps_every_bone_on_the_ground(self):
        # torso roll was dropped and the clavicles copied the chest, so FK arms ended 8-16 cm
        # under the floor while the simulated particles lay on it
        self.check_floor("fall")

    def test_shot_keeps_every_bone_on_the_ground(self):
        self.check_floor("shot")


class WristBlend(unittest.TestCase):
    def test_forearm_runs_into_the_hand_without_a_ridge(self):
        # the forearm loft ended in a flat 59 x 38 mm cap at the joint, 5 mm proud of the carpus
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
        from build_body import make_program
        from rh.skeleton import J
        prog = make_program()
        gh, wr = J["upperarm_l"], J["hand_l"]
        n = (wr - gh) / np.linalg.norm(wr - gh)
        e1 = np.cross(n, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        e2 = np.cross(n, e1)
        ts = np.arange(-0.012, 0.0091, 0.003)       # the thenar eminence rises ~1 cm past the joint
        angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
        radius = np.zeros((len(ts), len(angles)))
        for i, t in enumerate(ts):
            for j, a in enumerate(angles):
                d = np.cos(a) * e1 + np.sin(a) * e2
                lo, hi = 0.0, 0.06                     # inside at the axis, outside at 6 cm
                for _ in range(30):
                    mid = 0.5 * (lo + hi)
                    if prog(np.array([wr + n * t + d * mid]), 0.05)[0] < 0:
                        lo = mid
                    else:
                        hi = mid
                radius[i, j] = lo
        step = np.abs(np.diff(radius, axis=0)).max()
        self.assertLess(step, 0.0025, f"radius jumps {step * 1000:.1f} mm within 3 mm along the wrist")


if __name__ == "__main__":
    unittest.main()
