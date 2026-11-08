"""Invariants of the rigid-body layer (src/rh/physics.py) and of the pistol routine built on it.

python3 -m unittest discover tests      (no Blender; build/pocket.json is the only artifact used)
"""
import functools
import os
import sys
import unittest

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
np.seterr(all="ignore")  # Accelerate's arm64 matmul raises spurious FP warnings

from rh import gunplay, motions, physics as ph, rig  # noqa: E402
from rh.pose import Pose, Skeleton  # noqa: E402

SKEL = Skeleton({k: {"head": v[0].tolist(), "tail": v[1].tolist(), "parent": v[2], "z": v[3].tolist()}
                 for k, v in rig.all_bones().items()})


def standing(arms_forward=False):
    p = Pose(SKEL).root((0, 0, motions.LIFT), 0)
    motions.stand(SKEL, p)
    motions.hang_arms(p, (0, 0), 0.0)
    if arms_forward:
        _, _, _, H = p.solve()
        for side, sg in (("l", 1.0), ("r", -1.0)):
            sh = H[f"upperarm_{side}"]
            p.arm(side, sh + np.array([sg * 0.05, -0.52, -0.05]), sh + np.array([sg * 0.3, 0.0, -0.4]))
    return motions.solve(p)


def momentum(sim, about):
    p, L = np.zeros(3), np.zeros(3)
    for b in sim.B + sim.extra:
        mv = b.mass * np.array(b.v)
        p += mv
        L += np.cross(np.array(b.x) - about, mv) + np.array(b.Iw(b.w))
    return p, L


class FreeFlight(unittest.TestCase):
    def test_muscles_cannot_move_the_centre_of_mass_or_spin_the_body(self):
        # internal forces (sockets, limits, drives, gravity compensation, damping) must come in
        # equal and opposite pairs: a sign slip anywhere lets a ragdoll swim through the air
        sim = ph.Sim(SKEL, standing(), substeps=96)     # rotation drift is first order in the step
        sim.contacts_on = False
        rng = np.random.default_rng(0)
        for b in sim.B:
            b.x = (b.x[0], b.x[1], b.x[2] + 5.0)
            b.v, b.w = tuple(rng.normal(0, 1.0, 3)), tuple(rng.normal(0, 3.0, 3))
        sim.set_tone(motions.ALL_KINDS, 4.0, 0.7, 1.0)
        brace = motions._brace_pose(SKEL)                # far from the start: the drives pull hard
        mass = sum(b.mass for b in sim.B)
        c0 = np.array(sim.com())
        p0, L0 = momentum(sim, c0)
        n = 30
        for _ in range(n):
            sim.step(brace)
        t, h = n / ph.FPS, sim.h
        c = np.array(sim.com())
        # semi-implicit Euler: x = x0 + v0 t + g t (t + h) / 2 exactly
        want = c0 + p0 / mass * t + np.array([0.0, 0.0, ph.G]) * t * (t + h) / 2
        self.assertLess(np.linalg.norm(c - want), 1e-3)
        _, L1 = momentum(sim, c)
        self.assertLess(np.linalg.norm(L1 - L0) / np.linalg.norm(L0), 0.03)


class Tone(unittest.TestCase):
    def test_held_arms_do_not_sag(self):
        # gravity compensation must cancel the arms' weight: outstretched arms that hang 5 cm
        # low would turn every aim and reach into a droop
        f = standing(arms_forward=True)
        out, _ = ph.track(SKEL, [f] * 45)
        for side in ("l", "r"):
            self.assertLess(np.linalg.norm(out[-1].H[f"hand_{side}"] - f.H[f"hand_{side}"]), 0.01, side)


@functools.lru_cache(maxsize=None)
def pistol():
    return gunplay.pistol_sequence(SKEL, ROOT)


def rel_error(a, b):
    d = np.linalg.inv(a) @ b
    ang = np.degrees(np.arccos(np.clip((np.trace(d[:3, :3]) - 1) / 2, -1, 1)))
    return np.linalg.norm(d[:3, 3]), ang


@unittest.skipUnless(os.path.exists(os.path.join(ROOT, "build", "pocket.json")),
                     "needs build/pocket.json (python3 src/build_clothes.py)")
class Pistol(unittest.TestCase):
    def test_support_hand_stays_on_the_pistol(self):
        # the support hand is simulated and welded onto the pistol; it must sit where the grip
        # was solved (fingers round the firing hand, thumb along the frame), shots included
        frames = pistol()
        T = gunplay.T
        worst = (0.0, 0.0)
        for i, f in enumerate(frames):
            t = i / ph.FPS
            if T["meet"] + 0.15 <= t <= T["off"] - 0.03 or T["back"] + 0.15 <= t <= T["off2"] - 0.03:
                worst = max(worst, rel_error(f.extra["gun"] @ gunplay.SUPPORT, gunplay.bone_M(f.W, f.H, "hand_l")))
        self.assertLess(worst[0], 0.01, worst)
        self.assertLess(worst[1], 5.0, worst)

    def test_each_shot_flips_the_muzzle_and_it_comes_back_on_target(self):
        frames = pistol()
        pitch = [np.degrees(np.arcsin(np.clip(f.extra["gun"][2, 1], -1, 1))) for f in frames]
        for k, ts in enumerate(gunplay.SHOTS):
            i = int(round(ts * ph.FPS))
            rise = max(pitch[i:i + 6]) - pitch[i - 1]
            self.assertGreater(rise, 2.5, f"shot at {ts}: muzzle rose {rise:.1f} deg")
            if k < len(gunplay.SHOTS) - 1:                 # after the last he brings it in to reload
                back = abs(pitch[i + 9] - pitch[i - 1])
                self.assertLess(back, 2.0, f"shot at {ts}: {back:.1f} deg off target 0.3 s later")


if __name__ == "__main__":
    unittest.main()
