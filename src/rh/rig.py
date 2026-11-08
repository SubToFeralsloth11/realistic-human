"""Bone hierarchy (Unreal-mannequin naming) derived from the same skeleton the body was sculpted on.

Each bone: name -> (head, tail, parent, roll_z_axis). The roll axis is a world vector the bone's
local Z should point along (Blender bone convention: Y runs head -> tail).
Right-side bones are mirrored from the left (x -> -x).
"""
import numpy as np

from . import hand as handmod
from . import head as headmod
from .skeleton import FOOT, HAND, J, UPPERARM, norm


def _mirror_name(n):
    return n[:-2] + "_r" if n.endswith("_l") else n


def _digit_joints(name):
    if name == "thumb":
        base, d, dor, L, r, flex = handmod.THUMB
        R = HAND.R
        pts, frames = handmod._chain(HAND.p(*base), R @ np.array(d), R @ np.array(dor), L, flex)
    else:
        base, spread, L, r, flex = handmod.DIGITS[name]
        s = np.radians(spread)
        d = HAND.ey * np.cos(s) + HAND.ex * np.sin(s)
        pts, frames = handmod._chain(HAND.p(*base), d, HAND.ez, L, flex)
    return pts, frames


def left_bones():
    b = {}
    up = np.array([0.0, -1.0, 0.0])     # spine/neck/head bones: local Z forward (-Y)
    b["pelvis"] = (J["pelvis"], J["spine_01"], "root", up)
    b["spine_01"] = (J["spine_01"], J["spine_02"], "pelvis", up)
    b["spine_02"] = (J["spine_02"], J["spine_03"], "spine_01", up)
    b["spine_03"] = (J["spine_03"], J["spine_04"], "spine_02", up)
    b["spine_04"] = (J["spine_04"], J["neck_01"], "spine_03", up)
    b["neck_01"] = (J["neck_01"], J["neck_02"], "spine_04", up)
    b["neck_02"] = (J["neck_02"], J["head"], "neck_01", up)
    b["head"] = (J["head"], J["head_end"], "neck_02", up)
    O = headmod.O
    b["jaw"] = (O + [0, 0.058, -0.034], O + [0, -0.018, -0.104], "head", np.array([0.0, 0.0, 1.0]))
    for k in ("eye_l", "eye_r"):
        c = headmod.LANDMARKS[k]
        b[k] = (c, c + [0, -0.025, 0], "head", np.array([0.0, 0.0, 1.0]))
    # eyelids pivot on the globe centre: rotating about local X slides the lid shell over the eye
    c = headmod.LANDMARKS["eye_l"]
    b["lid_upper_l"] = (c, c + [0, -0.020, 0], "head", np.array([0.0, 0.0, 1.0]))
    b["lid_lower_l"] = (c, c + [0, -0.018, 0], "head", np.array([0.0, 0.0, 1.0]))
    # arm: local Z of the arm bones = the upper-arm frame's superior-lateral axis
    arm_z = UPPERARM.ez
    elbow, wrist = J["lowerarm_l"], J["hand_l"]
    b["clavicle_l"] = (J["clavicle_l"], J["upperarm_l"], "spine_04", np.array([0.0, 0.0, 1.0]))
    b["upperarm_l"] = (J["upperarm_l"], elbow, "clavicle_l", arm_z)
    b["lowerarm_l"] = (elbow, wrist, "upperarm_l", arm_z)
    b["lowerarm_twist_01_l"] = (0.5 * (elbow + wrist), wrist, "lowerarm_l", arm_z)
    b["upperarm_twist_01_l"] = (0.5 * (J["upperarm_l"] + elbow), elbow, "upperarm_l", arm_z)
    b["hand_l"] = (wrist, HAND.p(0.0085, 0.092, 0.0), "lowerarm_l", HAND.ez)
    for name in ("thumb", "index", "middle", "ring", "pinky"):
        pts, frames = _digit_joints(name)
        parent = "hand_l"
        for i in range(3):
            bn = f"{name}_{i + 1:02d}_l"
            dd, dor, lat = frames[i]
            b[bn] = (pts[i], pts[i + 1], parent, dor)
            parent = bn
    # leg (local Z forward)
    fwd = np.array([0.0, -1.0, 0.0])
    b["thigh_l"] = (J["thigh_l"], J["calf_l"], "pelvis", fwd)
    b["calf_l"] = (J["calf_l"], J["foot_l"], "thigh_l", fwd)
    b["foot_l"] = (J["foot_l"], J["ball_l"], "calf_l", np.array([0.0, 0.0, 1.0]))
    b["ball_l"] = (J["ball_l"], J["toe_end_l"], "foot_l", np.array([0.0, 0.0, 1.0]))
    return b


def all_bones():
    """Full table including root and mirrored right side."""
    b = {"root": (np.zeros(3), np.array([0.0, 0.0, 0.25]), None, np.array([0.0, -1.0, 0.0]))}
    left = left_bones()
    S = np.array([-1.0, 1.0, 1.0])
    for name, (h, t, p, z) in left.items():
        b[name] = (np.asarray(h, float), np.asarray(t, float), p, np.asarray(z, float))
        if name.endswith("_l"):
            b[_mirror_name(name)] = (h * S, t * S, _mirror_name(p) if p else p, z * S)
    return b
