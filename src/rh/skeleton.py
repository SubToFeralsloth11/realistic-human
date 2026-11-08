"""Skeleton of a 1.80 m adult male in an A-pose, plus local authoring frames for each body part.

World axes (Blender): +Z up, the character faces -Y, his left is +X. Right-side parts are built
by mirroring the left side (see sdf.Node.mirrored).
Proportions follow adult male anthropometry: head height 0.236 m, biacromial ~0.40 m,
hip joints 0.176 m apart, upper arm 0.30 m, forearm 0.258 m, thigh 0.428 m, shank 0.415 m.
"""
import numpy as np

from .sdf import norm, rot_from_axes

STATURE = 1.80


def rodrigues(v, axis, deg):
    axis = norm(axis)
    a = np.radians(deg)
    v = np.asarray(v, float)
    return v * np.cos(a) + np.cross(axis, v) * np.sin(a) + axis * np.dot(axis, v) * (1 - np.cos(a))


class Frame:
    """Origin + orthonormal axes. Local coordinates (x, y, z) map to origin + x*ex + y*ey + z*ez."""

    def __init__(self, origin, R):
        self.o = np.asarray(origin, float)
        self.R = np.asarray(R, float)

    @property
    def ex(self):
        return self.R[:, 0]

    @property
    def ey(self):
        return self.R[:, 1]

    @property
    def ez(self):
        return self.R[:, 2]

    def p(self, x, y=None, z=None):
        if y is None:
            x, y, z = x
        return self.o + self.R @ np.array([x, y, z], float)

    def rot(self, R_local=None):
        return self.R if R_local is None else self.R @ R_local

    def sub(self, origin_local, R_local=None):
        return Frame(self.p(origin_local), self.rot(R_local))


def axis_rot(axis, deg):
    """3x3 rotation of deg degrees about a unit axis."""
    return np.stack([rodrigues(e, axis, deg) for e in np.eye(3)], axis=1)


# ---------------------------------------------------------------- joints (left side)
J = {}
J["root"] = np.array([0.0, 0.0, 0.0])
J["pelvis"] = np.array([0.0, 0.0, 0.95])
J["spine_01"] = np.array([0.0, 0.02, 1.03])
J["spine_02"] = np.array([0.0, 0.025, 1.12])
J["spine_03"] = np.array([0.0, 0.04, 1.23])
J["spine_04"] = np.array([0.0, 0.05, 1.35])
J["neck_01"] = np.array([0.0, 0.045, 1.48])
J["neck_02"] = np.array([0.0, 0.03, 1.565])
J["head"] = np.array([0.0, 0.02, 1.625])
J["head_end"] = np.array([0.0, 0.02, 1.80])

J["clavicle_l"] = np.array([0.025, -0.045, 1.455])
J["upperarm_l"] = np.array([0.178, 0.012, 1.447])     # glenohumeral joint centre

ARM_ABDUCT = 48.0        # degrees from vertical
ELBOW_FLEX = 0.0        # straight limbs in the rest pose: one continuous loft per limb
UPPERARM_LEN = 0.300
FOREARM_LEN = 0.258
_up_dir = norm([np.sin(np.radians(ARM_ABDUCT)), -0.06, -np.cos(np.radians(ARM_ABDUCT))])
J["lowerarm_l"] = J["upperarm_l"] + UPPERARM_LEN * _up_dir
_flex_axis = norm(np.cross(_up_dir, [0, -1, 0]))
_fo_dir = rodrigues(_up_dir, _flex_axis, -ELBOW_FLEX)
if _fo_dir[1] > _up_dir[1]:   # flexion must bring the forearm forward (-Y)
    _fo_dir = rodrigues(_up_dir, _flex_axis, ELBOW_FLEX)
J["hand_l"] = J["lowerarm_l"] + FOREARM_LEN * _fo_dir

J["thigh_l"] = np.array([0.088, -0.005, 0.925])        # femoral head centre
J["foot_l"] = np.array([0.104, 0.018, 0.082])          # ankle (talocrural) centre
J["calf_l"] = J["thigh_l"] + (0.925 - 0.497) / (0.925 - 0.082) * (J["foot_l"] - J["thigh_l"])   # knee on the line
FOOT_TOE_OUT = 8.0
_foot_dir = norm(rodrigues([0.0, -1.0, 0.0], [0, 0, 1], -FOOT_TOE_OUT))   # toes point slightly outward (+X)
J["ball_l"] = J["foot_l"] * [1, 1, 0] + _foot_dir * 0.142 + [0.0, 0.0, 0.024]
J["toe_end_l"] = J["foot_l"] * [1, 1, 0] + _foot_dir * 0.222 + [0.0, 0.0, 0.016]

# ---------------------------------------------------------------- authoring frames
# Upper arm: x anterior, y distal, z superior-lateral (the top of the arm in the A-pose).
_a = np.radians(ARM_ABDUCT)
UPPERARM = Frame(J["upperarm_l"], rot_from_axes(y=_up_dir, z=[np.cos(_a), 0.0, np.sin(_a)]))
assert UPPERARM.ex[1] < 0, "upper arm x axis should point forward"

# Forearm / hand: x radial (thumb side, forward), y distal, z dorsal (back of the hand).
_fa_y = _fo_dir
_fa_x = norm(np.array([0.0, -1.0, 0.0]) - _fa_y * np.dot([0.0, -1.0, 0.0], _fa_y))
FOREARM = Frame(J["lowerarm_l"], rot_from_axes(x=_fa_x, y=_fa_y))
HAND = Frame(J["hand_l"], FOREARM.R @ axis_rot([1, 0, 0], 4.0))   # slight wrist extension

# Thigh / shank: x lateral, y distal, z posterior.
THIGH = Frame(J["thigh_l"], rot_from_axes(x=[1, 0, 0], y=J["calf_l"] - J["thigh_l"]))
SHANK = Frame(J["calf_l"], rot_from_axes(x=[1, 0, 0], y=J["foot_l"] - J["calf_l"]))
# Foot: x lateral, y forward (towards the toes); right-handed, so z points down. Author foot
# shapes with foot_p(lateral, forward, up) instead of the frame's z.
FOOT = Frame(J["foot_l"], rot_from_axes(x=-np.cross(_foot_dir, [0, 0, 1]), y=_foot_dir))

# Head: origin between the eyeball centres, world-aligned (Frankfort plane horizontal).
HEAD = Frame(np.array([0.0, -0.062, 1.682]), np.eye(3))


def foot_p(lateral, forward, up):
    """Point in the left foot's space measured from the ankle joint centre."""
    return J["foot_l"] + lateral * FOOT.ex + forward * FOOT.ey + up * np.array([0.0, 0.0, 1.0])


def mirror_x(p):
    p = np.asarray(p, float).copy()
    p[..., 0] *= -1
    return p


# Bones for the rig: name -> (head joint, tail point, parent). Right side is mirrored.
def bone_table():
    b = {
        "pelvis": (J["pelvis"], J["spine_01"], None),
        "spine_01": (J["spine_01"], J["spine_02"], "pelvis"),
        "spine_02": (J["spine_02"], J["spine_03"], "spine_01"),
        "spine_03": (J["spine_03"], J["spine_04"], "spine_02"),
        "spine_04": (J["spine_04"], J["neck_01"], "spine_03"),
        "neck_01": (J["neck_01"], J["neck_02"], "spine_04"),
        "neck_02": (J["neck_02"], J["head"], "neck_01"),
        "head": (J["head"], J["head_end"], "neck_02"),
        "clavicle_l": (J["clavicle_l"], J["upperarm_l"], "spine_04"),
        "upperarm_l": (J["upperarm_l"], J["lowerarm_l"], "clavicle_l"),
        "lowerarm_l": (J["lowerarm_l"], J["hand_l"], "upperarm_l"),
        "hand_l": (J["hand_l"], HAND.p(0, 0.094, 0), "lowerarm_l"),
        "thigh_l": (J["thigh_l"], J["calf_l"], "pelvis"),
        "calf_l": (J["calf_l"], J["foot_l"], "thigh_l"),
        "foot_l": (J["foot_l"], J["ball_l"], "calf_l"),
        "ball_l": (J["ball_l"], J["toe_end_l"], "foot_l"),
    }
    return b
