"""Pistol handling: draw from the right front jeans pocket, two-handed aim, three shots, reload
from the left pocket, scan, return the pistol to the pocket.

Two passes:
  1. kinematic plan: stance, gaze, where the pistol goes (key poses, below) and where each hand
     sits on it. The firing hand holds the grip through GRIP (palm on the right panel, fingers
     round the front strap, index along the frame until it goes to the trigger); the support hand
     fills the left panel through SUPPORT (thumbs-forward grip: palm on the left panel under the
     firing thumb, fingers wrapped over the firing fingers, thumb along the frame).
  2. rigid-body pass (src/rh/physics.py): pelvis and legs follow the plan, trunk, head and arms are
     driven toward it. The pistol is a 0.75 kg body welded to the right hand while held and the
     support hand is welded onto it in the two-handed grip, so each shot (a 3.2 N s impulse along
     the bore, which sits above the wrists) produces the muzzle flip, the arm and shoulder recoil
     and the recovery by itself.
Props are posed from the simulated bones: in the pocket the pistol rides the right thigh; the
dropped magazine and the casings are ballistic bodies with floor contact.
Pistol part conventions (blender/pistol.py): +Y muzzle, +Z sights, +X gun's right side; root =
palm centre on the grip axis 25 mm under the slide; slide travel -0.028 m along Y; trigger
rotates -12 deg about X; magazine slides out along its own -Z from (0, 0.0141, 0.0325) tilted
-18 deg about X.
"""
import json
import os

import numpy as np

from . import physics as ph
from .motions import FPS, LIFT, blink_at, hang_arms, interp_keys, smooth, solve, stand
from .pose import Pose, axis_angle, nrm, quat_from_matrix

# firing hand: pistol root in the right hand bone frame (hand axes: x ulnar, y distal, z dorsal)
GRIP_POS = np.array([0.0, 0.050, -0.030])
GRIP_ROT = np.stack([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], axis=1) @ axis_angle([1, 0, 0], 12.0)
# support hand: left hand bone frame in pistol space (hand axes: x radial, y distal, z dorsal).
# Metacarpals run forward along the frame (slightly up from the wrist), the index side on top so
# the thumb lies along the frame, the back of the hand faces left and a little down.
_SY = nrm([0.0, 1.0, 0.15])
_SZ = nrm(np.array([-1.0, 0.0, -0.35]) - _SY * np.dot([-1.0, 0.0, -0.35], _SY))
SUPPORT_ROT = np.stack([np.cross(_SY, _SZ), _SY, _SZ], 1)
SUPPORT_POS = np.array([-0.025, 0.040, -0.030]) - 0.092 * _SY      # wrist, from the middle knuckle
# Digit joint flexion (MCP, PIP, DIP deg; MCP spread deg) solved so each fingertip lands where it
# belongs and no segment enters the grip: firing fingers round the front strap with the tips on
# the left panel, the support fingers round the firing fingers onto the right side.
GRIP_FINGERS = {"middle": (68.1, 62.1, 29.1, 12.1), "ring": (64.8, 59.0, 28.1, 12.1), "pinky": (65.6, 36.9, 31.7, 9.1)}
INDEX_OFF = (-25.0, 18.5, 49.7, -13.5)            # straight along the frame, above the trigger guard
INDEX_ON = (38.0, 61.9, 85.0, 15.0)               # pad on the trigger face
SUPPORT_FINGERS = {"index": (58.8, -5.0, 82.4, 15.0), "middle": (44.5, 43.6, 62.8, 15.0),
                   "ring": (35.1, 64.8, 32.6, 6.6), "pinky": (37.7, 45.7, 80.0, -14.0)}
# thumbs: (rotation about the thumb base's x, then z; flexion of the three joints)
GRIP_THUMB = (39.9, -28.9, (14.2, 50.2, 34.7))     # over the frame to the left side, under the slide
SUPPORT_THUMB = (-9.2, 39.6, (-8.3, -11.6, -14.9))  # straight forward along the left side of the frame
RELAXED = 0.35                                     # finger curl of a hand at rest
DIGIT_MAX = (85.0, 100.0, 70.0)                     # Pose.fingers' joint flexion at curl 1
THUMB_MAX = (25.0, 35.0, 45.0)
MAG_FINGERS = {f: tuple(m * 0.62 for m in DIGIT_MAX) + (0.0,) for f in ("middle", "ring", "pinky")}
MAG_INDEX = tuple(m * 0.15 for m in DIGIT_MAX) + (0.0,)   # index finger along the magazine's front

MAG_LOCAL = (np.array([0.0, 0.0141, 0.0325]), axis_angle([1, 0, 0], -18.0))
# spare magazine in the left hand: baseplate in the palm, index finger along its front
MAG_HOLD_ROT = np.stack([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], axis=1)
MAG_HOLD_POS = np.array([0.0, 0.125, -0.026])
PORT_LOCAL = np.array([0.008, 0.036, 0.046])
MUZZLE_LOCAL = np.array([0.0, 0.1121, 0.0395])
BORE_LOCAL = np.array([0.0, 0.040, 0.0395])         # on the bore axis above the trigger
SIGHT_LOCAL = np.array([0.0, -0.035, 0.0545])       # rear sight notch
POCKET_DEPTH = 0.088                                # root below the pocket mouth: grip just under the denim
GUN_MASS = 0.75                                     # loaded compact 9 mm, kg
GUN_COM = np.array([0.0, 0.020, 0.012])
GUN_INERTIA = (0.0031, 0.0011, 0.0021)              # about the COM, pistol axes
RECOIL = 3.2                                        # N s: 7.5 g at 360 m/s plus the powder gas

# timeline (s)
T = dict(reach=0.55, pocket=1.00, grip=1.32, pulled=1.62, retention=1.90, meet=2.08, compressed=2.15, aim=2.52,
         trigger_on=2.75, trigger_off=3.92, off=4.05, workspace=4.30, l_pocket=4.50, l_grab=4.62, l_hold=4.68,
         l_out=4.98, l_below=5.30, seated=5.44, slap=5.50, back=5.72, aim2=6.10, trigger2_on=6.15,
         trigger2_off=6.70, hold2=6.75, low=7.15, off2=7.95, pocket_up=8.45, holstered=8.72, hand_out=9.02,
         hang=9.62)
SHOTS = (3.05, 3.42, 3.79)
MAG_RELEASE = 4.30
MAG_SEATED = T["seated"]
SLIDE_RELEASE = T["back"] - 0.02
SEGMENTS = {"draw": (0.4, 2.9), "shoot": (2.7, 4.4), "reload": (4.0, 6.5), "holster": (6.5, 10.0)}
SECONDS = 10.0
TARGET = np.array([-0.03, -8.0, 1.30])              # the target he engages, 8 m down range
G = np.array([0.0, 0.0, -9.81])


def M(R, p):
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = p
    return m


def inv(m):
    r = m[:3, :3].T
    return M(r, -r @ m[:3, 3])


def slerp_R(Ra, Rb, t):
    qa, qb = quat_from_matrix(Ra), quat_from_matrix(Rb)
    if np.dot(qa, qb) < 0:
        qb = -qb
    d = np.clip(np.dot(qa, qb), -1, 1)
    if d > 0.9995:
        q = qa + t * (qb - qa)
    else:
        th = np.arccos(d)
        q = (np.sin((1 - t) * th) * qa + np.sin(t * th) * qb) / np.sin(th)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def lerp_M(a, b, u):
    u = float(np.clip(u, 0.0, 1.0))
    return M(slerp_R(a[:3, :3], b[:3, :3], u), a[:3, 3] + (b[:3, 3] - a[:3, 3]) * u)


def interp_pose(t, keys):
    """keys: [(time, 4x4)] -> 4x4. Positions: non-uniform Catmull-Rom (velocity carries through
    passing keys) with zero velocity at the ends and wherever a pose is held, so a hand never
    overshoots the place it is going to stop; rotations: eased slerp per segment."""
    ts = [k[0] for k in keys]
    if t <= ts[0]:
        return keys[0][1].copy()
    if t >= ts[-1]:
        return keys[-1][1].copy()
    P = [k[1][:3, 3] for k in keys]

    def tangent(j):
        if j == 0 or j == len(keys) - 1 or np.allclose(P[j], P[j - 1]) or np.allclose(P[j], P[j + 1]):
            return np.zeros(3)
        return (P[j + 1] - P[j - 1]) / (ts[j + 1] - ts[j - 1])

    i = int(np.searchsorted(ts, t) - 1)
    h = ts[i + 1] - ts[i]
    u = (t - ts[i]) / h
    m0, m1 = tangent(i) * h, tangent(i + 1) * h
    pos = (2 * u ** 3 - 3 * u ** 2 + 1) * P[i] + (u ** 3 - 2 * u ** 2 + u) * m0 + \
        (-2 * u ** 3 + 3 * u ** 2) * P[i + 1] + (u ** 3 - u ** 2) * m1
    return M(slerp_R(keys[i][1][:3, :3], keys[i + 1][1][:3, :3], smooth(u)), pos)


def load_pocket(root):
    """Pistol transform inside the right front pocket (rest coordinates, sole lift included) and
    the pocket opening. build/pocket.json comes from src/build_clothes.py."""
    d = json.load(open(os.path.join(root, "build", "pocket.json")))
    opening = np.array(d["opening_centre"], float)
    depth = nrm(np.array(d["depth_dir"], float))
    out = np.array(d["outward_normal"], float)
    # muzzle down the bag, flat of the pistol against the thigh (its right side faces out), deep
    # enough that the grip sits just under the denim
    Y = depth
    X = nrm(out - Y * np.dot(out, Y))
    Z = np.cross(X, Y)
    origin = opening + depth * POCKET_DEPTH
    lift = np.array([0.0, 0.0, LIFT])
    return origin + lift, opening + lift, np.stack([X, Y, Z], 1)


def bone_M(W, H, b):
    return M(W[b], H[b])


GRIP = M(GRIP_ROT, GRIP_POS)
SUPPORT = M(SUPPORT_ROT, SUPPORT_POS)
MAG_HOLD = M(MAG_HOLD_ROT, MAG_HOLD_POS)
MAG_SEAT = M(MAG_LOCAL[1], MAG_LOCAL[0])


# ------------------------------------------------------------------ the plan
def _digits(p, side, w, fingers, thumb, index=None):
    """Pose one hand's digits, blending from a relaxed hand (w = 0) to the joint table (w = 1).
    fingers: {name: (mcp, pip, dip, spread)} in degrees; thumb: (rx, rz, (f1, f2, f3))."""
    sgn = 1.0 if side == "l" else -1.0
    for b in [b for b in p.B if b.split("_")[0] in ("thumb", "index", "middle", "ring", "pinky") and b.endswith(side)]:
        del p.B[b]                                  # replace the curl stand() put there
    table = dict(fingers)
    if index is not None:
        table["index"] = index
    relaxed = tuple(m * RELAXED for m in DIGIT_MAX) + (0.0,)
    for fn in ("index", "middle", "ring", "pinky"):
        a = [r + (x - r) * w for r, x in zip(relaxed, table.get(fn, relaxed))]
        for k in range(3):
            p.rot(f"{fn}_0{k + 1}_{side}", 0, -a[k])
        p.rot(f"{fn}_01_{side}", 2, a[3] * sgn)
    rx, rz, flex = thumb
    rest = tuple(m * 0.3 for m in THUMB_MAX)
    p.rot(f"thumb_01_{side}", 0, rx * w).rot(f"thumb_01_{side}", 2, rz * w)
    for k in range(3):
        p.rot(f"thumb_0{k + 1}_{side}", 0, -(rest[k] + (flex[k] - rest[k]) * w))


def ready(t):
    """0 relaxed .. 1 shooting stance (feet staggered, knees soft, shoulders over the toes)."""
    return smooth((t - 0.85) / 0.9) * (1 - smooth((t - 8.9) / 0.8))


def _body(skel, t, look_yaw=0.0, look_pitch=0.0):
    """Stance, trunk and head at time t, arms hanging."""
    r = ready(t)
    p = Pose(skel)
    drop = 0.012 + 0.034 * r
    breath = np.sin(2 * np.pi * t / 4.4)
    p.root((0.0, -0.015 * r, LIFT - drop), 0.0)
    p.pelvis(pitch=6.0 * r)
    p.spine(pitch=10.0 * r - 0.6 * breath + 1.0, yaw=look_yaw * 0.25)
    p.neck(pitch=-7.0 * r + look_pitch, yaw=look_yaw * 0.75)
    p.gaze(yaw=look_yaw * 0.25, pitch=look_pitch * 0.4)
    for s in ("l", "r"):
        p.clavicle(s, up=0.8 * breath - 2.0 * r, fwd=11.0 * r)
    ank = stand(skel, p, base=(0.0, -0.012 * r), spread=0.11 + 0.045 * r, stagger=0.03 + 0.055 * r, curl=0.35)
    hang_arms(p, (0.0, -0.015 * r), -drop)
    return p, drop, ank


def _gaze(t):
    """Head yaw / pitch (deg): on the target, down at the magwell, a scan, a look at the pocket."""
    yaw = interp_keys(t, [(0.0, 0.0), (T["low"], 0.0), (T["low"] + 0.20, 32.0), (T["low"] + 0.35, 32.0),
                          (T["low"] + 0.65, -30.0), (T["low"] + 0.75, -30.0), (T["off2"] + 0.2, 0.0), (SECONDS, 0.0)])
    pitch = interp_keys(t, [(0.0, 0.0), (T["off"], 0.0), (T["workspace"], 16.0), (T["seated"], 16.0),
                            (T["back"], 0.0), (T["pocket_up"] - 0.25, 0.0), (T["pocket_up"], 18.0),
                            (T["hand_out"], 18.0), (T["hang"] - 0.3, 0.0), (SECONDS, 0.0)])
    return yaw, pitch


def _aim_pose(skel):
    """Pistol pose with the sight line running from the right eye to the target, at full but not
    locked extension of the firing arm."""
    p, _, _ = _body(skel, 3.0)
    _, _, W, H = p.solve()
    eye = H["eye_r"]
    u = nrm(TARGET - eye)
    R = np.stack([nrm(np.cross(u, [0.0, 0.0, 1.0])), u, np.zeros(3)], 1)
    R[:, 2] = np.cross(R[:, 0], R[:, 1])
    sh = H["upperarm_r"]
    reach = 0.97 * (skel.length["upperarm_r"] + skel.length["lowerarm_r"])
    best = M(R, eye + u * 0.40 - R @ SIGHT_LOCAL)
    for d in np.arange(0.40, 0.80, 0.005):
        gun = M(R, eye + u * d - R @ SIGHT_LOCAL)
        if np.linalg.norm((gun @ inv(GRIP))[:3, 3] - sh) > reach:
            break
        best = gun
    return best


def _plan(skel, root_dir):
    """Kinematic frames for the routine; extra['gun_kin'] carries the planned pistol pose."""
    s = skel
    n = int(round(SECONDS * FPS))
    bag, opening, R_pocket = load_pocket(root_dir)
    W0, H0 = s.fk({}, np.array([0.0, 0.0, LIFT]))
    gun_in_pocket = inv(bone_M(W0, H0, "thigh_r")) @ M(R_pocket, bag)
    spare_bag = bag * np.array([-1.0, 1.0, 1.0])
    mag_in_pocket = inv(bone_M(W0, H0, "thigh_l")) @ M(axis_angle([1, 0, 0], 180.0) @ MAG_LOCAL[1], spare_bag)
    aim = _aim_pose(s)
    A, a = aim[:3, :3], aim[:3, 3]
    # retention at the hip, compressed ready at the chest, workspace under the chin, low ready
    retention = M(A, np.array([-0.16, -0.16, 1.06]))
    compressed = M(A @ axis_angle([1, 0, 0], 4.0), np.array([-0.04, -0.27, 1.30]))
    workspace = M(A @ axis_angle([0, 1, 0], -25.0) @ axis_angle([1, 0, 0], 6.0), np.array([-0.05, -0.38, 1.33]))
    low_ready = M(A @ axis_angle([1, 0, 0], -42.0), a + np.array([0.0, 0.16, -0.30]))
    out = []
    for i in range(n):
        t = i / FPS
        yaw, pitch = _gaze(t)
        p, drop, _ = _body(s, t, yaw, pitch)
        flinch = sum(max(0.0, 1 - abs(t - ts - 0.04) / 0.07) for ts in SHOTS)      # blink at every shot
        p.blink(min(1.0, blink_at(t, (0.35, 2.62, 4.15, 6.45, 9.3)) + flinch))
        _, _, W, H = p.solve()
        pocket = bone_M(W, H, "thigh_r") @ gun_in_pocket
        pocket_up = M(pocket[:3, :3], pocket[:3, 3] - pocket[:3, 1] * 0.17)
        gun = interp_pose(t, [
            (0.0, pocket), (T["grip"], pocket), (T["pulled"], pocket_up), (T["retention"], retention),
            (T["compressed"], compressed), (T["aim"], aim), (T["off"] - 0.03, aim), (T["workspace"], workspace),
            (T["back"] - 0.12, workspace), (T["aim2"], aim), (T["hold2"], aim), (T["low"], low_ready),
            (T["off2"], low_ready), (T["pocket_up"], pocket_up), (T["holstered"], pocket), (SECONDS, pocket)])
        in_hand = T["grip"] <= t <= T["holstered"]
        if not in_hand:
            gun = pocket
        # ---- right hand: into the pocket, onto the grip, then wherever the pistol goes
        hang_r = bone_M(W, H, "hand_r")
        grab = pocket @ inv(GRIP)
        above = M(grab[:3, :3], grab[:3, 3] - pocket[:3, 1] * 0.13)
        if t < T["reach"] or t > T["hang"]:
            hand_r = None
        elif t < T["pocket"]:
            hand_r = lerp_M(hang_r, above, smooth((t - T["reach"]) / (T["pocket"] - T["reach"])))
        elif t < T["grip"]:
            hand_r = lerp_M(above, grab, smooth((t - T["pocket"]) / 0.2))
        elif t <= T["holstered"]:
            hand_r = gun @ inv(GRIP)
        elif t < T["holstered"] + 0.14:
            hand_r = grab
        elif t < T["hand_out"]:
            hand_r = lerp_M(grab, above, smooth((t - T["holstered"] - 0.14) / (T["hand_out"] - T["holstered"] - 0.14)))
        else:
            hand_r = lerp_M(above, hang_r, smooth((t - T["hand_out"]) / (T["hang"] - T["hand_out"])))
        if hand_r is not None:
            p.arm("r", hand_r[:3, 3], H["upperarm_r"] + np.array([-0.30, 0.25, -0.40]), hand_rot=hand_r[:3, :3])
        # ---- left hand: support grip, spare magazine from the left pocket, support grip again
        hang_l = bone_M(W, H, "hand_l")
        sup = gun @ SUPPORT
        mag_pocket = bone_M(W, H, "thigh_l") @ mag_in_pocket
        mag_up = M(mag_pocket[:3, :3], mag_pocket[:3, 3] + np.array([0.0, -0.03, 0.15]))
        well = gun @ MAG_SEAT
        below = well @ M(np.eye(3), np.array([0.0, 0.0, -0.09]))
        lgrab = mag_pocket @ inv(MAG_HOLD)
        labove = M(lgrab[:3, :3], lgrab[:3, 3] + np.array([0.0, 0.0, 0.12]))
        slap = (well @ inv(MAG_HOLD)) @ M(np.eye(3), np.array([0.0, -0.02, 0.012]))   # heel of the palm on the baseplate
        mag2 = None
        if T["l_hold"] <= t <= T["seated"]:
            mag2 = interp_pose(t, [(T["l_hold"], mag_pocket), (T["l_out"], mag_up), (T["l_below"], below),
                                   (T["seated"], well)])
        if t < T["compressed"] - 0.55 or t > T["off2"] + 0.4:
            hand_l = None
        elif t < T["meet"]:
            u0 = T["compressed"] - 0.55
            hand_l = lerp_M(hang_l, sup, smooth((t - u0) / (T["meet"] - u0)))
        elif t < T["off"]:
            hand_l = sup
        elif t < T["l_pocket"]:
            hand_l = lerp_M(sup, labove, smooth((t - T["off"]) / (T["l_pocket"] - T["off"])))
        elif t < T["l_grab"]:
            hand_l = lerp_M(labove, lgrab, smooth((t - T["l_pocket"]) / (T["l_grab"] - T["l_pocket"])))
        elif t < T["l_hold"]:
            hand_l = lgrab
        elif t <= T["seated"]:
            hand_l = mag2 @ inv(MAG_HOLD)
        elif t < T["slap"]:
            hand_l = lerp_M(well @ inv(MAG_HOLD), slap, smooth((t - T["seated"]) / (T["slap"] - T["seated"])))
        elif t < T["back"]:
            hand_l = lerp_M(slap, sup, smooth((t - T["slap"]) / (T["back"] - T["slap"])))
        elif t < T["off2"]:
            hand_l = sup
        else:
            hand_l = lerp_M(sup, hang_l, smooth((t - T["off2"]) / 0.4))
        if hand_l is not None:
            p.arm("l", hand_l[:3, 3], H["upperarm_l"] + np.array([0.30, 0.25, -0.40]), hand_rot=hand_l[:3, :3])
        # ---- fingers
        trig = smooth((t - T["trigger_on"]) / 0.12) * (1 - smooth((t - T["trigger_off"]) / 0.10)) + \
            smooth((t - T["trigger2_on"]) / 0.12) * (1 - smooth((t - T["trigger2_off"]) / 0.10))
        press = sum(max(0.0, 1 - abs(t - ts) / 0.05) for ts in SHOTS)
        closing = smooth((t - T["grip"] + 0.14) / 0.14) * (1 - smooth((t - T["holstered"]) / 0.14))
        index = tuple(a + (b - a) * trig for a, b in zip(INDEX_OFF, INDEX_ON))
        index = (index[0], index[1] + 8.0 * press, index[2] + 8.0 * press, index[3])
        release = max(0.0, 1 - abs(t - MAG_RELEASE) / 0.08)          # thumb presses the magazine catch
        rx, rz, flex = GRIP_THUMB
        _digits(p, "r", closing, GRIP_FINGERS, (rx, rz, (flex[0], flex[1] + 25.0 * release, flex[2] + 20.0 * release)),
                index=index)
        support = smooth((t - T["meet"] + 0.12) / 0.12) * (1 - smooth((t - T["off"]) / 0.10)) + \
            smooth((t - T["back"] + 0.12) / 0.12) * (1 - smooth((t - T["off2"]) / 0.10))
        on_mag = smooth((t - T["l_grab"] + 0.06) / 0.06) * (1 - smooth((t - T["seated"]) / 0.06))
        rx, rz, flex = SUPPORT_THUMB
        stop = max(0.0, 1 - abs(t - SLIDE_RELEASE) / 0.08)               # support thumb rides the slide stop down
        if on_mag > 0.0:
            _digits(p, "l", on_mag, MAG_FINGERS, (0.0, 0.0, tuple(m * 0.35 for m in THUMB_MAX)), index=MAG_INDEX)
        else:
            _digits(p, "l", support, SUPPORT_FINGERS, (rx, rz, (flex[0], flex[1] + 30.0 * stop, flex[2] + 25.0 * stop)))
        f = solve(p, {"cam": np.array([0.0, -0.15, 1.24]), "gun_kin": gun, "mag_pocket": mag_pocket})
        out.append(f)
    return out


# ------------------------------------------------------------------ physics pass
def pistol_sequence(skel, root_dir):
    plan = _plan(skel, root_dir)
    rel = {}

    def gun_ref_pose(ref):
        g = ref.extra["gun_kin"]
        return ph.tup(g[:3, 3] + g[:3, :3] @ GUN_COM), ph.quat(g[:3, :3])

    def setup(sim):
        sim.add_prop("pistol", GUN_MASS, GUN_INERTIA, gun_ref_pose)
        hand_r, hand_l = sim.bodies["hand_r"], sim.bodies["hand_l"]
        # weld offsets are between body centres of mass, in each body's own frame
        rel["grip"] = (ph.tup(GRIP_POS + GRIP_ROT @ GUN_COM - np.array(hand_r.com)), ph.quat(GRIP_ROT))
        rel["hand_l"] = lambda Hg: (ph.tup(Hg[:3, 3] + Hg[:3, :3] @ np.array(hand_l.com) - GUN_COM), ph.quat(Hg[:3, :3]))
        sim.set_tone(("lumbar", "thoracic"), 6.0, 0.9, 1.0)
        sim.set_tone(("neck", "skull"), 5.0, 0.8, 1.0)
        sim.set_tone(("shoulder", "elbow"), 5.0, 1.0, 1.0)
        # wrists braced but not rigid: they give a little under recoil
        sim.set_tone(("wrist",), k=22.0, c=0.6, comp=1.0)
        # hands go into the pockets and onto each other
        for a, b in (("hand_r", "thigh_r"), ("hand_r", "pelvis"), ("hand_l", "thigh_l"), ("hand_l", "pelvis"),
                     ("hand_l", "hand_r"), ("hand_l", "lowerarm_r"), ("hand_r", "lowerarm_l"),
                     ("lowerarm_l", "lowerarm_r"), ("hand_l", "abdomen"), ("hand_r", "abdomen")):
            sim.nocollide.add((a, b))

    f = lambda t: int(round(t * FPS))
    shots = {f(ts) for ts in SHOTS}
    two_hands = [(f(T["meet"]), f(T["off"])), (f(T["back"]), f(T["off2"]))]

    def soft_weld(sim, key, i, start, Hg):
        """Left hand onto the pistol at Hg (hand pose in pistol space), closing over 4 frames."""
        sim.weld(key, sim.bodies["pistol"], sim.bodies["hand_l"], alpha=2e-7 * 10.0 ** max(0, start + 3 - i),
                 rel=rel["hand_l"](Hg))

    def per_frame(sim, i):
        gun, hand_r = sim.bodies["pistol"], sim.bodies["hand_r"]
        t = i / FPS
        if i == f(T["grip"]):                              # fingers closed on the grip
            gun.kin = False
            sim.weld("grip", hand_r, gun, rel=rel["grip"])
            hand_r.props = [gun]
        elif i == f(T["holstered"]):                       # let go in the pocket
            sim.welds.pop("grip", None)
            gun.kin = True
            hand_r.props = []
        for a, b in two_hands:
            if a <= i < b:
                soft_weld(sim, "support", i, a, SUPPORT)
            if i == a:                                     # both arms braced against each other
                sim.set_tone(("shoulder", "elbow"), 6.5, 1.1, 1.0)
            if i == b:
                sim.welds.pop("support", None)
                sim.set_tone(("shoulder", "elbow"), 5.0, 1.0, 1.0)
        if i == f(T["off"]):                               # the reload: quick, deliberate left arm
            sim.set_tone(("shoulder_l", "elbow_l"), 8.0, 1.0, 1.0)
        # the spare meets the magwell: from then the hand travels with the pistol, along the well
        a = f(T["l_below"] - 0.12)
        if a <= i < f(T["slap"]):
            u = smooth((t - T["l_below"]) / (T["seated"] - T["l_below"]))
            push = smooth((t - T["seated"]) / (T["slap"] - T["seated"]))
            Hg = MAG_SEAT @ M(np.eye(3), np.array([0.0, 0.0, -0.09 * (1 - u)])) @ inv(MAG_HOLD) @ \
                M(np.eye(3), np.array([0.0, -0.02, 0.012]) * push)
            soft_weld(sim, "magwell", i, a, Hg)
        elif i == f(T["slap"]):
            sim.welds.pop("magwell", None)
        if i in shots:
            sim.impulse(gun, gun.point(ph.tup(BORE_LOCAL - GUN_COM)), ph.qrot(gun.q, (0.0, -RECOIL, 0.0)))

    frames, _ = ph.track(skel, plan, setup, per_frame)
    for i, fr in enumerate(frames):
        x, q = fr.extra.pop("props")["pistol"]
        if T["grip"] <= i / FPS <= T["holstered"]:
            # drawn exactly in the hand the viewer sees (the simulated pistol is welded to it)
            fr.extra["gun"] = bone_M(fr.W, fr.H, "hand_r") @ GRIP
        else:
            R = ph.qmat(q)
            fr.extra["gun"] = M(R, np.array(x) - R @ GUN_COM)
    _props(frames)
    return frames


def _props(frames):
    """Slide, trigger, flash, buttons, both magazines and the casings, from the simulated pistol
    and hands."""
    dt = 1.0 / FPS
    mag_pos = mag_vel = mag_R = None
    mag_rest = False
    casings = [dict(on=False) for _ in SHOTS]
    rng = np.random.default_rng(7)
    for i, fr in enumerate(frames):
        t = i * dt
        gun = fr.extra["gun"]
        ex = fr.extra
        # slide: cycles in 3 frames on each shot, locks back after the last round until released
        slide = 0.0
        for ts in SHOTS:
            if ts <= t < ts + 3 * dt:
                slide = -0.028 * (1 - (t - ts) / (3 * dt)) if t > ts + dt else -0.028
        if SHOTS[-1] + dt <= t < SLIDE_RELEASE:
            slide = -0.028
        ex["slide"] = slide
        ex["trigger"] = -12.0 * sum(max(0.0, 1 - abs(t - ts) / 0.05) for ts in SHOTS)
        ex["flash"] = any(0 <= t - ts < dt for ts in SHOTS)
        ex["mag_release"] = 0.0012 if abs(t - MAG_RELEASE) <= 0.06 else 0.0
        ex["slide_stop"] = -6.0 if SHOTS[-1] < t < SLIDE_RELEASE else 0.0
        # spent magazine: seated until released, then it drops free and falls
        seated = gun @ MAG_SEAT
        if t < MAG_RELEASE:
            ex["mag1"] = seated
            prev_seated = seated
        else:
            if mag_pos is None:
                mag_R = seated[:3, :3]
                mag_pos = seated[:3, 3].copy()
                mag_vel = (seated[:3, 3] - prev_seated[:3, 3]) / dt - mag_R[:, 2] * 0.6
            if not mag_rest:
                mag_vel = mag_vel + G * dt
                mag_pos = mag_pos + mag_vel * dt
                mag_R = axis_angle(mag_R[:, 0], 160.0 * dt) @ mag_R
                if mag_pos[2] < 0.014:
                    mag_pos[2] = 0.014
                    # topple onto its side: long axis flat along the ground, side panel up
                    zh = np.array([mag_R[0, 2], mag_R[1, 2], 0.0])
                    zh = nrm(zh) if np.linalg.norm(zh) > 1e-3 else np.array([1.0, 0.0, 0.0])
                    upv = np.array([0.0, 0.0, 1.0])
                    mag_R = np.stack([upv, np.cross(zh, upv), zh], 1)
                    mag_rest = True
            ex["mag1"] = M(mag_R, mag_pos)
        # spare magazine: left pocket, then the left hand, then seated in the pistol
        held = bone_M(fr.W, fr.H, "hand_l") @ MAG_HOLD
        if t < T["l_grab"]:
            ex["mag2"] = ex["mag_pocket"]
        elif t < T["l_hold"] + 0.02:
            ex["mag2"] = lerp_M(ex["mag_pocket"], held, smooth((t - T["l_grab"]) / (T["l_hold"] + 0.02 - T["l_grab"])))
        elif t < MAG_SEATED:                     # the hand carries it up the magwell
            ex["mag2"] = held
        else:
            ex["mag2"] = seated
        for k in ("gun_kin", "mag_pocket"):
            ex.pop(k, None)
        # casings leave the ejection port up and to the right, bounce and settle
        cs = []
        for k, ts in enumerate(SHOTS):
            c = casings[k]
            if t < ts + dt:
                cs.append(None)
                continue
            if not c["on"]:
                gR, gp = gun[:3, :3], gun[:3, 3]
                c.update(on=True, p=gp + gR @ PORT_LOCAL, R=gR.copy(),
                         v=gR @ np.array([3.0, -0.6, 2.2]) + rng.normal(0, 0.25, 3), spin=900.0, rest=False)
            if not c["rest"]:
                c["v"] = c["v"] + G * dt
                c["p"] = c["p"] + c["v"] * dt
                c["R"] = axis_angle(c["R"][:, 0], c["spin"] * dt) @ c["R"]
                if c["p"][2] < 0.005:
                    c["p"][2] = 0.005
                    c["v"] = np.array([c["v"][0] * 0.35, c["v"][1] * 0.35, -c["v"][2] * 0.25])
                    c["spin"] *= 0.4
                    if np.linalg.norm(c["v"]) < 0.25:
                        c["rest"] = True
                        y = nrm(np.array([c["R"][0, 1], c["R"][1, 1], 0.0]) + 1e-9)
                        c["R"] = np.stack([np.cross(y, [0, 0, 1.0]), y, np.array([0, 0, 1.0])], 1)
            cs.append(M(c["R"], c["p"]))
        ex["casings"] = cs
