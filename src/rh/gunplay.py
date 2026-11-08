"""Pistol handling: draw from the right front jeans pocket, aim, fire, reload, return to the pocket.

All props are posed from the skeleton every frame:
  - in the pocket the gun rides the right thigh bone (the pocket moves with the leg);
  - in hand it is fixed to the right hand bone through GRIP (palm wrapped around the grip,
    thumb on the left side, barrel along the metacarpals);
  - the dropped magazine and ejected casings are ballistic bodies with floor contact.
Times are seconds; SEGMENTS gives the sub-clip ranges (draw / shoot / reload / holster).
Pistol part conventions (blender/pistol.py): +Y muzzle, +Z sights, +X gun's right side; root =
grip centre; slide travel -0.028 m along Y; trigger rotates -12 deg about X; magazine slides out
along its own -Z from (0, 0.0141, 0.0325) tilted -18 deg about X.
"""
import json
import os

import numpy as np

from .motions import FPS, LIFT, Frame, hang_arms, interp_keys, smooth, solve, stand
from .pose import Pose, axis_angle, nrm

GRIP_POS = np.array([0.0, 0.050, -0.030])          # gun root in the right hand bone frame
# hand bone axes (right hand): x ulnar, y distal, z dorsal. Gun axes in that frame:
GRIP_ROT = np.stack([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], axis=1) @ axis_angle([1, 0, 0], 12.0)
MAG_LOCAL = (np.array([0.0, 0.0141, 0.0325]), axis_angle([1, 0, 0], -18.0))
PORT_LOCAL = np.array([0.008, 0.036, 0.046])
MUZZLE_LOCAL = np.array([0.0, 0.1121, 0.0395])
SEGMENTS = {"draw": (0.6, 2.6), "shoot": (2.4, 4.2), "reload": (3.9, 6.9), "holster": (6.7, 9.6)}
SHOTS = (2.80, 3.25, 3.70)
G = np.array([0.0, 0.0, -9.81])


def M(R, p):
    m = np.eye(4)
    m[:3, :3] = R
    m[:3, 3] = p
    return m


def inv(m):
    r = m[:3, :3].T
    return M(r, -r @ m[:3, 3])


def quat(R):
    from .pose import quat_from_matrix
    return quat_from_matrix(R)


def slerp_R(Ra, Rb, t):
    qa, qb = quat(Ra), quat(Rb)
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


def interp_pose(t, keys):
    """keys: [(time, 4x4)] -> 4x4 with cubic position and slerped rotation."""
    ts = [k[0] for k in keys]
    pos = interp_keys(t, [(k[0], k[1][:3, 3]) for k in keys])
    if t <= ts[0]:
        R = keys[0][1][:3, :3]
    elif t >= ts[-1]:
        R = keys[-1][1][:3, :3]
    else:
        i = int(np.searchsorted(ts, t) - 1)
        u = smooth((t - ts[i]) / (ts[i + 1] - ts[i]))
        R = slerp_R(keys[i][1][:3, :3], keys[i + 1][1][:3, :3], u)
    return M(R, pos)


GUN_REAR = 0.073          # pistol root origin to the back of the slide along -Y (muzzle is +Y)


def load_pocket(root):
    """Gun transform inside the right front pocket (rest coordinates, sole lift included) and the
    pocket opening. build/pocket.json comes from src/build_clothes.py."""
    d = json.load(open(os.path.join(root, "build", "pocket.json")))
    opening = np.array(d["opening_centre"], float)
    depth = nrm(np.array(d["depth_dir"], float))
    out = np.array(d["outward_normal"], float)
    # muzzle down the bag, flat of the gun against the thigh (its right side faces out),
    # the back of the slide 15 mm below the pocket mouth so only the grip is in reach
    Y = depth
    X = nrm(out - Y * np.dot(out, Y))
    Z = np.cross(X, Y)
    origin = opening + depth * (GUN_REAR + 0.015)
    lift = np.array([0.0, 0.0, LIFT])
    return origin + lift, opening + lift, np.stack([X, Y, Z], 1)


def bone_M(W, H, b):
    return M(W[b], H[b])


def pistol_sequence(skel, root_dir, seconds=9.8):
    s = skel
    n = int(seconds * FPS)
    bag, opening, R_pocket = load_pocket(root_dir)
    # rest-pose thigh frame: the gun's pocket transform relative to the right thigh bone
    W0, H0 = s.fk({}, np.array([0.0, 0.0, LIFT]))
    gun_in_pocket = inv(bone_M(W0, H0, "thigh_r")) @ M(R_pocket, bag)
    spare_bag = bag * np.array([-1.0, 1.0, 1.0])
    mag_in_pocket_l = inv(bone_M(W0, H0, "thigh_l")) @ M(axis_angle([1, 0, 0], 180.0) @ MAG_LOCAL[1], spare_bag)
    grip = M(GRIP_ROT, GRIP_POS)
    aim_dir = nrm(np.array([0.0, -1.0, 0.02]))

    def gun_aim_M(wrist_offset=(0.0, 0.0, 0.0), pitch=0.0, cant=0.0, at=None):
        """Gun pose when aiming at the target: barrel along aim_dir, sights up."""
        Y = aim_dir
        Z = np.array([0.0, 0.0, 1.0])
        X = nrm(np.cross(Y, Z))
        Z = np.cross(X, Y)
        R = np.stack([X, Y, Z], 1) @ axis_angle([1, 0, 0], pitch) @ axis_angle([0, 1, 0], cant)
        p = (np.array([-0.025, -0.555, 1.505]) if at is None else at) + np.asarray(wrist_offset)
        return M(R, p)

    # ---------------- right-hand / gun key poses (world 4x4 of the GUN)
    def pocket_world(t_frame_W, t_frame_H):
        return bone_M(t_frame_W, t_frame_H, "thigh_r") @ gun_in_pocket

    out = []
    held = []                     # per frame: True if the gun is in hand
    gun_world = []
    mag_state = []
    events = {"shots": SHOTS, "mag_release": 4.22, "mag_insert": (5.18, 5.32), "slide_release": 5.55}
    recoil = lambda t: sum(np.exp(-((t - ts) / 0.035) ** 2 * 0.5) * (t >= ts - 0.02) for ts in SHOTS)
    workspace = gun_aim_M(at=np.array([-0.035, -0.30, 1.30]), pitch=8.0, cant=-28.0)
    low_ready = gun_aim_M(at=np.array([-0.04, -0.30, 1.16]), pitch=-38.0)
    for i in range(n):
        t = i / FPS
        p = Pose(s)
        aim_w = smooth((t - 1.9) / 0.5) * (1 - smooth((t - 6.95) / 0.5))
        drop = 0.012 + 0.025 * aim_w
        p.root((0.0, 0.0, LIFT - drop), 0.0)
        p.pelvis(pitch=2.0 * aim_w)
        p.spine(pitch=8.0 * aim_w + 2.0 * recoil(t) * -1, roll=-3.0 * smooth((t - 0.9) / 0.4) * (1 - smooth((t - 1.8) / 0.3)))
        p.neck(pitch=-4.0 * aim_w + 12.0 * smooth((t - 0.9) / 0.3) * (1 - smooth((t - 1.6) / 0.3))
               + 18.0 * smooth((t - 3.95) / 0.3) * (1 - smooth((t - 5.6) / 0.3)), yaw=-6.0 * aim_w)
        stand(s, p, base=(0.0, 0.0), spread=0.13 * aim_w + 0.11 * (1 - aim_w), stagger=0.06 * aim_w, curl=0.35)
        hang_arms(p, (0.0, 0.0), -drop)
        p.gaze(pitch=-8.0 * smooth((t - 0.9) / 0.3) * (1 - smooth((t - 1.6) / 0.3)))
        # blink now and then
        for tb in (0.4, 3.0, 6.4, 8.9):
            p.blink(max(0.0, 1 - abs(t - tb) / 0.07))
        q0, loc0, W, H = p.solve()
        in_pocket_gun = pocket_world(W, H)
        # key poses of the gun through the clip
        keys = [
            (0.0, in_pocket_gun), (1.30, in_pocket_gun), (1.42, in_pocket_gun),
            (1.72, M(in_pocket_gun[:3, :3], in_pocket_gun[:3, 3] + [0.0, -0.02, 0.17])),
            (2.05, workspace), (2.42, gun_aim_M()),
            (3.95, gun_aim_M()), (4.15, workspace), (5.60, workspace), (5.95, gun_aim_M()),
            (6.85, gun_aim_M()), (7.25, low_ready),
            (7.55, M(in_pocket_gun[:3, :3], in_pocket_gun[:3, 3] + [0.0, -0.02, 0.17])),
            (7.80, in_pocket_gun), (seconds, in_pocket_gun),
        ]
        gun = interp_pose(t, keys)
        # recoil: muzzle flip and push-back, recovering in ~0.15 s
        r = recoil(t)
        if r > 1e-3:
            gun = gun @ M(axis_angle([1, 0, 0], 13.0 * r), np.array([0.0, -0.025 * r, 0.012 * r]))
        in_hand = 1.30 <= t <= 7.85
        held.append(in_hand)
        gun_world.append(gun if in_hand else in_pocket_gun)
        # right hand follows the gun through the inverse grip
        hand_r = gun @ inv(grip)
        if in_hand or 1.0 < t < 8.3:
            reach = smooth((t - 0.75) / 0.55) * (1 - smooth((t - 7.85) / 0.45))
            hang = bone_M(W, H, "hand_r")
            if in_hand:
                target = hand_r
            else:
                target = M(slerp_R(hang[:3, :3], hand_r[:3, :3], reach), hang[:3, 3] + (hand_r[:3, 3] - hang[:3, 3]) * reach)
            p.arm("r", target[:3, 3], H["upperarm_r"] + np.array([-0.35, 0.25, -0.35]), hand_rot=target[:3, :3])
        # right fingers: grip, index on the trigger only while shooting
        trig = aim_w * (smooth((t - 2.6) / 0.15) * (1 - smooth((t - 3.9) / 0.15)))
        pull = sum(max(0.0, 1 - abs(t - ts) / 0.06) for ts in SHOTS)
        grip_c = 0.85 if in_hand else 0.35 + 0.3 * smooth((t - 1.2) / 0.2)
        p.fingers("r", curl=grip_c, thumb=0.15 if in_hand else 0.3,
                  per={"index": (0.25 + 0.30 * trig + 0.12 * pull) if in_hand else grip_c, "thumb": 0.15 if in_hand else 0.3})
        # ---------------- left hand: support grip, reload
        support = smooth((t - 2.15) / 0.3) * (1 - smooth((t - 4.05) / 0.12)) + smooth((t - 5.70) / 0.25) * (1 - smooth((t - 6.95) / 0.3))
        gun_R, gun_p = gun[:3, :3], gun[:3, 3]
        X, Y, Z = gun_R[:, 0], gun_R[:, 1], gun_R[:, 2]
        # support hand: thumb (radial x) along the frame, fingers wrapping under the trigger guard
        wrap = nrm(X - 0.7 * Z)
        sup_R = np.stack([nrm(Y - wrap * np.dot(Y, wrap)), wrap, np.zeros(3)], 1)
        sup_R[:, 2] = np.cross(sup_R[:, 0], sup_R[:, 1])
        sup_wrist = gun_p - 0.040 * X - 0.045 * Z + 0.012 * Y - sup_R[:, 1] * 0.050
        # reload: left hand to the left pocket, grab the spare magazine, bring it to the magwell
        mag_pocket_world = bone_M(W, H, "thigh_l") @ mag_in_pocket_l
        mag_hold = M(np.stack([np.array([0, 0, 1.0]), np.array([0, 1.0, 0]), np.array([-1.0, 0, 0])], 1), np.array([0.0, 0.065, -0.022]))
        well = gun @ M(MAG_LOCAL[1], MAG_LOCAL[0])
        below_well = well @ M(np.eye(3), np.array([0.0, 0.0, -0.095]))
        mag_keys = [(4.30, mag_pocket_world), (4.62, mag_pocket_world),
                    (4.90, M(mag_pocket_world[:3, :3], mag_pocket_world[:3, 3] + [0.0, -0.05, 0.20])),
                    (5.18, below_well), (5.32, well)]
        mag2 = interp_pose(t, mag_keys) if 4.30 <= t <= 5.32 else (mag_pocket_world if t < 4.30 else well)
        left_on_mag = smooth((t - 4.30) / 0.30) * (1 - smooth((t - 5.36) / 0.10))
        hand_l_on_mag = mag2 @ inv(mag_hold) @ M(np.eye(3), np.zeros(3))
        hang_l = bone_M(W, H, "hand_l")
        if left_on_mag > 0 or support > 0:
            tgt_p = hang_l[:3, 3] * (1 - max(left_on_mag, support)) + sup_wrist * support + hand_l_on_mag[:3, 3] * left_on_mag
            tgt_R = slerp_R(hang_l[:3, :3], sup_R, support) if left_on_mag <= 0 else slerp_R(hang_l[:3, :3], hand_l_on_mag[:3, :3], left_on_mag)
            # slap the base home at 5.35
            if 5.32 <= t <= 5.45:
                tgt_p = tgt_p + Z * 0.0
            p.arm("l", tgt_p, H["upperarm_l"] + np.array([0.35, 0.25, -0.35]), hand_rot=tgt_R)
        p.fingers("l", curl=0.35 + 0.45 * max(support, left_on_mag), thumb=0.3)
        f = solve(p, {"pelvis": np.array([0.0, 0.0, 1.0]), "cam": np.array([0.0, -0.15, 1.24])})
        # the gun follows the solved hand exactly while held (IK may clamp reach)
        if in_hand:
            gun = bone_M(f.W, f.H, "hand_r") @ grip
            gun_world[-1] = gun
        f.extra["gun"] = gun_world[-1]
        f.extra["mag2"] = mag2 if t <= 5.32 else (gun_world[-1] @ M(MAG_LOCAL[1], MAG_LOCAL[0]))
        out.append(f)
    _props(out, n)
    return out


def _props(frames, n):
    """Slide, trigger, flash, magazine drop and casing trajectories."""
    t_rel = 4.22
    mag_pos, mag_vel, mag_R, mag_rest = None, None, None, False
    casings = [dict(on=False) for _ in SHOTS]
    for i, f in enumerate(frames):
        t = i / FPS
        gun = f.extra["gun"]
        # slide: cycles on each shot, locks back after the last round, released at 5.55
        slide = 0.0
        for ts in SHOTS:
            if ts <= t < ts + 3 / FPS:
                slide = -0.028 * (1 - (t - ts) * FPS / 3) if t > ts + 1 / FPS else -0.028
        if SHOTS[-1] + 1 / FPS <= t < 5.55:
            slide = -0.028
        f.extra["slide"] = slide
        f.extra["trigger"] = -12.0 * sum(max(0.0, 1 - abs(t - ts) / 0.05) for ts in SHOTS)
        f.extra["flash"] = any(0 <= t - ts < 1 / FPS for ts in SHOTS)
        f.extra["mag_release"] = 0.0012 if 4.17 <= t <= 4.30 else 0.0
        f.extra["slide_stop"] = -6.0 if SHOTS[-1] < t < 5.55 else 0.0
        # spent magazine: rides the gun until released, then slides out and falls
        seated = gun @ M(MAG_LOCAL[1], MAG_LOCAL[0])
        if t < t_rel:
            f.extra["mag1"] = seated
            prev_seated = seated
        else:
            dt = 1 / FPS
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
                    mag_vel = np.zeros(3)
                    # topple onto its side: long axis flat along the ground, side panel facing up
                    zh = np.array([mag_R[0, 2], mag_R[1, 2], 0.0])
                    zh = nrm(zh) if np.linalg.norm(zh) > 1e-3 else np.array([1.0, 0.0, 0.0])
                    up = np.array([0.0, 0.0, 1.0])
                    mag_R = np.stack([up, np.cross(zh, up), zh], 1)
                    mag_rest = True
            f.extra["mag1"] = M(mag_R, mag_pos)
        # casings
        cs = []
        for k, ts in enumerate(SHOTS):
            c = casings[k]
            if t < ts + 1 / FPS:
                cs.append(None)
                continue
            dt = 1 / FPS
            if not c["on"]:
                gR, gp = gun[:3, :3], gun[:3, 3]
                c.update(on=True, p=gp + gR @ PORT_LOCAL, R=gR.copy(),
                         v=gR @ np.array([3.0, -0.6, 2.2]) + np.random.default_rng(k).normal(0, 0.25, 3), spin=900.0, rest=False)
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
        f.extra["casings"] = cs
