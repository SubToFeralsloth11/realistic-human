"""Procedural, biomechanics-driven motions. Each clip returns a list of solved frames.

Coordinates: world metres, +Z up, the character starts at the origin facing -Y.
LIFT raises the whole body so the sneaker outsoles (12 mm under the bare soles) touch z = 0.
"""
import numpy as np

from .pose import Pose, axis_angle, nrm

FPS = 30
LIFT = 0.012
ANKLE_H = 0.082          # rest ankle height above the bare sole
HEEL = np.array([0.0, 0.055, -ANKLE_H])     # heel contact relative to the ankle (foot flat, heading frame)
BALL = np.array([0.0, -0.135, -ANKLE_H])    # ball contact


def smooth(x):
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3 - 2 * x)


def interp_keys(t, keys):
    """Piecewise cubic (Catmull-Rom) through (time, value) keys; values may be vectors."""
    ts = np.array([k[0] for k in keys], float)
    vs = np.array([np.atleast_1d(k[1]) for k in keys], float)
    if t <= ts[0]:
        return vs[0] if vs.shape[1] > 1 else vs[0, 0]
    if t >= ts[-1]:
        return vs[-1] if vs.shape[1] > 1 else vs[-1, 0]
    i = int(np.searchsorted(ts, t) - 1)
    p1, p2 = vs[i], vs[i + 1]
    p0 = vs[i - 1] if i > 0 else 2 * p1 - p2
    p3 = vs[i + 2] if i + 2 < len(vs) else 2 * p2 - p1
    t0, t1 = ts[i], ts[i + 1]
    u = (t - t0) / (t1 - t0)
    m1 = (p2 - p0) * 0.5
    m2 = (p3 - p1) * 0.5
    h = (2 * u ** 3 - 3 * u ** 2 + 1) * p1 + (u ** 3 - 2 * u ** 2 + u) * m1 + (-2 * u ** 3 + 3 * u ** 2) * p2 + (u ** 3 - u ** 2) * m2
    return h if vs.shape[1] > 1 else h[0]


def pitch_rot(toes_up_deg, heading_deg=0.0):
    """World rotation tilting a foot about its lateral axis (toes up positive) under a heading."""
    return axis_angle([0, 0, 1], heading_deg) @ axis_angle([1, 0, 0], -toes_up_deg)


class Frame:
    def __init__(self, quats, root_loc, W, H, extra=None):
        self.quats, self.root_loc, self.W, self.H = quats, root_loc, W, H
        self.extra = extra or {}


def solve(p, extra=None):
    q, loc, W, H = p.solve()
    return Frame(q, loc, W, H, extra)


# ------------------------------------------------------------------ walk / run
def stance_ankle(flat_ankle, theta, heading=0.0):
    """Ankle position for a planted foot pitched by theta (deg, toes up +) about its heel or ball."""
    Rh = axis_angle([0, 0, 1], heading)
    local_piv = HEEL if theta >= 0 else BALL
    piv = flat_ankle + Rh @ local_piv
    Rp = Rh @ axis_angle([1, 0, 0], -theta) @ Rh.T
    return piv + Rp @ (flat_ankle - piv)


def gait_leg(phase, k, stride, x, y0, params):
    """Ankle target, foot pitch and toe bend for one leg at gait phase in [0, 1) of cycle k."""
    st = params["stance"]
    hs, ff, ho = params["heel_strike"], params["foot_flat"], params["heel_off"]
    to = params["toe_off"]
    z0 = ANKLE_H + LIFT
    flat = lambda kk: np.array([x, y0 - kk * stride, z0])
    if phase < st:
        f = phase / st
        if f < ff:
            theta = hs * (1 - smooth(f / ff))
        elif f < ho:
            theta = 0.0
        else:
            theta = to * smooth((f - ho) / (1 - ho))
        ankle = stance_ankle(flat(k), theta)
        toe = max(-theta, 0.0)
        return ankle, theta, toe
    s = (phase - st) / (1 - st)
    a0 = stance_ankle(flat(k), to)
    a1 = stance_ankle(flat(k + 1), hs)
    yprog = 0.5 - 0.5 * np.cos(np.pi * s)
    yprog = 0.75 * yprog + 0.25 * s
    zk = params["swing_z"]
    lift = interp_keys(s, [(0.0, a0[2]), (0.3, zk[0]), (0.6, zk[1]), (0.85, zk[2]), (1.0, a1[2])])
    ankle = np.array([x, a0[1] + (a1[1] - a0[1]) * yprog, lift])
    theta = interp_keys(s, [(0.0, to), (0.35, params["swing_pitch"][0]), (0.7, params["swing_pitch"][1]), (1.0, hs)])
    return ankle, theta, 0.0


WALK = dict(cycle=1.06, speed=1.20, stance=0.60, contact=0.30, heel_strike=8.0, foot_flat=0.17, heel_off=0.62,
            toe_off=-26.0, swing_z=(0.235, 0.17, 0.115), swing_pitch=(-6.0, 0.0), width=0.078, bob=0.018,
            bob_phase=0.0, sway=0.020, pel_yaw=4.0, pel_roll=4.0, lean=3.0, thorax_yaw=6.0, arm_swing=0.13,
            arm_z=0.86, arm_x=0.215, arm_lift=0.03, crouch=0.022, head_pitch=2.0, curl=0.30, clav_up=-3.0, run=False)

RUN = dict(cycle=0.72, speed=3.0, stance=0.32, contact=0.24, heel_strike=4.0, foot_flat=0.25, heel_off=0.40,
           toe_off=-32.0, swing_z=(0.52, 0.40, 0.17), swing_pitch=(-20.0, 4.0), width=0.055, bob=0.030,
           bob_phase=0.16, sway=0.012, pel_yaw=8.0, pel_roll=5.0, lean=10.0, thorax_yaw=12.0, arm_swing=0.19,
           arm_z=1.00, arm_x=0.20, arm_lift=0.12, crouch=0.050, head_pitch=5.0, curl=0.60, clav_up=0.0, run=True)


def _hip_drop(skel, H, ankles, margin=0.985):
    """How far the hips must drop so both ankles are within margin x leg length."""
    drop = 0.0
    for side, ank in ankles.items():
        L = (skel.length[f"thigh_{side}"] + skel.length[f"calf_{side}"]) * margin
        hip = H[f"thigh_{side}"]
        horiz = np.linalg.norm((ank - hip)[:2])
        if horiz >= L:
            horiz = L * 0.999
        need = (hip[2] - ank[2]) - np.sqrt(L * L - horiz * horiz)
        drop = max(drop, need)
    return drop


def locomotion(skel, seconds, params, start_y=0.0):
    """Cyclic gait (walk or run). Returns frames; extra['pelvis'] holds the hip centre for the camera."""
    T, v = params["cycle"], params["speed"]
    stride = v * T
    n = int(round(seconds * FPS))
    controls = []
    for i in range(n):
        t = i / FPS
        ph_l = (t / T) % 1.0
        k_l = int(t // T)
        ph_r = (t / T + 0.5) % 1.0
        k_r = int((t / T + 0.5) // 1.0)
        bob = -params["bob"] * np.cos(4 * np.pi * (ph_l - params["bob_phase"]))
        if params["run"]:
            bob = -bob                     # running: lowest at mid-stance, highest in flight
        sway = params["sway"] * np.sin(2 * np.pi * ph_l)
        y = start_y - v * t + 0.006 * np.sin(4 * np.pi * ph_l)
        legs = {}
        for side, ph, k, sg in (("l", ph_l, k_l, 1.0), ("r", ph_r, k_r, -1.0)):
            y0 = start_y - params["contact"] + (0.0 if side == "l" else stride * 0.5)
            legs[side] = (ph,) + gait_leg(ph, k, stride, sg * params["width"], y0, params)
        controls.append((t, ph_l, bob, sway, y, legs))

    def build(c, drop):
        t, ph_l, bob, sway, y, legs = c
        p = Pose(skel)
        z = LIFT - params["crouch"] + bob - drop
        p.root((sway, y, z), 0.0)
        p.pelvis(pitch=1.5 * np.sin(4 * np.pi * ph_l) + params["lean"] * 0.3,
                 roll=params["pel_roll"] * np.sin(2 * np.pi * ph_l),
                 yaw=-params["pel_yaw"] * np.cos(2 * np.pi * ph_l))
        p.spine(pitch=params["lean"] * 0.7, roll=-0.5 * params["pel_roll"] * np.sin(2 * np.pi * ph_l),
                yaw=(params["thorax_yaw"] + params["pel_yaw"]) * np.cos(2 * np.pi * ph_l))
        p.neck(pitch=params["head_pitch"] - params["lean"] * 0.6, yaw=-params["thorax_yaw"] * 0.8 * np.cos(2 * np.pi * ph_l))
        for side, sg in (("l", 1.0), ("r", -1.0)):
            ph, ank, theta, toe = legs[side]
            p.leg(side, ank, ank + np.array([0.0, -0.6, 0.45]), foot_rot=pitch_rot(theta) @ skel.A[f"foot_{side}"], toe=toe)
            sw = np.cos(2 * np.pi * ph)        # +1 when this leg strikes: same-side arm swings back
            fwd = max(-sw, 0.0)
            wrist = np.array([sway + sg * (params["arm_x"] - 0.03 * fwd * params["run"]),
                              y + params["arm_swing"] * sw - 0.03 - 0.03 * params["run"],
                              params["arm_z"] + z + params["arm_lift"] * fwd])
            p.arm(side, wrist, np.array([sg * 0.30, y + 0.55, 1.10 + z]))
            p.clavicle(side, up=params["clav_up"], fwd=2.0)
            p.fingers(side, curl=params["curl"], thumb=0.3)
        return p

    # pass 1: hip heights needed for both feet to reach; widen with a max filter, then smooth
    drops = np.array([_hip_drop(skel, build(c, 0.0).solve()[3], {s: c[5][s][1] for s in ("l", "r")}) for c in controls])
    w = 4
    padded = np.pad(drops, w, mode="edge")
    mx = np.array([padded[i:i + 2 * w + 1].max() for i in range(n)])
    kern = np.exp(-0.5 * (np.arange(-w, w + 1) / (w / 2)) ** 2)
    kern /= kern.sum()
    sm = np.convolve(np.pad(mx, w, mode="edge"), kern, mode="valid")
    out = []
    for c, d in zip(controls, sm):
        p = build(c, max(d, 0.0))
        out.append(solve(p, {"pelvis": np.array([c[3], c[4], 0.95 + c[2] - d])}))
    return out


# ------------------------------------------------------------------ shared helpers
G = 9.81
# Dempster segment masses (fraction of body mass): bone -> (fraction, centre position along the bone 0..1)
SEGMENTS = {"head": (0.081, 0.45), "spine_04": (0.16, 0.5), "spine_03": (0.14, 0.5), "spine_02": (0.10, 0.5),
            "spine_01": (0.06, 0.5), "pelvis": (0.142, 0.4),
            "upperarm_l": (0.028, 0.44), "upperarm_r": (0.028, 0.44), "lowerarm_l": (0.016, 0.43),
            "lowerarm_r": (0.016, 0.43), "hand_l": (0.006, 0.5), "hand_r": (0.006, 0.5),
            "thigh_l": (0.10, 0.43), "thigh_r": (0.10, 0.43), "calf_l": (0.0465, 0.43), "calf_r": (0.0465, 0.43),
            "foot_l": (0.0145, 0.5), "foot_r": (0.0145, 0.5)}


def com(skel, W, H):
    c, m = np.zeros(3), 0.0
    for b, (frac, at) in SEGMENTS.items():
        tail = H[b] + W[b][:, 1] * skel.length[b]
        c += frac * (H[b] + at * (tail - H[b]))
        m += frac
    return c / m


def stand(skel, p, base=(0.0, 0.0), spread=0.110, stagger=0.03, toe_out=8.0, z_drop=0.0, curl=0.35):
    """Relaxed standing: planted feet, hanging arms. Returns the ankle targets used."""
    bx, by = base
    ank = {"l": np.array([bx + spread, by - stagger, ANKLE_H + LIFT]), "r": np.array([bx - spread, by + stagger, ANKLE_H + LIFT])}
    for side, sg in (("l", 1.0), ("r", -1.0)):
        rot = axis_angle([0, 0, 1], -sg * toe_out) @ skel.A[f"foot_{side}"]
        rot = axis_angle([0, 0, 1], sg * toe_out - sg * 8.0) @ skel.A[f"foot_{side}"]
        p.leg(side, ank[side], ank[side] + np.array([sg * 0.05, -0.6, 0.45]), foot_rot=rot)
        p.fingers(side, curl=curl, thumb=0.3)
    return ank


def hang_arms(p, root_xy, z_off, sway=0.0, fwd=0.0, width=0.23, height=0.865):
    x, y = root_xy
    for side, sg in (("l", 1.0), ("r", -1.0)):
        wrist = np.array([x + sg * width, y - 0.005 + sg * fwd * 0.0 + fwd, height + z_off + sway])
        p.arm(side, wrist, np.array([x + sg * 0.32, y + 0.55, 1.15 + z_off]))
        p.clavicle(side, up=-2.0, fwd=1.0)


# ------------------------------------------------------------------ idle
def idle(skel, seconds=6.0):
    n = int(seconds * FPS)
    out = []
    rng = np.random.default_rng(3)
    look = [(0.0, (0.0, 0.0))] + [(t, (rng.normal(0, 9), rng.normal(-2, 3))) for t in np.arange(1.2, seconds + 1.2, 1.4)]
    for i in range(n):
        t = i / FPS
        breath = np.sin(2 * np.pi * t / 4.2)
        shift = np.sin(2 * np.pi * t / seconds)           # weight shifts left <-> right once per loop
        p = Pose(skel)
        drop = 0.012 + 0.004 * abs(shift)
        p.root((0.022 * shift, 0.0, LIFT - drop), 0.0)
        p.pelvis(pitch=1.0, roll=-2.2 * shift, yaw=1.5 * shift)
        p.spine(pitch=-0.8 * breath + 1.5, roll=1.6 * shift, yaw=-1.0 * shift)
        yaw, pitch = interp_keys(t, look)
        p.neck(pitch=pitch, yaw=yaw, roll=-1.0 * shift)
        stand(skel, p, base=(0.0, 0.0), curl=0.36 + 0.03 * np.sin(t * 1.3))
        hang_arms(p, (0.022 * shift, 0.0), -drop, sway=0.004 * breath)
        for side in ("l", "r"):
            p.clavicle(side, up=0.9 * breath, fwd=0.0)
        f = solve(p, {"pelvis": np.array([0.0, 0.0, 1.0])})
        f.extra["look"] = (yaw, pitch)
        out.append(f)
    return out


# ------------------------------------------------------------------ jump (standing broad jump)
def jump(skel, seconds=3.2, distance=1.10):
    n = int(seconds * FPS)
    t_dip, t_off, flight = 0.95, 1.25, 0.46
    t_land = t_off + flight
    v_up = G * flight / 2
    out = []
    # timeline of controls: (time, hip drop, torso pitch, arm swing angle (deg, + forward/up))
    drop_k = [(0.0, 0.012), (0.55, 0.020), (t_dip, 0.27), (t_off - 0.03, 0.03), (t_off, 0.0)]
    pitch_k = [(0.0, 2.0), (0.55, 6.0), (t_dip, 40.0), (t_off, 12.0)]
    arm_k = [(0.0, 0.0), (0.55, -10.0), (t_dip, -55.0), (t_off - 0.05, 120.0), (t_off + 0.15, 135.0), (t_land - 0.1, 70.0),
             (t_land + 0.12, 35.0), (t_land + 0.6, 10.0), (seconds, 0.0)]
    land_drop_k = [(t_land, 0.05), (t_land + 0.16, 0.30), (t_land + 0.55, 0.10), (t_land + 1.0, 0.015), (seconds, 0.012)]
    land_pitch_k = [(t_land, 15.0), (t_land + 0.16, 38.0), (t_land + 0.6, 12.0), (seconds, 3.0)]
    for i in range(n):
        t = i / FPS
        p = Pose(skel)
        if t < t_off:
            y_feet = 0.0
            drop = interp_keys(t, drop_k)
            pitch = interp_keys(t, pitch_k)
            p.root((0.0, 0.03 * smooth(t / t_dip) - 0.06 * smooth((t - t_dip) / (t_off - t_dip)), LIFT - drop), 0.0)
            p.pelvis(pitch=pitch * 0.35)
            p.spine(pitch=pitch * 0.65)
            p.neck(pitch=-pitch * 0.55)
            ank = stand(skel, p, base=(0.0, 0.0), spread=0.11, stagger=0.0, curl=0.3)
            if t > t_off - 0.10:                      # toe push: heels rise in the last 100 ms
                k = smooth((t - (t_off - 0.10)) / 0.10)
                for side, sg in (("l", 1.0), ("r", -1.0)):
                    flat = ank[side]
                    a2 = stance_ankle(flat, -35.0 * k)
                    p.leg(side, a2, a2 + np.array([0, -0.6, 0.45]), foot_rot=pitch_rot(-35.0 * k) @ skel.A[f"foot_{side}"], toe=35.0 * k)
        elif t < t_land:
            s = (t - t_off) / flight
            y = -0.06 - distance * s
            z = v_up * (t - t_off) - 0.5 * G * (t - t_off) ** 2
            tuck = np.sin(np.pi * s) ** 1.2
            p.root((0.0, y, LIFT + z - 0.0), 0.0)
            p.pelvis(pitch=10 - 25 * tuck)
            p.spine(pitch=8 + 10 * tuck)
            p.neck(pitch=-10)
            for side, sg in (("l", 1.0), ("r", -1.0)):
                p.rot(f"thigh_{side}", 0, 20 + 75 * tuck - 15 * (1 - s)).rot(f"calf_{side}", 0, -(30 + 70 * tuck))
                p.fingers(side, curl=0.3, thumb=0.3)
        else:
            y_l = -0.06 - distance
            drop = interp_keys(t, land_drop_k)
            pitch = interp_keys(t, land_pitch_k)
            settle = smooth((t - t_land) / 0.9)
            p.root((0.0, y_l - 0.10 * (1 - settle) + 0.0, LIFT - drop), 0.0)
            p.pelvis(pitch=pitch * 0.35)
            p.spine(pitch=pitch * 0.65)
            p.neck(pitch=-pitch * 0.55)
            stand(skel, p, base=(0.0, y_l - 0.05), spread=0.12, stagger=0.0, curl=0.3)
        # arms: swing in the sagittal plane around the shoulders
        a = np.radians(interp_keys(t, arm_k))
        q, loc, W, H = p.solve()
        for side, sg in (("l", 1.0), ("r", -1.0)):
            sh = H[f"upperarm_{side}"]
            reach = 0.52
            wrist = sh + np.array([sg * 0.10, -np.sin(a) * reach, -np.cos(a) * reach])
            p.arm(side, wrist, sh + np.array([sg * 0.25, 0.4, -0.2]) if np.degrees(a) < 60 else sh + np.array([sg * 0.4, 0.2, 0.3]))
        out.append(solve(p, {"pelvis": np.array([0.0, H["pelvis"][1], 1.0])}))
    return out


# ------------------------------------------------------------------ backflip
def backflip(skel, seconds=3.3):
    n = int(seconds * FPS)
    t_dip, t_off, flight = 0.80, 1.05, 0.72
    t_land = t_off + flight
    v_up = G * flight / 2
    out = []
    drop_k = [(0.0, 0.012), (0.4, 0.02), (t_dip, 0.25), (t_off - 0.02, 0.02), (t_off, 0.0)]
    pitch_k = [(0.0, 2.0), (0.4, 5.0), (t_dip, 30.0), (t_off, -12.0)]
    arm_k = [(0.0, 0.0), (0.4, -15.0), (t_dip, -60.0), (t_off, 150.0), (t_off + 0.12, 170.0)]
    land_drop_k = [(t_land, 0.06), (t_land + 0.18, 0.32), (t_land + 0.65, 0.10), (t_land + 1.1, 0.015), (seconds, 0.012)]
    land_pitch_k = [(t_land, 20.0), (t_land + 0.18, 42.0), (t_land + 0.7, 12.0), (seconds, 3.0)]
    land_arm_k = [(t_land - 0.15, 60.0), (t_land + 0.2, 70.0), (t_land + 0.8, 20.0), (seconds, 0.0)]
    travel = 0.30          # backflips drift backwards (+Y)
    com0 = None
    for i in range(n):
        t = i / FPS
        p = Pose(skel)
        if t < t_off:
            drop = interp_keys(t, drop_k)
            pitch = interp_keys(t, pitch_k)
            p.root((0.0, 0.02 * smooth(t / t_dip), LIFT - drop), 0.0)
            p.pelvis(pitch=pitch * 0.35)
            p.spine(pitch=pitch * 0.65)
            p.neck(pitch=-pitch * 0.4 - (20 if t > t_dip else 0) * smooth((t - t_dip) / 0.2))
            ank = stand(skel, p, base=(0.0, 0.0), spread=0.11, stagger=0.0, curl=0.3)
            if t > t_off - 0.10:
                k = smooth((t - (t_off - 0.10)) / 0.10)
                for side, sg in (("l", 1.0), ("r", -1.0)):
                    a2 = stance_ankle(ank[side], -30.0 * k)
                    p.leg(side, a2, a2 + np.array([0, -0.6, 0.45]), foot_rot=pitch_rot(-30.0 * k) @ skel.A[f"foot_{side}"], toe=30.0 * k)
            a = np.radians(interp_keys(t, arm_k))
            q, loc, W, H = p.solve()
            if t > t_off - 0.25:
                com0 = com(skel, W, H)
        elif t < t_land:
            s = (t - t_off) / flight
            tuck = smooth(s / 0.32) * (1 - smooth((s - 0.62) / 0.30))
            ang = -360.0 * (smooth(s) * 0.85 + s * 0.15)        # backward rotation (pitch negative)
            # body shape in its own frame (unrotated), then rotate about the COM
            p.root((0.0, 0.0, LIFT), 0.0)
            p.spine(pitch=10 + 25 * tuck)
            p.neck(pitch=-15 + 25 * tuck)
            for side, sg in (("l", 1.0), ("r", -1.0)):
                p.rot(f"thigh_{side}", 0, 15 + 110 * tuck).rot(f"calf_{side}", 0, -(20 + 115 * tuck))
                p.rot(f"foot_{side}", 0, -15 * tuck)
                p.fingers(side, curl=0.5, thumb=0.4)
            q, loc, W, H = p.solve()
            c_body = com(skel, W, H)
            c_world = com0 + np.array([0.0, travel * s, v_up * (t - t_off) - 0.5 * G * (t - t_off) ** 2])
            R = axis_angle([1, 0, 0], ang)        # negative rotation about +X carries the head backwards (+Y)
            pel = H["pelvis"]
            # apply world rotation R at the pelvis: W_pelvis = W_root @ L @ B  ->  B' = L^T W_root^T R W_pelvis_old
            L = skel.A["root"].T @ skel.A["pelvis"]
            p.B["pelvis"] = L.T @ W["root"].T @ R @ W["pelvis"]
            new_pel = c_world + R @ (pel - c_body)
            p.root_loc = p.root_loc + (new_pel - pel)
            a = np.radians(interp_keys(min(t, t_off + 0.2), arm_k)) if s < 0.25 else np.radians(160 - 100 * smooth((s - 0.25) / 0.7))
        else:
            drop = interp_keys(t, land_drop_k)
            pitch = interp_keys(t, land_pitch_k)
            yl = travel + 0.02
            p.root((0.0, yl, LIFT - drop), 0.0)
            p.pelvis(pitch=pitch * 0.35)
            p.spine(pitch=pitch * 0.65)
            p.neck(pitch=-pitch * 0.5)
            stand(skel, p, base=(0.0, yl), spread=0.12, stagger=0.0, curl=0.3)
            a = np.radians(interp_keys(t, land_arm_k))
        q, loc, W, H = p.solve()
        for side, sg in (("l", 1.0), ("r", -1.0)):
            sh = H[f"upperarm_{side}"]
            Wsp = W["spine_04"]
            up = Wsp[:, 1]
            wrist = sh + sg * 0.08 * Wsp[:, 0] + 0.50 * (np.sin(a) * Wsp[:, 2] - np.cos(a) * up)
            p.arm(side, wrist, sh - 0.3 * Wsp[:, 2] + sg * 0.2 * Wsp[:, 0])
        f = solve(p, {"pelvis": np.array([0.0, 0.15, 1.1])})
        out.append(f)
    return out


# ------------------------------------------------------------------ physics clips (ragdoll)
def _ragdoll_frames(skel, start, prev, seconds, setup, per_frame=None):
    """Simulate from an animated frame; returns frames keyed from the particle state."""
    from .ragdoll import Ragdoll, to_rotations
    rd = Ragdoll(skel, start.W, start.H, prev.W, prev.H)
    setup(rd)
    out = []
    pel_off = start.H["pelvis"] - start.H["root"]
    for i in range(int(seconds * FPS)):
        if per_frame:
            per_frame(rd, i)
        rd.step()
        W, P = to_rotations(skel, rd, rd.x, start.W)
        root_head = P - pel_off
        q, loc = skel.basis(W, root_head - skel.head["root"])
        Wf, Hf = skel.fk({b: _mat(qq) for b, qq in q.items()}, root_head - skel.head["root"])
        out.append(Frame(q, loc, Wf, Hf, {"pelvis": P.copy()}))
    return out


def _mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def fall(skel, seconds=4.5):
    """Walk, the right foot catches mid-swing, he pitches forward, braces with his arms, hits the ground."""
    walk = locomotion(skel, 1.75, WALK)
    trip_i = len(walk) - 1
    from .ragdoll import Ragdoll

    def setup(rd):
        # swing foot caught on an obstacle: pin the right toe briefly
        rd.pins["toe_r"] = rd.x[rd.idx["toe_r"]].copy()
        # protective reaction: arms forward and down, head up
        tgt = rd.x.copy()
        F, P = rd.pelvis_frame(rd.x)
        for side, sg in (("l", 1.0), ("r", -1.0)):
            sh = rd.x[rd.idx[f"sh_{side}"]]
            tgt[rd.idx[f"el_{side}"]] = sh + F[:, 2] * 0.22 - F[:, 1] * 0.12 + sg * F[:, 0] * 0.08
            tgt[rd.idx[f"wr_{side}"]] = sh + F[:, 2] * 0.45 - F[:, 1] * 0.25 + sg * F[:, 0] * 0.10
            tgt[rd.idx[f"fi_{side}"]] = sh + F[:, 2] * 0.62 - F[:, 1] * 0.30 + sg * F[:, 0] * 0.11
        rd.set_tone(0.03, tgt)

    def per_frame(rd, i):
        if i == 2:
            rd.pins.pop("toe_r", None)
        rd.tone = 0.03 if i < 20 else (0.012 if i < 36 else 0.0)
        rd.friction = 0.75

    sim = _ragdoll_frames(skel, walk[trip_i], walk[trip_i - 1], seconds - 1.75, setup, per_frame)
    return walk + sim


def shot(skel, seconds=4.5):
    """Standing; a round hits the upper left chest; flinch, stagger, knees give, collapse."""
    base = idle(skel, 1.2)
    hit = len(base) - 1

    def setup(rd):
        F, P = rd.pelvis_frame(rd.x)
        back = -F[:, 2]
        # a 9 mm round carries ~3 kg m/s: barely moves an 80 kg man; the jolt is his own flinch
        rd.impulse("chest", back * 0.35 + F[:, 0] * 0.1)
        rd.impulse("sh_l", back * 0.9)
        # flinch: right hand to the wound, shoulders hunch, knees soften
        tgt = rd.x.copy()
        wound = rd.x[rd.idx["chest"]] + F[:, 2] * 0.11 + F[:, 0] * 0.08 + F[:, 1] * 0.02
        tgt[rd.idx["wr_r"]] = wound + F[:, 0] * -0.02
        tgt[rd.idx["fi_r"]] = wound + F[:, 0] * 0.06
        tgt[rd.idx["el_r"]] = rd.x[rd.idx["sh_r"]] + F[:, 2] * 0.18 - F[:, 1] * 0.20
        rd.set_tone(0.05, tgt)

    def per_frame(rd, i):
        rd.tone = 0.05 if i < 12 else (0.025 if i < 30 else (0.008 if i < 48 else 0.0))

    sim = _ragdoll_frames(skel, base[hit], base[hit - 1], seconds - 1.2, setup, per_frame)
    for f in base:
        f.extra["pelvis"] = np.array([0.0, 0.0, 0.95])
    return base + sim


# ------------------------------------------------------------------ crouch
def crouch(skel, seconds=5.5):
    n = int(seconds * FPS)
    out = []
    drop_k = [(0.0, 0.012), (0.7, 0.015), (1.7, 0.43), (3.9, 0.43), (4.9, 0.015), (seconds, 0.012)]
    lean_k = [(0.0, 2.0), (0.7, 3.0), (1.7, 32.0), (3.9, 30.0), (4.9, 3.0), (seconds, 2.0)]
    look_k = [(0.0, 0.0), (1.9, 0.0), (2.4, 38.0), (2.9, 38.0), (3.4, -34.0), (3.8, -30.0), (4.3, 0.0), (seconds, 0.0)]
    heel_k = [(0.0, 0.0), (1.2, 0.0), (1.7, 18.0), (3.9, 18.0), (4.5, 0.0), (seconds, 0.0)]
    for i in range(n):
        t = i / FPS
        p = Pose(skel)
        drop = interp_keys(t, drop_k)
        lean = interp_keys(t, lean_k)
        look = interp_keys(t, look_k)
        heel = interp_keys(t, heel_k)
        crouched = smooth((drop - 0.02) / 0.4)
        p.root((0.0, 0.035 * crouched, LIFT - drop), 0.0)
        p.pelvis(pitch=lean * 0.5)
        p.spine(pitch=lean * 0.5, yaw=look * 0.25)
        p.neck(pitch=-lean * 0.8, yaw=look * 0.75)
        p.gaze(yaw=look * 0.15)
        for side, sg in (("l", 1.0), ("r", -1.0)):
            flat = np.array([sg * (0.11 + 0.05 * crouched), -0.02, ANKLE_H + LIFT])
            a = stance_ankle(flat, -heel)
            rot = axis_angle([0, 0, 1], sg * 14.0 * crouched) @ pitch_rot(-heel) @ skel.A[f"foot_{side}"]
            p.leg(side, a, a + np.array([sg * 0.25 * crouched, -0.6, 0.45]), foot_rot=rot, toe=heel)
            knee_front = np.array([sg * 0.20, -0.30, 0.50 + 0.45 * (1 - crouched)])
            wrist = (1 - crouched) * np.array([sg * 0.23, 0.0, 0.865 - drop]) + crouched * (knee_front + [sg * -0.04, -0.08, 0.03])
            p.arm(side, wrist, np.array([sg * 0.45, 0.4, 0.9]))
            p.fingers(side, curl=0.35 + 0.2 * crouched, thumb=0.3)
        out.append(solve(p, {"pelvis": np.array([0.0, 0.0, 1.0 - 0.3 * crouched])}))
    return out


# ------------------------------------------------------------------ punch combo
def punch(skel, seconds=4.6):
    n = int(seconds * FPS)
    out = []
    stance = lambda t: smooth((t - 0.2) / 0.5) * (1 - smooth((t - 3.9) / 0.6))
    hits = {"jab": (1.10, "l"), "cross": (1.75, "r"), "hook": (2.45, "l"), "jab2": (3.05, "l")}
    face = np.array([0.0, -0.78, 1.60])

    def ext(t, t0, out_t=0.11, back_t=0.17):
        if t < t0 - out_t or t > t0 + back_t:
            return 0.0
        return smooth((t - (t0 - out_t)) / out_t) if t < t0 else 1 - smooth((t - t0) / back_t)

    for i in range(n):
        t = i / FPS
        s = stance(t)
        jab = max(ext(t, hits["jab"][0]), ext(t, hits["jab2"][0]))
        cross = ext(t, hits["cross"][0], 0.13, 0.20)
        hook = ext(t, hits["hook"][0], 0.14, 0.22)
        p = Pose(skel)
        twist = -25.0 * s + 40.0 * cross - 35.0 * hook + 8.0 * jab
        p.root((0.0, -0.05 * cross, LIFT - 0.012 - 0.06 * s), -14.0 * s)
        p.pelvis(pitch=3.0 * s, yaw=twist * 0.45)
        p.spine(pitch=8.0 * s + 4 * cross, yaw=twist * 0.55, roll=-6.0 * hook)
        p.neck(pitch=-6.0 * s, yaw=-twist * 0.7 + 14.0 * s)
        # feet: orthodox stance, left foot forward; rear heel lifts and pivots on the cross
        lf = np.array([0.10 * (1 - s) + 0.05 * s, -0.02 - 0.24 * s, ANKLE_H + LIFT])
        rf = np.array([-0.11 * (1 - s) - 0.17 * s, 0.03 + 0.12 * s, ANKLE_H + LIFT])
        heel_r = 25.0 * cross + 8.0 * s
        ar = stance_ankle(rf, -heel_r, heading=-30 * s)
        p.leg("l", lf, lf + [0.05, -0.6, 0.45], foot_rot=axis_angle([0, 0, 1], -10 * s) @ skel.A["foot_l"])
        p.leg("r", ar, ar + [-0.1, -0.6, 0.45], foot_rot=axis_angle([0, 0, 1], -30 * s + 25 * cross) @ pitch_rot(-heel_r) @ skel.A["foot_r"], toe=heel_r)
        q, loc, W, H = p.solve()
        for side, sg in (("l", 1.0), ("r", -1.0)):
            sh = H[f"upperarm_{side}"]
            guard = np.array([sg * 0.06, -0.22 if side == "l" else -0.14, 1.52 if side == "l" else 1.47])
            rest = np.array([sg * 0.23, 0.0, 0.865])
            base = rest * (1 - s) + guard * s
            if side == "l":
                tgt = face + np.array([0.04, 0.0, -0.02])
                hook_pt = np.array([-0.05, -0.50, 1.55])
                wrist = base + (tgt - base) * jab + (hook_pt - base) * hook
                pole = sh + np.array([0.35, 0.15, -0.35]) if hook < 0.2 else sh + np.array([0.3, 0.0, 0.25])
            else:
                tgt = face + np.array([-0.02, 0.05, 0.0])
                wrist = base + (tgt - base) * cross
                pole = sh + np.array([-0.35, 0.2, -0.35])
            p.arm(side, wrist, pole, hand_twist=(-70.0 * (jab if side == "l" else cross) * sg) - 20 * s * sg)
            p.fingers(side, curl=0.35 + 0.65 * s, thumb=0.35 + 0.55 * s)
        out.append(solve(p, {"pelvis": np.array([0.0, 0.0, 1.05])}))
    return out


# ------------------------------------------------------------------ registry: name -> (frames, camera spec)
CLIPS = {
    "walk": lambda s: (locomotion(s, 4.4, WALK), {"offset": (2.9, -2.4, 0.25), "lens": 38, "smooth": 12}),
    "run": lambda s: (locomotion(s, 3.0, RUN), {"offset": (3.6, -3.0, 0.20), "lens": 35, "smooth": 10}),
    "idle": lambda s: (idle(s), {"offset": (2.2, -3.6, 0.0), "lens": 45, "smooth": 30}),
    "jump": lambda s: (jump(s), {"offset": (3.4, -1.2, 0.0), "lens": 35, "smooth": 30}),
    "backflip": lambda s: (backflip(s), {"offset": (3.6, 0.0, 0.1), "lens": 35, "smooth": 30}),
    "fall": lambda s: (fall(s), {"offset": (3.0, -1.6, 0.0), "lens": 35, "smooth": 18}),
    "shot": lambda s: (shot(s), {"offset": (2.8, -2.2, 0.0), "lens": 38, "smooth": 20}),
    "crouch": lambda s: (crouch(s), {"offset": (2.6, -3.2, -0.1), "lens": 40, "smooth": 30}),
    "punch": lambda s: (punch(s), {"offset": (3.2, -2.0, 0.0), "lens": 40, "smooth": 30}),
}
