"""Pose engine (numpy only, runs inside Blender too).

Rest data comes from build/rig.json (bone head/tail/parent/roll axis, Blender conventions: bone
Y runs head->tail, Z points along the roll axis). A pose is solved into world rotations W[b]
(3x3, columns = bone x/y/z axes in world space) and world heads, then converted to Blender pose
basis quaternions:  W_b = W_parent @ (A_parent^T A_b) @ B_b   (A = rest frames).

Controls (all optional, world-space or in the character's heading frame):
  root(pos, yaw)            pelvis(rot)            spine(pitch, roll, yaw)
  neck(pitch, roll, yaw)    clavicle(side, up, fwd)
  arm(side, wrist, pole, hand_rot)                 leg(side, ankle, pole, foot_rot, toe)
  fingers(side, curl, spread, thumb)               jaw(open)    look(point)
"""
import json

import numpy as np


def nrm(v):
    v = np.asarray(v, float)
    return v / max(np.linalg.norm(v), 1e-12)


def axis_angle(axis, deg):
    a = nrm(axis)
    t = np.radians(deg)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(t) * K + (1 - np.cos(t)) * K @ K


def frame_from(y, z_hint):
    y = nrm(y)
    z = nrm(z_hint - y * np.dot(z_hint, y))
    return np.stack([np.cross(y, z), y, z], axis=1)


def quat_from_matrix(m):
    t = np.trace(m)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        q = [0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s]
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        q = [(m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s]
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        q = [(m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s]
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        q = [(m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def twist_about_y(m):
    """Signed rotation angle (deg) of local rotation m about its Y axis (swing-twist)."""
    q = quat_from_matrix(m)
    return np.degrees(2 * np.arctan2(q[2], q[0]))


class Skeleton:
    def __init__(self, rig):
        if isinstance(rig, str):
            rig = json.load(open(rig))
        self.rig = rig
        order, seen = [], set()

        def visit(b):
            if b in seen:
                return
            p = rig[b]["parent"]
            if p:
                visit(p)
            seen.add(b)
            order.append(b)

        for b in rig:
            visit(b)
        self.order = order
        self.parent = {b: rig[b]["parent"] for b in rig}
        self.head = {b: np.array(rig[b]["head"]) for b in rig}
        self.tail = {b: np.array(rig[b]["tail"]) for b in rig}
        self.A = {b: frame_from(self.tail[b] - self.head[b], np.array(rig[b]["z"])) for b in rig}
        self.length = {b: float(np.linalg.norm(self.tail[b] - self.head[b])) for b in rig}

    def offset(self, b):
        """Bone head relative to parent head, in the parent's rest frame."""
        p = self.parent[b]
        return self.A[p].T @ (self.head[b] - self.head[p])

    def basis(self, W, root_loc=None):
        """World rotations -> {bone: quaternion (w, x, y, z)} and root location (bone-local)."""
        out = {}
        for b in self.order:
            p = self.parent[b]
            if p is None:
                B = self.A[b].T @ W[b]
            else:
                L = self.A[p].T @ self.A[b]
                B = L.T @ W[p].T @ W[b]
            out[b] = quat_from_matrix(B)
        loc = self.A["root"].T @ (np.zeros(3) if root_loc is None else np.asarray(root_loc))
        return out, loc

    def fk(self, B, root_loc=np.zeros(3)):
        """Local basis rotations (3x3) -> world rotations and heads."""
        W, H = {}, {}
        for b in self.order:
            p = self.parent[b]
            Bb = B.get(b, np.eye(3))
            if p is None:
                W[b] = self.A[b] @ Bb
                H[b] = self.head[b] + np.asarray(root_loc)
            else:
                W[b] = W[p] @ (self.A[p].T @ self.A[b]) @ Bb
                H[b] = H[p] + W[p] @ self.offset(b)
        return W, H


class Pose:
    """High-level pose builder; call solve() -> (quaternions, root_loc, W, H)."""

    SPINE = [("spine_01", 0.20), ("spine_02", 0.25), ("spine_03", 0.30), ("spine_04", 0.25)]

    def __init__(self, skel):
        self.s = skel
        self.B = {}
        self.root_loc = np.zeros(3)
        self.yaw = 0.0
        self.ik = {}

    # ---- local helpers: rotate bone b by deg about one of ITS rest axes (0=x, 1=y, 2=z)
    def rot(self, b, axis, deg):
        self.B[b] = self.B.get(b, np.eye(3)) @ axis_angle(np.eye(3)[axis], deg)
        return self

    def root(self, pos=(0.0, 0.0, 0.0), yaw=0.0):
        """pos: world offset of the whole body; yaw: heading in degrees (CCW from above)."""
        self.root_loc = np.asarray(pos, float)
        self.yaw = yaw
        self.B["root"] = self.s.A["root"].T @ axis_angle([0, 0, 1], yaw) @ self.s.A["root"]
        return self

    def pelvis(self, pitch=0.0, roll=0.0, yaw=0.0):
        """pitch + tilts the pelvis forward (anterior tilt), roll + drops his right side, yaw + turns left."""
        return self.rot("pelvis", 0, pitch).rot("pelvis", 2, -roll).rot("pelvis", 1, yaw)

    def spine(self, pitch=0.0, roll=0.0, yaw=0.0, weights=None):
        """pitch + bends forward, roll + bends to his left, yaw + twists to his left."""
        for b, w in (weights or self.SPINE):
            self.rot(b, 0, pitch * w).rot(b, 2, -roll * w).rot(b, 1, yaw * w)
        return self

    def neck(self, pitch=0.0, roll=0.0, yaw=0.0, head_share=0.55):
        for b, w in (("neck_01", (1 - head_share) * 0.5), ("neck_02", (1 - head_share) * 0.5), ("head", head_share)):
            self.rot(b, 0, pitch * w).rot(b, 2, -roll * w).rot(b, 1, yaw * w)
        return self

    def clavicle(self, side, up=0.0, fwd=0.0):
        """Shrug (up +) and protraction (fwd +) of the shoulder girdle in degrees."""
        b = f"clavicle_{side}"
        sgn = 1.0 if side == "l" else -1.0
        return self.rot(b, 0, up).rot(b, 2, fwd * sgn)

    def jaw(self, open_deg):
        return self.rot("jaw", 0, -open_deg)

    def arm(self, side, wrist, pole, hand_rot=None, hand_twist=0.0):
        """IK the arm so the wrist reaches `wrist` (world); elbow towards `pole` (world point).
        hand_rot: world 3x3 for the hand, or None to continue the forearm; hand_twist: extra roll (deg)."""
        self.ik[f"arm_{side}"] = (np.asarray(wrist, float), np.asarray(pole, float), hand_rot, hand_twist)
        return self

    def leg(self, side, ankle, pole, foot_rot=None, toe=0.0):
        """IK the leg so the ankle reaches `ankle`; knee towards `pole`; foot_rot world 3x3; toe bend deg."""
        self.ik[f"leg_{side}"] = (np.asarray(ankle, float), np.asarray(pole, float), foot_rot, toe)
        return self

    def fingers(self, side, curl=0.15, spread=0.0, thumb=0.2, per=None):
        """curl 0 (flat) .. 1 (fist); per = {"index": c, ...} overrides; thumb 0..1 opposition/curl."""
        per = per or {}
        for f in ("index", "middle", "ring", "pinky"):
            c = per.get(f, curl)
            sp = spread * {"index": 1.0, "middle": 0.2, "ring": -0.6, "pinky": -1.2}[f]
            for i, mx in enumerate((85, 100, 70)):
                self.rot(f"{f}_{i + 1:02d}_{side}", 0, -mx * c)
            self.rot(f"{f}_01_{side}", 2, sp * (1 if side == "l" else -1))
        t = per.get("thumb", thumb)
        self.rot(f"thumb_01_{side}", 0, -25 * t).rot(f"thumb_02_{side}", 0, -35 * t).rot(f"thumb_03_{side}", 0, -45 * t)
        return self

    # ---- solve
    def _two_bone(self, upper, lower, start, target, pole, rest_pole_dir):
        s = self.s
        L1, L2 = s.length[upper], s.length[lower]
        d_vec = target - start
        d = np.clip(np.linalg.norm(d_vec), abs(L1 - L2) + 1e-4, L1 + L2 - 1e-5)
        u = nrm(d_vec)
        v = pole - start
        v = nrm(v - u * np.dot(v, u))
        ca = np.clip((L1 * L1 + d * d - L2 * L2) / (2 * L1 * d), -1, 1)
        mid = start + L1 * (ca * u + np.sqrt(1 - ca * ca) * v)
        end = start + u * d
        # hinge axis sign matched to the rest frame
        u0 = nrm(s.tail[lower] - s.head[upper])
        h0 = nrm(np.cross(u0, rest_pole_dir))
        sign = 1.0 if np.dot(h0, s.A[upper][:, 2]) >= 0 else -1.0
        h = sign * nrm(np.cross(u, v))
        Wu = frame_from(mid - start, h)
        Wl = frame_from(end - mid, h)
        return Wu, Wl, end

    def solve(self):
        s = self.s
        W, H = s.fk(self.B, self.root_loc)
        heading = axis_angle([0, 0, 1], self.yaw)
        for side in ("l", "r"):
            key = f"leg_{side}"
            if key in self.ik:
                ankle, pole, foot_rot, toe = self.ik[key]
                Wu, Wl, end = self._two_bone(f"thigh_{side}", f"calf_{side}", H[f"thigh_{side}"], ankle, pole,
                                             np.array([0.0, -1.0, 0.0]))
                W[f"thigh_{side}"], W[f"calf_{side}"] = Wu, Wl
                self.B[f"thigh_{side}"] = self._local(W, f"thigh_{side}")
                self.B[f"calf_{side}"] = self._local(W, f"calf_{side}")
                fr = foot_rot if foot_rot is not None else heading @ s.A[f"foot_{side}"]
                W2, _ = s.fk(self.B, self.root_loc)
                self.B[f"foot_{side}"] = (W2[f"calf_{side}"] @ (s.A[f"calf_{side}"].T @ s.A[f"foot_{side}"])).T @ fr
                self.B[f"ball_{side}"] = axis_angle([1, 0, 0], toe)
            key = f"arm_{side}"
            if key in self.ik:
                wrist, pole, hand_rot, hand_twist = self.ik[key]
                W, H = s.fk(self.B, self.root_loc)
                Wu, Wl, end = self._two_bone(f"upperarm_{side}", f"lowerarm_{side}", H[f"upperarm_{side}"], wrist, pole,
                                             np.array([0.0, 1.0, 0.0]))
                W[f"upperarm_{side}"], W[f"lowerarm_{side}"] = Wu, Wl
                self.B[f"upperarm_{side}"] = self._local(W, f"upperarm_{side}")
                self.B[f"lowerarm_{side}"] = self._local(W, f"lowerarm_{side}")
                W, H = s.fk(self.B, self.root_loc)
                if hand_rot is None:
                    Bh = np.eye(3)
                else:
                    Bh = (W[f"lowerarm_{side}"] @ (s.A[f"lowerarm_{side}"].T @ s.A[f"hand_{side}"])).T @ hand_rot
                Bh = Bh @ axis_angle([0, 1, 0], hand_twist)
                tw = twist_about_y(Bh)
                self.B[f"hand_{side}"] = Bh
                self.B[f"lowerarm_twist_01_{side}"] = axis_angle([0, 1, 0], 0.5 * tw)
        W, H = s.fk(self.B, self.root_loc)
        quats, loc = s.basis(W, self.root_loc)
        return quats, loc, W, H

    def _local(self, W, b):
        s = self.s
        p = s.parent[b]
        L = s.A[p].T @ s.A[b]
        Wp = s.fk(self.B, self.root_loc)[0][p]
        return L.T @ Wp.T @ W[b]
