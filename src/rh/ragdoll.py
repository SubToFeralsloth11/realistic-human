"""Position-based (Verlet) ragdoll for falls and hit reactions.

Particles sit on skeleton joints. Constraints: rigid bone lengths, cross-bracing that keeps the
pelvis, rib cage and head blocks rigid, joint limits (no knee/elbow hyperextension, flexion caps,
spine bend cap), ground contact with Coulomb-like friction, optional pinning (a tripped foot) and
"muscle tone": springs pulling particles toward a target pose expressed in the pelvis frame,
with a stiffness that can fade to zero (unconsciousness).
After simulation each frame is converted back to bone rotations with minimal twist relative to
the parent, so the result keys straight onto the rig.
"""
import numpy as np

from .pose import axis_angle, nrm

G = np.array([0.0, 0.0, -9.81])

# particle name -> (bone, at)   at=0 head of the bone, 1 tail
PARTICLES = {
    "pelvis": ("pelvis", 0.0), "spine": ("spine_03", 0.0), "chest": ("spine_04", 0.0), "neck": ("neck_01", 0.0),
    "head": ("head", 0.25), "top": ("head", 1.0),
    "hip_l": ("thigh_l", 0.0), "knee_l": ("calf_l", 0.0), "ankle_l": ("foot_l", 0.0), "toe_l": ("ball_l", 1.0),
    "hip_r": ("thigh_r", 0.0), "knee_r": ("calf_r", 0.0), "ankle_r": ("foot_r", 0.0), "toe_r": ("ball_r", 1.0),
    "sh_l": ("upperarm_l", 0.0), "el_l": ("lowerarm_l", 0.0), "wr_l": ("hand_l", 0.0), "fi_l": ("hand_l", 1.0),
    "sh_r": ("upperarm_r", 0.0), "el_r": ("lowerarm_r", 0.0), "wr_r": ("hand_r", 0.0), "fi_r": ("hand_r", 1.0),
    "nose": ("head", 0.30),     # offset forward later so the head has an orientation
}
# contact radii are kept a little under the real flesh thickness so the rigid blocks never have
# to satisfy contradictory floor contacts (that is what makes ragdolls pop off the ground)
RADIUS = {"head": 0.085, "top": 0.05, "nose": 0.02, "chest": 0.075, "spine": 0.065, "pelvis": 0.075, "neck": 0.04,
          "sh_l": 0.04, "sh_r": 0.04, "hip_l": 0.055, "hip_r": 0.055, "knee_l": 0.045, "knee_r": 0.045,
          "el_l": 0.035, "el_r": 0.035, "wr_l": 0.025, "wr_r": 0.025, "fi_l": 0.015, "fi_r": 0.015,
          "ankle_l": 0.04, "ankle_r": 0.04, "toe_l": 0.015, "toe_r": 0.015}
BONES = [("pelvis", "spine"), ("spine", "chest"), ("chest", "neck"), ("neck", "head"), ("head", "top"),
         ("head", "nose"), ("hip_l", "knee_l"), ("knee_l", "ankle_l"), ("ankle_l", "toe_l"),
         ("hip_r", "knee_r"), ("knee_r", "ankle_r"), ("ankle_r", "toe_r"), ("sh_l", "el_l"), ("el_l", "wr_l"),
         ("wr_l", "fi_l"), ("sh_r", "el_r"), ("el_r", "wr_r"), ("wr_r", "fi_r")]
BLOCKS = [("pelvis", "hip_l", "hip_r", "spine"), ("chest", "sh_l", "sh_r", "neck", "spine"),
          ("head", "top", "nose", "neck")]


class Ragdoll:
    def __init__(self, skel, W, H, W_prev=None, H_prev=None, dt_frame=1 / 30, substeps=8):
        self.s = skel
        self.names = list(PARTICLES)
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.x = self._positions(W, H)
        prev = self._positions(W_prev, H_prev) if W_prev is not None else self.x.copy()
        self.dt = dt_frame / substeps
        self.sub = substeps
        self.x0 = self.x - (self.x - prev) / substeps      # carry the animation's velocity into the sim
        self.rest = self.x.copy()
        self.r = np.array([RADIUS[n] for n in self.names])
        self.cons = []
        for a, b in BONES:
            self._link(a, b)
        for blk in BLOCKS:
            for i in range(len(blk)):
                for j in range(i + 1, len(blk)):
                    self._link(blk[i], blk[j])
        # hips attach rigidly to the pelvis block, shoulders loosely to the chest
        for a, b in (("hip_l", "spine"), ("hip_r", "spine"), ("sh_l", "spine"), ("sh_r", "spine"),
                     ("sh_l", "pelvis"), ("sh_r", "pelvis")):
            self._link(a, b, stiff=0.6)
        self.pins = {}
        self.tone = 0.0
        self.target_local = None
        self.friction = 0.6

    def _positions(self, W, H):
        P = np.zeros((len(PARTICLES), 3))
        for i, (n, (bone, at)) in enumerate(PARTICLES.items()):
            tail = H[bone] + W[bone][:, 1] * self.s.length[bone]
            P[i] = H[bone] + at * (tail - H[bone])
            if n == "nose":
                P[i] = H["head"] + W["head"][:, 1] * 0.07 + W["head"][:, 2] * 0.10
        return P

    def _link(self, a, b, stiff=1.0):
        i, j = self.idx[a], self.idx[b]
        self.cons.append((i, j, np.linalg.norm(self.x[i] - self.x[j]), stiff))

    def pelvis_frame(self, x):
        L, R, S, P = x[self.idx["hip_l"]], x[self.idx["hip_r"]], x[self.idx["spine"]], x[self.idx["pelvis"]]
        xa = nrm(L - R)
        up = S - P
        up = nrm(up - xa * np.dot(up, xa))
        return np.stack([xa, up, np.cross(xa, up)], 1), P

    def set_tone(self, stiffness, target=None, only=None):
        """Springs toward the target pose (particle positions, world) captured in the pelvis frame.
        only: particle names that are actively held (default: arms)."""
        self.tone = stiffness
        if target is not None:
            F, P = self.pelvis_frame(target)
            self.target_local = (target - P) @ F
        names = only or ["sh_l", "el_l", "wr_l", "fi_l", "sh_r", "el_r", "wr_r", "fi_r"]
        self.tone_mask = np.zeros(len(self.names))
        self.tone_mask[[self.idx[n] for n in names]] = 1.0

    def impulse(self, name, dv):
        """Add a velocity change (m/s) to one particle: Verlet keeps velocity in x - x0 per substep."""
        self.x0[self.idx[name]] -= np.asarray(dv) * self.dt

    def step(self, vmax=7.0):
        for _ in range(self.sub):
            v = (self.x - self.x0) * 0.995
            # inelastic ground: particles touching the floor lose most of their velocity
            touching = (self.x[:, 2] - self.r) < 0.003
            v[touching] *= 0.80
            sp = np.linalg.norm(v, axis=1, keepdims=True)
            v *= np.minimum(1.0, vmax * self.dt / np.maximum(sp, 1e-12))
            self.x0 = self.x.copy()
            self.x = self.x + v + G * self.dt * self.dt
            if self.tone > 0 and self.target_local is not None:
                # internal muscle forces on the held particles, minus their mean so the centre of
                # mass is not dragged (muscles cannot propel a body through the air)
                F, P = self.pelvis_frame(self.x)
                tgt = P + self.target_local @ F.T
                k = 1 - (1 - self.tone) ** (1.0 / self.sub)
                pull = (tgt - self.x) * k * self.tone_mask[:, None]
                pull -= pull.sum(axis=0) / len(pull)
                self.x += pull
            for _ in range(12):
                self._solve()

    def _solve(self):
        x = self.x
        for i, j, L, k in self.cons:
            d = x[j] - x[i]
            dist = np.linalg.norm(d)
            if dist < 1e-9:
                continue
            corr = d * ((dist - L) / dist) * 0.5 * k
            x[i] += corr
            x[j] -= corr
        # flexion caps (~155 deg knee, ~150 deg elbow), corrected softly so a folding collapse is
        # not turned into a jump off the floor
        for root, end, min_span in (("hip_l", "ankle_l", 0.20), ("hip_r", "ankle_r", 0.20),
                                    ("sh_l", "wr_l", 0.14), ("sh_r", "wr_r", 0.14)):
            a, c = x[self.idx[root]], x[self.idx[end]]
            ac = c - a
            span = np.linalg.norm(ac)
            if span < min_span:
                d = (min_span - span) * 0.1 * nrm(ac)
                x[self.idx[root]] -= d
                x[self.idx[end]] += d
        # ground
        z = x[:, 2] - self.r
        hit = z < 0
        if hit.any():
            x[hit, 2] = self.r[hit]
            v = x[hit] - self.x0[hit]
            v[:, :2] *= (1 - self.friction)
            self.x0[hit] = x[hit] - v
            self.x0[hit, 2] = x[hit, 2]
        for n, pos in self.pins.items():
            x[self.idx[n]] = pos


def to_rotations(skel, rd, x, W_rest_like):
    """Particle positions -> world bone rotations (minimal twist down the hierarchy)."""
    s = skel
    idx = rd.idx
    W = dict(W_rest_like)
    F, P = rd.pelvis_frame(x)
    F0, _ = rd.pelvis_frame(rd.rest)
    Rp = F @ F0.T
    for b in ("root",):
        W[b] = W_rest_like[b]
    W["pelvis"] = Rp @ W_rest_like["pelvis"]

    def aim(bone, parent_R, a, b):
        cur = nrm(x[idx[b]] - x[idx[a]])
        rest_dir = parent_R @ W_rest_like[bone][:, 1]
        axis = np.cross(rest_dir, cur)
        sa = np.linalg.norm(axis)
        ang = np.degrees(np.arctan2(sa, np.dot(rest_dir, cur)))
        R = axis_angle(axis, ang) @ parent_R if sa > 1e-9 else parent_R
        W[bone] = R @ W_rest_like[bone]
        return R

    # spine: share the pelvis->chest bend
    Rs = aim("spine_01", Rp, "pelvis", "spine")
    W["spine_02"] = Rs @ W_rest_like["spine_02"]
    Rc = aim("spine_03", Rs, "spine", "chest")
    Rc2 = aim("spine_04", Rc, "chest", "neck")
    Rn = aim("neck_01", Rc2, "neck", "head")
    W["neck_02"] = Rn @ W_rest_like["neck_02"]
    # head orientation from top + nose
    up = nrm(x[idx["top"]] - x[idx["head"]])
    fw = x[idx["nose"]] - x[idx["head"]]
    fw = nrm(fw - up * np.dot(fw, up))
    up0 = nrm(rd.rest[idx["top"]] - rd.rest[idx["head"]])
    fw0 = rd.rest[idx["nose"]] - rd.rest[idx["head"]]
    fw0 = nrm(fw0 - up0 * np.dot(fw0, up0))
    M = np.stack([np.cross(up, fw), up, fw], 1)
    M0 = np.stack([np.cross(up0, fw0), up0, fw0], 1)
    Rh = M @ M0.T
    for b in ("head", "jaw", "eye_l", "eye_r", "lid_upper_l", "lid_lower_l", "lid_upper_r", "lid_lower_r"):
        if b in W_rest_like:
            W[b] = Rh @ W_rest_like[b]
    for side in ("l", "r"):
        Rcl = Rc2
        W[f"clavicle_{side}"] = Rcl @ W_rest_like[f"clavicle_{side}"]
        Ru = aim(f"upperarm_{side}", Rcl, f"sh_{side}", f"el_{side}")
        W[f"upperarm_twist_01_{side}"] = Ru @ W_rest_like[f"upperarm_twist_01_{side}"]
        Rl = aim(f"lowerarm_{side}", Ru, f"el_{side}", f"wr_{side}")
        W[f"lowerarm_twist_01_{side}"] = Rl @ W_rest_like[f"lowerarm_twist_01_{side}"]
        Rha = aim(f"hand_{side}", Rl, f"wr_{side}", f"fi_{side}")
        for f in ("thumb", "index", "middle", "ring", "pinky"):
            for i in (1, 2, 3):
                bn = f"{f}_0{i}_{side}"
                W[bn] = Rha @ W_rest_like[bn]
        Rt = aim(f"thigh_{side}", Rp, f"hip_{side}", f"knee_{side}")
        Rk = aim(f"calf_{side}", Rt, f"knee_{side}", f"ankle_{side}")
        Rf = aim(f"foot_{side}", Rk, f"ankle_{side}", f"toe_{side}")
        W[f"ball_{side}"] = Rf @ W_rest_like[f"ball_{side}"]
    return W, P
