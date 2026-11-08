"""Articulated rigid-body dynamics (XPBD) for the ragdoll and the physical-animation layer.

The body is 17 rigid segments (pelvis, abdomen, thorax, neck, head, upper arms, forearms, hands,
thighs, shanks, feet) with de Leva (1996) male mass fractions for an 80 kg man and the inertias of
their capsule / ellipsoid shapes. Segments are joined by ball sockets with anatomical swing-twist
limits (knees and elbows are hinges). Muscles are compliant angular drives that pull each joint
toward a target pose (the kinematic animation, or a reaction pose), with damping, an optional
torque cap and gravity compensation (the holding torque the nervous system supplies without
thinking). The ground is a plane with Coulomb friction; limbs collide with the trunk, the head and
each other.

Integration follows Mueller et al. 2020, "Detailed Rigid Body Simulation with Extended Position
Based Dynamics": many substeps, one constraint pass each, then a velocity pass for restitution,
friction and joint damping. Plain Python on tuples: at 3-vector size that is several times faster
than numpy, and it runs inside Blender's Python too.

Each body's orientation is the world rotation of its primary bone (W[bone] of the pose engine);
bones riding on a body (spine_02 on the abdomen, fingers on the hand, ...) keep their pose
relative to it from the reference animation. Two uses, see track() and Sim:
  - physical animation: the pelvis and legs follow the kinematic clip exactly; trunk, neck,
    head, arms and hands are dynamic and driven toward the clip, so weight, lag, follow-through
    and impacts come from the dynamics instead of being keyed;
  - ragdoll: every segment dynamic; drives hold a reaction pose with a tone that can fade.
"""
import math

import numpy as np

from .pose import axis_angle, quat_from_matrix, twist_about_y

FPS = 30
G = -9.81
DEG = math.pi / 180.0


# ------------------------------------------------------------------ tuple math
def vadd(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def vsub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def vmul(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def vdot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def vcross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def vlen(a):
    return math.sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2])


def qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw)


def qconj(q):
    return (q[0], -q[1], -q[2], -q[3])


def qnorm(q):
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


def qrot(q, v):
    """Rotate v by unit quaternion q."""
    w, x, y, z = q
    tx = 2.0 * (y * v[2] - z * v[1])
    ty = 2.0 * (z * v[0] - x * v[2])
    tz = 2.0 * (x * v[1] - y * v[0])
    return (v[0] + w * tx + y * tz - z * ty, v[1] + w * ty + z * tx - x * tz, v[2] + w * tz + x * ty - y * tx)


def qrotinv(q, v):
    return qrot((q[0], -q[1], -q[2], -q[3]), v)


def qexp(r):
    """Quaternion of rotation vector r (radians)."""
    a = vlen(r)
    if a < 1e-12:
        return qnorm((1.0, 0.5 * r[0], 0.5 * r[1], 0.5 * r[2]))
    s = math.sin(0.5 * a) / a
    return (math.cos(0.5 * a), r[0] * s, r[1] * s, r[2] * s)


def qlog(q):
    """Rotation vector of q (shortest way round)."""
    w, x, y, z = q
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    s = math.sqrt(x * x + y * y + z * z)
    if s < 1e-12:
        return (2.0 * x, 2.0 * y, 2.0 * z)
    k = 2.0 * math.atan2(s, w) / s
    return (x * k, y * k, z * k)


def qslerp(a, b, t):
    d = a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
    if d < 0.0:
        b, d = (-b[0], -b[1], -b[2], -b[3]), -d
    if d > 0.9995:
        return qnorm(tuple(a[i] + t * (b[i] - a[i]) for i in range(4)))
    th = math.acos(min(d, 1.0))
    s = math.sin(th)
    ka, kb = math.sin((1 - t) * th) / s, math.sin(t * th) / s
    return (ka * a[0] + kb * b[0], ka * a[1] + kb * b[1], ka * a[2] + kb * b[2], ka * a[3] + kb * b[3])


def quat(m):
    return tuple(float(c) for c in quat_from_matrix(np.asarray(m, float)))


def qmat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def tup(v):
    return (float(v[0]), float(v[1]), float(v[2]))


def clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def segment_closest(p1, q1, p2, q2):
    """Closest points between segments p1-q1 and p2-q2 (Ericson, Real-Time Collision Detection)."""
    d1, d2, r = vsub(q1, p1), vsub(q2, p2), vsub(p1, p2)
    a, e, f = vdot(d1, d1), vdot(d2, d2), vdot(d2, r)
    if a < 1e-12 and e < 1e-12:
        return p1, p2
    if a < 1e-12:
        s, t = 0.0, clamp(f / e, 0.0, 1.0)
    else:
        c = vdot(d1, r)
        if e < 1e-12:
            s, t = clamp(-c / a, 0.0, 1.0), 0.0
        else:
            b = vdot(d1, d2)
            den = a * e - b * b
            s = clamp((b * f - c * e) / den, 0.0, 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, clamp((b - c) / a, 0.0, 1.0)
    return vadd(p1, vmul(d1, s)), vadd(p2, vmul(d2, t))


# ------------------------------------------------------------------ bodies and constraints
class Body:
    """Rigid segment. x: centre of mass (world); q: world rotation of its primary bone (w, x, y, z).
    com: COM relative to the primary bone's head, in the bone frame. inertia: principal moments
    about the COM along the bone frame's axes. shapes: capsules (a, b, radius), ends relative to
    the COM in the bone frame."""

    def __init__(self, name, prim, mass, inertia, com, shapes=()):
        self.name, self.prim = name, prim
        self.mass = mass
        self.im0 = 1.0 / mass
        self.iI0 = tuple(1.0 / i for i in inertia)
        self.I0 = tuple(inertia)
        self.com = tup(com)
        self.shapes = list(shapes)
        self.kin = False
        self.x = self.xp = (0.0, 0.0, 0.0)
        self.q = self.qp = (1.0, 0.0, 0.0, 0.0)
        self.v = self.w = (0.0, 0.0, 0.0)
        self.tau = (0.0, 0.0, 0.0)
        self.props = []

    @property
    def im(self):
        return 0.0 if self.kin else self.im0

    def iI(self, v):
        """World inverse inertia times v."""
        if self.kin:
            return (0.0, 0.0, 0.0)
        l = qrotinv(self.q, v)
        i = self.iI0
        return qrot(self.q, (l[0] * i[0], l[1] * i[1], l[2] * i[2]))

    def Iw(self, v):
        l = qrotinv(self.q, v)
        i = self.I0
        return qrot(self.q, (l[0] * i[0], l[1] * i[1], l[2] * i[2]))

    def point(self, r):
        return vadd(self.x, qrot(self.q, r))

    def winv(self, r, n):
        """Generalised inverse mass for a correction along n at world offset r from the COM."""
        if self.kin:
            return 0.0
        rn = vcross(r, n)
        return self.im0 + vdot(rn, self.iI(rn))

    def push(self, r, p):
        """Positional impulse p (m kg) at world offset r from the COM."""
        if self.kin:
            return
        self.x = vadd(self.x, vmul(p, self.im0))
        self.turn(self.iI(vcross(r, p)))

    def turn(self, dw):
        """Rotate by the small world rotation vector dw."""
        q = self.q
        d = qmul((0.0, dw[0], dw[1], dw[2]), q)
        self.q = qnorm((q[0] + 0.5 * d[0], q[1] + 0.5 * d[1], q[2] + 0.5 * d[2], q[3] + 0.5 * d[3]))

    def kick(self, r, p):
        """Velocity impulse p (N s) at world offset r."""
        if self.kin:
            return
        self.v = vadd(self.v, vmul(p, self.im0))
        self.w = vadd(self.w, self.iI(vcross(r, p)))


def point_constraint(b1, r1l, b2, r2l, alpha=0.0):
    """Make body points coincide (local offsets from each COM). Returns the impulse magnitude."""
    r1, r2 = qrot(b1.q, r1l), qrot(b2.q, r2l)
    d = vsub(vadd(b2.x, r2), vadd(b1.x, r1))
    c = vlen(d)
    if c < 1e-10:
        return 0.0
    n = vmul(d, 1.0 / c)
    w = b1.winv(r1, n) + b2.winv(r2, n)
    if w + alpha <= 0.0:
        return 0.0
    lam = c / (w + alpha)
    p = vmul(n, lam)
    b1.push(r1, p)
    b2.push(r2, vmul(p, -1.0))
    return lam


def rot_constraint(b1, b2, e, alpha=0.0, cap=None):
    """Rotate b2 relative to b1 by the world rotation vector e (b1 takes its share the other way)."""
    th = vlen(e)
    if th < 1e-10:
        return 0.0
    n = vmul(e, 1.0 / th)
    i1, i2 = b1.iI(n), b2.iI(n)
    w = vdot(n, i1) + vdot(n, i2)
    if w + alpha <= 0.0:
        return 0.0
    lam = th / (w + alpha)
    if cap is not None and lam > cap:
        lam = cap
    if not b2.kin:
        b2.turn(vmul(i2, lam))
    if not b1.kin:
        b1.turn(vmul(i1, -lam))
    return lam


class Joint:
    """Ball socket between parent body P and child body C at the child's primary bone head.
    Limits: flexion and lateral swing (an ellipse, or a box for hinges and the wide-ranging hips
    and shoulders) and a twist range, in a joint frame whose y axis is the child's rest bone axis;
    angles are relative to the rest pose."""

    def __init__(self, name, P, C, ap, ac, LP, LC, lim, sgn_abd, sgn_tw, box):
        self.name, self.P, self.C = name, P, C
        self.ap, self.ac = ap, ac               # anchors relative to each COM, bone frames
        self.LP, self.LC = LP, LC
        (fmin, fmax), (amin, amax), (tmin, tmax) = lim
        self.lim = (fmin * DEG, fmax * DEG, amin * DEG, amax * DEG, tmin * DEG, tmax * DEG)
        self.sa, self.st = sgn_abd, sgn_tw
        self.box = box
        self.target = None                      # conj(q_P) q_C wanted (bone frames)
        self.k = 0.0                            # drive stiffness, N m / rad
        self.c = 0.0                            # drive damping, N m s / rad
        self.cap = None                         # drive torque cap, N m
        self.comp = 0.0                         # gravity compensation 0..1
        self.cp = 0.0                           # passive damping, N m s / rad
        self.isub = 1.0                         # inertia of the distal chain about the joint
        self.sub = []                           # bodies distal of the joint

    def angles(self):
        """(flexion, lateral, twist) in radians, anatomical signs."""
        qj = qmul(qmul(self.LP, qconj(self.P.q)), qmul(self.C.q, self.LC))
        return swing_twist(qj, self.sa, self.st)

    def set_angles(self, f, a, t):
        """Child orientation that puts the joint at the given anatomical angles."""
        qj = qmul(qexp((f, 0.0, a * self.sa)), (math.cos(0.5 * t * self.st), 0.0, math.sin(0.5 * t * self.st), 0.0))
        return qmul(qmul(self.P.q, qconj(self.LP)), qmul(qj, qconj(self.LC)))


def swing_twist(qj, sa=1.0, st=1.0):
    w, x, y, z = qj
    if w < 0.0:
        w, x, y, z = -w, -x, -y, -z
    tn = math.sqrt(w * w + y * y)
    if tn < 1e-9:
        tw = (1.0, 0.0, 0.0, 0.0)
        t = 0.0
    else:
        tw = (w / tn, 0.0, y / tn, 0.0)
        t = 2.0 * math.atan2(tw[2], tw[0])
    s = qlog(qmul((w, x, y, z), qconj(tw)))
    return s[0], s[2] * sa, t * st


def clamp_angles(f, a, t, lim, box):
    fmin, fmax, amin, amax, tmin, tmax = lim
    if box:
        f2, a2 = clamp(f, fmin, fmax), clamp(a, amin, amax)
    else:
        ef = fmax if f >= 0.0 else -fmin
        ea = amax if a >= 0.0 else -amin
        e = (f / ef) ** 2 + (a / ea) ** 2
        f2, a2 = (f, a) if e <= 1.0 else (f / math.sqrt(e), a / math.sqrt(e))
    t2 = clamp(t, tmin, tmax)
    return f2, a2, t2, (f2 != f or a2 != a or t2 != t)


# ------------------------------------------------------------------ the human body
TOTAL_MASS = 80.0
# body: (primary bone, other bones it carries, mass kg). de Leva 1996 (male) segment fractions of
# 80 kg: lower/middle/upper trunk 11.17/16.33/15.96 %, head+neck 6.94 %, upper arm 2.71 %,
# forearm 1.62 %, hand 0.61 %, thigh 14.16 %, shank 4.33 %, foot 1.37 % (+0.45 kg sneaker).
SEGMENTS = {
    "pelvis": ("pelvis", (), 8.94),
    "abdomen": ("spine_01", ("spine_02",), 13.06),
    "thorax": ("spine_03", ("spine_04",), 12.77),
    "neck": ("neck_01", ("neck_02",), 1.05),
    "head": ("head", (), 4.50),
}
for _s in ("l", "r"):
    SEGMENTS.update({f"upperarm_{_s}": (f"upperarm_{_s}", (), 2.17), f"lowerarm_{_s}": (f"lowerarm_{_s}", (), 1.30),
                     f"hand_{_s}": (f"hand_{_s}", (), 0.49), f"thigh_{_s}": (f"thigh_{_s}", (), 11.33),
                     f"calf_{_s}": (f"calf_{_s}", (), 3.46), f"foot_{_s}": (f"foot_{_s}", (), 1.55)})

# joint: (parent, child, kind)
JOINTS = [("lumbar", "pelvis", "abdomen", "lumbar"), ("thoracic", "abdomen", "thorax", "thoracic"),
          ("neck", "thorax", "neck", "neck"), ("skull", "neck", "head", "skull")]
for _s in ("l", "r"):
    JOINTS += [(f"shoulder_{_s}", "thorax", f"upperarm_{_s}", "shoulder"),
               (f"elbow_{_s}", f"upperarm_{_s}", f"lowerarm_{_s}", "elbow"),
               (f"wrist_{_s}", f"lowerarm_{_s}", f"hand_{_s}", "wrist"),
               (f"hip_{_s}", "pelvis", f"thigh_{_s}", "hip"),
               (f"knee_{_s}", f"thigh_{_s}", f"calf_{_s}", "knee"),
               (f"ankle_{_s}", f"calf_{_s}", f"foot_{_s}", "ankle")]

# anatomical ranges relative to the rest pose (arms in the 48 deg A-pose), degrees:
# (flexion min, max), (lateral min, max), (twist min, max), box (swing clamped per axis)
LIMITS = {
    "lumbar": ((-22, 40), (-22, 22), (-12, 12), False),
    "thoracic": ((-22, 35), (-22, 22), (-28, 28), False),
    "neck": ((-38, 45), (-30, 30), (-38, 38), False),
    "skull": ((-35, 32), (-22, 22), (-45, 45), False),
    "shoulder": ((-75, 170), (-95, 125), (-90, 120), True),
    "elbow": ((-4, 150), (-6, 6), (-12, 12), True),
    "wrist": ((-75, 80), (-35, 25), (-90, 90), False),
    "hip": ((-28, 128), (-32, 48), (-48, 48), True),
    "knee": ((-3, 152), (-3, 3), (-12, 12), True),
    "ankle": ((-55, 30), (-28, 28), (-18, 18), False),
}

ARM_BODIES = ("upperarm_l", "lowerarm_l", "hand_l", "upperarm_r", "lowerarm_r", "hand_r")
LEG_BODIES = ("thigh_l", "calf_l", "foot_l", "thigh_r", "calf_r", "foot_r")
TRUNK_BODIES = ("pelvis", "abdomen", "thorax")
# limb pairs that may collide (adjacent bodies never do)
COLLIDE = ([(a, t) for a in ARM_BODIES for t in ("pelvis", "abdomen", "thorax", "head", "neck")
            if not (a.startswith("upperarm") and t == "thorax")]
           + [(a, t) for a in ARM_BODIES for t in ("thigh_l", "thigh_r")]
           + [(a, b) for a in ARM_BODIES[:3] for b in ARM_BODIES[3:]]
           + [("thigh_l", "thigh_r"), ("calf_l", "calf_r"), ("foot_l", "foot_r"), ("calf_l", "foot_r"),
              ("foot_l", "calf_r"), ("thigh_l", "calf_r"), ("calf_l", "thigh_r")])


def _shapes(s):
    """Collision capsules (world rest coordinates). Kept a little inside the skin so limbs can
    brush the body without being shoved off their animated paths."""
    h, t = s.head, s.tail
    P = lambda *v: np.array(v, float)
    sh = {
        "pelvis": [(P(-0.075, 0.015, 0.915), P(0.075, 0.015, 0.915), 0.090)],
        "abdomen": [(P(-0.05, 0.02, 1.09), P(0.05, 0.02, 1.09), 0.095)],
        "thorax": [(P(-0.08, 0.03, 1.27), P(0.08, 0.03, 1.27), 0.11), (P(-0.12, 0.025, 1.40), P(0.12, 0.025, 1.40), 0.07)],
        "neck": [(h["neck_01"], h["head"], 0.048)],
        "head": [(P(0.0, -0.005, 1.67), P(0.0, 0.0, 1.735), 0.092)],
    }
    for sd in ("l", "r"):
        hy = s.A[f"hand_{sd}"][:, 1]
        sh[f"upperarm_{sd}"] = [(h[f"upperarm_{sd}"], h[f"lowerarm_{sd}"], 0.043)]
        sh[f"lowerarm_{sd}"] = [(h[f"lowerarm_{sd}"], h[f"hand_{sd}"], 0.034)]
        sh[f"hand_{sd}"] = [(h[f"hand_{sd}"] + 0.025 * hy, h[f"hand_{sd}"] + 0.10 * hy, 0.026)]
        sh[f"thigh_{sd}"] = [(h[f"thigh_{sd}"], h[f"calf_{sd}"], 0.068)]
        sh[f"calf_{sd}"] = [(h[f"calf_{sd}"], h[f"foot_{sd}"], 0.046)]
        ank, toe = h[f"foot_{sd}"], t[f"ball_{sd}"]
        fwd = (toe - ank) * np.array([1.0, 1.0, 0.0])
        fwd /= np.linalg.norm(fwd)
        r = 0.038                                      # outsole 12 mm under the bare sole (z = 0 at rest)
        sh[f"foot_{sd}"] = [(P(ank[0], ank[1], r - 0.012) - 0.045 * fwd, P(toe[0], toe[1], r - 0.012) - 0.01 * fwd, r)]
    return sh


def _com_and_inertia(s, name, prim, mass, shapes):
    """COM (world, rest) and principal moments in the primary bone frame."""
    h, t = s.head, s.tail
    if name in ("pelvis", "abdomen", "thorax", "head"):
        semi = {"pelvis": (0.16, 0.10, 0.11), "abdomen": (0.15, 0.09, 0.11), "thorax": (0.18, 0.14, 0.12),
                "head": (0.08, 0.11, 0.10)}[name]
        a, b, c = semi
        com = {"pelvis": np.array([0.0, 0.012, 0.93]), "abdomen": np.array([0.0, 0.02, 1.09]),
               "thorax": np.array([0.0, 0.03, 1.31]), "head": np.array([0.0, -0.01, 1.70])}[name]
        return com, (mass * (b * b + c * c) / 5, mass * (a * a + c * c) / 5, mass * (a * a + b * b) / 5)
    a, b, r = shapes[0]
    if name.startswith("foot"):
        com = a + 0.45 * (b - a)
        L = np.linalg.norm(b - a) + r
    else:
        frac = {"neck": 0.5, "upperarm": 0.47, "lowerarm": 0.43, "hand": 0.35, "thigh": 0.41, "calf": 0.44}[name.split("_")[0]]
        j0, j1 = h[prim], (t[prim] if name != "neck" else h["head"])
        com = j0 + frac * (j1 - j0)
        L = np.linalg.norm(j1 - j0)
    it = mass * (3 * r * r + L * L) / 12
    return com, (it, 0.5 * mass * r * r, it)


def _subtree_inertia(bodies, joints, j):
    """Mean transverse inertia of the chain distal of joint j about the joint (rest pose)."""
    p = np.array(j.C.point(j.ac))
    tot = 0.0
    for b in j.sub:
        d = np.array(b.x) - p
        tot += b.mass * float(d @ d) + (b.I0[0] + b.I0[2]) / 2
    return tot


def build_human(skel):
    """Bodies (dict), joints (list) and bone -> body map for the rig in skel, posed at rest."""
    s = skel
    shapes = _shapes(s)
    bodies, bone_body = {}, {}
    for name, (prim, extra, mass) in SEGMENTS.items():
        com_w, inertia = _com_and_inertia(s, name, prim, mass, shapes[name])
        A = s.A[prim]
        com_l = A.T @ (com_w - s.head[prim])
        caps = [(tup(A.T @ (a - com_w)), tup(A.T @ (b - com_w)), r) for a, b, r in shapes[name]]
        b = Body(name, prim, mass, inertia, com_l, caps)
        b.x, b.q = tup(com_w), quat(A)
        b.xp, b.qp = b.x, b.q
        bodies[name] = b
        for bn in (prim,) + extra:
            bone_body[bn] = name
    for bn in s.order:                      # everything else rides on its nearest ancestor's body
        if bn in bone_body or bn == "root":
            continue
        a = s.parent[bn]
        while a not in bone_body and a is not None:
            a = s.parent[a]
        if a is not None:
            bone_body[bn] = bone_body[a]
    joints = []
    for name, pn, cn, kind in JOINTS:
        P, C = bodies[pn], bodies[cn]
        anchor = s.head[C.prim]
        ap = tup(s.A[P.prim].T @ (anchor - np.array(P.x)))
        ac = tup(s.A[C.prim].T @ (anchor - np.array(C.x)))
        side = name[-1] if name[-2] == "_" else ""
        y0 = s.A[C.prim][:, 1]
        dflex = {"knee": np.array([0.0, 1.0, 0.0]), "ankle": np.array([0.0, 0.0, 1.0]),
                 "wrist": -s.A[C.prim][:, 2]}.get(kind, np.array([0.0, -1.0, 0.0]))
        ef = np.cross(y0, dflex)
        ef /= np.linalg.norm(ef)
        e3 = np.cross(ef, y0)
        Jf = np.stack([ef, y0, e3], 1)
        qJ = quat(Jf)
        LP = qmul(qconj(qJ), quat(s.A[P.prim]))
        LC = qmul(qconj(quat(s.A[C.prim])), qJ)
        if kind == "wrist":
            lateral = s.A[C.prim][:, 0] * (1.0 if side == "l" else -1.0)     # radial deviation
        else:
            lateral = np.array([-1.0 if side == "r" else 1.0, 0.0, 0.0])
        sa = 1.0 if float(np.dot(-ef, lateral)) >= 0 else -1.0
        st = -1.0 if side == "r" else 1.0
        lim = LIMITS[kind]
        joints.append(Joint(name, P, C, ap, ac, LP, LC, lim[:3], sa, st, lim[3]))
    children = {}
    for j in joints:
        children.setdefault(j.P.name, []).append(j.C.name)

    def distal(n):
        out = [bodies[n]]
        for c in children.get(n, []):
            out += distal(c)
        return out

    for j in joints:
        j.sub = distal(j.C.name)
        j.isub = _subtree_inertia(bodies, joints, j)
        j.kind = [k for nm, _, _, k in JOINTS if nm == j.name][0]
    return bodies, joints, bone_body


# ------------------------------------------------------------------ simulation
class Sim:
    """Articulated body following a reference animation.

    ref frames (pose-engine Frames: W, H per bone) supply, per frame: the pose of kinematic
    bodies, the drive targets (relative rotations of the joints) and the internal pose of the
    bones riding on each body. Call step(ref_next) once per frame, frame(ref) to read the state
    back as a Frame."""

    def __init__(self, skel, ref, prev=None, substeps=24, fps=FPS, nxt=None):
        """prev (the frame before ref) or nxt (the one after) gives the starting velocities."""
        self.s = skel
        self.bodies, self.joints, self.bone_body = build_human(skel)
        self.B = list(self.bodies.values())
        self.J = {j.name: j for j in self.joints}
        self.sub, self.fps = substeps, fps
        self.h = 1.0 / (fps * substeps)
        self.friction = (0.9, 0.7)            # static, dynamic (cotton / denim on asphalt)
        self.restitution = 0.12
        self.pins, self.welds, self.extra, self.goals = {}, {}, [], {}
        self.nocollide = set()
        self.collide = True
        self.contacts_on = True
        self.ref = ref
        self.pairs = [(self.bodies[a], self.bodies[b]) for a, b in COLLIDE]
        self.time = 0.0
        for j in self.joints:
            j.cp = 1.6 * j.isub                 # passive tissue damping: ~0.6 s decay of joint motion
        kin = self._kin_pose(ref)
        for b in self.B:
            b.x, b.q = kin[b.name]
            b.xp, b.qp = b.x, b.q
        other, sgn = (prev, 1.0) if prev is not None else (nxt, -1.0)
        if other is not None:
            kp = self._kin_pose(other)
            for b in self.B:
                x0, q0 = kp[b.name]
                b.v = vmul(vsub(b.x, x0), fps * sgn)
                b.w = vmul(qlog(qmul(b.q, qconj(q0))), fps * sgn)
        self._targets(ref)

    # ---- reference handling
    def _kin_pose(self, ref):
        out = {}
        for b in self.B:
            Wp = ref.W[b.prim]
            out[b.name] = (tup(ref.H[b.prim] + Wp @ np.array(b.com)), quat(Wp))
        for p in self.extra:
            if p.ref_pose is not None:
                out[p.name] = p.ref_pose(ref)
        return out

    def _targets(self, ref):
        """Drive targets and joint anchors (the parent anchor follows the riding bones, e.g. the
        clavicle carrying the shoulder) from a reference frame."""
        for j in self.joints:
            WP, WC = ref.W[j.P.prim], ref.W[j.C.prim]
            j.target = qmul(qconj(quat(WP)), quat(WC))
            if j.name in self.goals:
                g, w = self.goals[j.name]
                j.target = qslerp(j.target, g, w)
            j.ap = tup(WP.T @ (ref.H[j.C.prim] - ref.H[j.P.prim]) - np.array(j.P.com))

    def set_kinematic(self, names, on=True):
        for n in names:
            self.bodies[n].kin = on

    def set_tone(self, joints, freq=0.0, zeta=0.8, comp=1.0, cap=None, k=None, c=None):
        """Drive the named joints (or kinds: 'shoulder', 'knee', ...) toward the reference with a
        natural frequency freq (Hz) of the distal chain, damping ratio zeta, gravity compensation
        comp (0..1) and an optional torque cap (N m). Support joints (a stance ankle carrying the
        whole body) need k (N m / rad) and c (N m s / rad) given directly."""
        if isinstance(joints, str):
            joints = (joints,)
        for j in self.joints:
            if j.name in joints or j.kind in joints:
                j.k = k if k is not None else j.isub * (2 * math.pi * freq) ** 2
                j.c = c if c is not None else (2 * zeta * math.sqrt(j.k * j.isub) if j.k > 0 else 0.0)
                j.comp = comp
                j.cap = cap

    def limp(self, joints):
        """No muscle drive at all (unconscious): passive tissue damping and the limits only."""
        self.set_tone(joints, 0.0, comp=0.0)

    def goal(self, joint, angles=None, weight=1.0):
        """Override a joint's drive target with anatomical angles (flexion, lateral, twist; degrees,
        relative to the rest pose), blended with the reference by weight. angles=None clears it."""
        j = self.J[joint]
        if angles is None:
            self.goals.pop(joint, None)
            return
        f, a, t = (v * DEG for v in angles)
        qj = qmul(qexp((f, 0.0, a * j.sa)), (math.cos(0.5 * t * j.st), 0.0, math.sin(0.5 * t * j.st), 0.0))
        self.goals[joint] = (qmul(qconj(j.LP), qmul(qj, qconj(j.LC))), weight)

    def impulse(self, body, point, J):
        b = self.bodies[body] if isinstance(body, str) else body
        b.kick(vsub(tup(point), b.x), tup(J))

    def pin(self, key, body, world_point, target=None, alpha=0.0):
        """Hold the body point currently at world_point on target (default: where it is)."""
        b = self.bodies[body] if isinstance(body, str) else body
        r = qrotinv(b.q, vsub(tup(world_point), b.x))
        self.pins[key] = (b, r, tup(target if target is not None else world_point), alpha)

    def weld(self, key, a, b, alpha=0.0, rel=None):
        """Lock body b to body a (current relative pose unless rel = (offset in a, rotation q))."""
        A = self.bodies[a] if isinstance(a, str) else a
        Bb = self.bodies[b] if isinstance(b, str) else b
        if rel is None:
            off = qrotinv(A.q, vsub(Bb.x, A.x))
            qr = qmul(qconj(A.q), Bb.q)
        else:
            off, qr = rel
        self.welds[key] = (A, Bb, tup(off), qr, alpha)

    def add_prop(self, name, mass, inertia, ref_pose):
        """Extra rigid body (e.g. the pistol). ref_pose(ref) -> (x, q) places it while kinematic."""
        b = Body(name, name, mass, inertia, (0.0, 0.0, 0.0))
        b.ref_pose = ref_pose
        b.kin = True
        self.extra.append(b)
        self.bodies[name] = b
        x, q = ref_pose(self.ref)
        b.x = b.xp = x
        b.q = b.qp = q
        return b

    # ---- stepping
    def step(self, ref_next):
        a, b = self._kin_pose(self.ref), self._kin_pose(ref_next)
        ta = {j.name: j.target for j in self.joints}
        self._targets(ref_next)
        tb = {j.name: j.target for j in self.joints}
        allb = self.B + self.extra
        for k in range(self.sub):
            u = (k + 1) / self.sub
            for j in self.joints:
                j.target = qslerp(ta[j.name], tb[j.name], u)
            self._substep(allb, a, b, u)
        self.ref = ref_next
        self.time += 1.0 / self.fps

    def _substep(self, allb, ka, kb, u):
        h = self.h
        h2 = h * h
        self._gravity_comp()
        for b in allb:
            b.xp, b.qp = b.x, b.q
            if b.kin:
                xa, qa = ka[b.name]
                xb, qb = kb[b.name]
                b.x = vadd(xa, vmul(vsub(xb, xa), u))
                b.q = qslerp(qa, qb, u)
                continue
            b.v = (b.v[0], b.v[1], b.v[2] + G * h)
            b.x = vadd(b.x, vmul(b.v, h))
            # angular: torques, gyroscopic term
            w = b.w
            t = vsub(b.tau, vcross(w, b.Iw(w)))
            w = vadd(w, vmul(b.iI(t), h))
            b.w = w
            b.turn(vmul(w, h))
        for b in allb:
            b.tau = (0.0, 0.0, 0.0)
        # joints: socket, drive, limits
        for j in self.joints:
            P, C = j.P, j.C
            if P.kin and C.kin:
                continue
            point_constraint(P, j.ap, C, j.ac)
            if j.k > 0.0 and j.target is not None:
                want = qmul(P.q, j.target)
                e = qlog(qmul(want, qconj(C.q)))
                rot_constraint(P, C, e, 1.0 / (j.k * h2), None if j.cap is None else j.cap * h2)
            f, a, t = j.angles()
            f2, a2, t2, out = clamp_angles(f, a, t, j.lim, j.box)
            if out:
                e = qlog(qmul(j.set_angles(f2, a2, t2), qconj(C.q)))
                rot_constraint(P, C, e)
        for key, (A, Bb, off, qr, alpha) in self.welds.items():
            point_constraint(A, off, Bb, (0.0, 0.0, 0.0), alpha / h2)
            e = qlog(qmul(qmul(A.q, qr), qconj(Bb.q)))
            rot_constraint(A, Bb, e, alpha / h2)
        for key, (b, r, target, alpha) in self.pins.items():
            if b.kin:
                continue
            rw = qrot(b.q, r)
            d = vsub(target, vadd(b.x, rw))
            c = vlen(d)
            if c > 1e-10:
                n = vmul(d, 1.0 / c)
                lam = c / (b.winv(rw, n) + alpha / h2)
                b.push(rw, vmul(n, lam))
        contacts = self._ground(allb, h2) if self.contacts_on else []
        if self.collide:
            self._self_collide()
        # velocities from positions
        for b in allb:
            if b.kin:
                b.v = vmul(vsub(b.x, b.xp), 1.0 / h)
                b.w = vmul(qlog(qmul(b.q, qconj(b.qp))), 1.0 / h)
                continue
            b.v = vmul(vsub(b.x, b.xp), 1.0 / h)
            dq = qmul(b.q, qconj(b.qp))
            sgn = 2.0 / h if dq[0] >= 0 else -2.0 / h
            b.w = (dq[1] * sgn, dq[2] * sgn, dq[3] * sgn)
        self._contact_velocities(contacts, h)
        self._joint_damping(h)

    def _gravity_comp(self):
        for j in self.joints:
            if j.comp <= 0.0 or j.C.kin:
                continue
            p = j.C.point(j.ac)
            tx = ty = 0.0
            for b in j.sub:
                for bb in [b] + b.props:
                    d = vsub(bb.x, p)
                    # (d x m g) with g = (0, 0, G)
                    tx += d[1] * bb.mass * G
                    ty -= d[0] * bb.mass * G
            t = (-tx * j.comp, -ty * j.comp, 0.0)
            j.C.tau = vadd(j.C.tau, t)
            j.P.tau = vsub(j.P.tau, t)

    def _ground(self, allb, h2):
        mu_s = self.friction[0]
        out = []
        for b in self.B:
            if b.kin:
                continue
            for a, c, r in b.shapes:
                for end in (a, c):
                    rw = qrot(b.q, end)
                    z = b.x[2] + rw[2] - r
                    if z >= 0.0:
                        continue
                    rc = (rw[0], rw[1], rw[2] - r)                  # lowest point of the end sphere
                    rl = qrotinv(b.q, rc)                           # ... as a material point
                    vpre = b.v[2] + b.w[0] * rc[1] - b.w[1] * rc[0]
                    lam = -z / b.winv(rc, (0.0, 0.0, 1.0))
                    b.push(rc, (0.0, 0.0, lam))
                    # static friction: undo the contact point's slide over this substep while the
                    # tangential correction stays inside the friction cone
                    rc = qrot(b.q, rl)
                    before = vadd(b.xp, qrot(b.qp, rl))
                    dt = (b.x[0] + rc[0] - before[0], b.x[1] + rc[1] - before[1], 0.0)
                    lt = vlen(dt)
                    if lt > 1e-12:
                        tn = vmul(dt, 1.0 / lt)
                        lt_lam = lt / b.winv(rc, tn)
                        if lt_lam < mu_s * lam:
                            b.push(rc, vmul(tn, -lt_lam))
                    out.append((b, rl, lam, vpre))
        return out

    def _contact_velocities(self, contacts, h):
        mu_d = self.friction[1]
        e = self.restitution
        for b, rl, lam, vpre in contacts:
            r = qrot(b.q, rl)
            v = vadd(b.v, vcross(b.w, r))
            dv = [0.0, 0.0, 0.0]
            lt = math.hypot(v[0], v[1])
            if lt > 1e-9:
                fr = min(mu_d * lam / h, lt)
                dv[0], dv[1] = -v[0] / lt * fr, -v[1] / lt * fr
            # normal: a little bounce for hard impacts, otherwise inelastic; this also removes the
            # velocity the position projection injected
            dv[2] = (-e * vpre if vpre < -0.5 else 0.0) - v[2]
            dl = vlen(dv)
            if dl < 1e-12:
                continue
            n = vmul(tuple(dv), 1.0 / dl)
            b.kick(r, vmul(n, dl / b.winv(r, n)))

    def _self_collide(self):
        for A, Bb in self.pairs:
            if A.kin and Bb.kin:
                continue
            if (A.name, Bb.name) in self.nocollide or (Bb.name, A.name) in self.nocollide:
                continue
            for a1, b1, r1 in A.shapes:
                p1, q1 = A.point(a1), A.point(b1)
                for a2, b2, r2 in Bb.shapes:
                    p2, q2 = Bb.point(a2), Bb.point(b2)
                    c1, c2 = segment_closest(p1, q1, p2, q2)
                    d = vsub(c2, c1)
                    dist = vlen(d)
                    pen = r1 + r2 - dist
                    if pen <= 0.0 or dist < 1e-9:
                        continue
                    n = vmul(d, 1.0 / dist)
                    ra = vsub(vadd(c1, vmul(n, r1)), A.x)
                    rb = vsub(vsub(c2, vmul(n, r2)), Bb.x)
                    w = A.winv(ra, n) + Bb.winv(rb, n)
                    if w <= 0.0:
                        continue
                    lam = pen / w
                    A.push(ra, vmul(n, -lam))
                    Bb.push(rb, vmul(n, lam))

    def _joint_damping(self, h):
        for j in self.joints:
            P, C = j.P, j.C
            if P.kin and C.kin:
                continue
            c = j.c + j.cp
            if c <= 0.0:
                continue
            wr = vsub(C.w, P.w)
            m = vlen(wr)
            if m < 1e-9:
                continue
            n = vmul(wr, 1.0 / m)
            i1, i2 = P.iI(n), C.iI(n)
            w = vdot(n, i1) + vdot(n, i2)
            if w <= 0.0:
                continue
            p = min(c * h * m, m / w)
            C.w = vsub(C.w, vmul(i2, p))
            P.w = vadd(P.w, vmul(i1, p))

    # ---- read back
    def com(self):
        m = sum(b.mass for b in self.B + self.extra)
        c = (0.0, 0.0, 0.0)
        for b in self.B + self.extra:
            c = vadd(c, vmul(b.x, b.mass))
        return vmul(c, 1.0 / m)

    def frame(self, ref, extra=None):
        """Current state as a pose-engine Frame: each bone keeps its pose relative to its body
        from ref; the forearm twist bone is re-split from the simulated wrist."""
        from .motions import Frame
        s = self.s
        W = {}
        R = {n: qmat(b.q) for n, b in self.bodies.items()}
        for bn in s.order:
            if bn == "root":
                W[bn] = ref.W["root"]
                continue
            body = self.bodies[self.bone_body[bn]]
            W[bn] = R[body.name] @ ref.W[body.prim].T @ ref.W[bn]
        pel = self.bodies["pelvis"]
        h_pel = np.array(pel.x) - R["pelvis"] @ np.array(pel.com)
        root_head = h_pel - W["root"] @ s.offset("pelvis")
        root_loc = root_head - s.head["root"]
        B = {}
        for bn in s.order:
            p = s.parent[bn]
            B[bn] = s.A[bn].T @ W[bn] if p is None else (W[p] @ (s.A[p].T @ s.A[bn])).T @ W[bn]
        for sd in ("l", "r"):
            B[f"lowerarm_twist_01_{sd}"] = axis_angle([0, 1, 0], 0.5 * twist_about_y(B[f"hand_{sd}"]))
        W2, H2 = s.fk(B, root_loc)
        if min(H2["hand_l"][2], H2["hand_r"][2]) < 0.25:
            clamp_fingers(s, W2, H2)                # updates W2, H2 in place
        quats, loc = s.basis(W2, root_loc)
        return Frame(quats, loc, W2, H2, dict(extra or {}))


FINGERS = ("thumb", "index", "middle", "ring", "pinky")


def clamp_fingers(skel, W, H, floor=0.008):
    """Lift finger segments that would pass through the floor (fingers have no collision
    shapes): each offending segment is pitched up about its horizontal axis and its children
    follow rigidly. Returns True if anything changed; W and H are updated in place."""
    changed = False
    for side in ("l", "r"):
        for f in FINGERS:
            chain = [f"{f}_0{i}_{side}" for i in (1, 2, 3)]
            for k, b in enumerate(chain):
                if k > 0:
                    p = chain[k - 1]
                    H[b] = H[p] + W[p] @ skel.offset(b)
                d = W[b][:, 1]
                if H[b][2] + d[2] * skel.length[b] >= floor:
                    continue
                want = np.clip((floor - H[b][2]) / skel.length[b], -1.0, 1.0)
                delta = np.degrees(np.arcsin(want) - np.arcsin(np.clip(d[2], -1.0, 1.0)))
                axis = np.cross(d, [0.0, 0.0, 1.0])
                if np.linalg.norm(axis) < 1e-6:
                    continue
                R = axis_angle(axis / np.linalg.norm(axis), delta)
                for c in chain[k:]:
                    W[c] = R @ W[c]
                changed = True
    return changed


def track(skel, frames, setup=None, per_frame=None, substeps=24):
    """Physical animation: run the kinematic clip through the body. setup(sim) configures tone and
    kinematic bodies (default: pelvis and legs kinematic, the rest driven by DEFAULT_TONE);
    per_frame(sim, i) may change anything before frame i is simulated. Returns new Frames with
    each source frame's extra data."""
    sim = Sim(skel, frames[0], substeps=substeps, nxt=frames[1] if len(frames) > 1 else None)
    sim.set_kinematic(("pelvis",) + LEG_BODIES)
    for joints, args in DEFAULT_TONE:
        sim.set_tone(joints, *args)
    if setup:
        setup(sim)
    out = [sim.frame(frames[0], frames[0].extra)]
    for i in range(1, len(frames)):
        if per_frame:
            per_frame(sim, i)
        sim.step(frames[i])
        out.append(sim.frame(frames[i], frames[i].extra))
    return out, sim


# joint kinds -> (Hz, damping ratio, gravity compensation): a relaxed but upright adult
DEFAULT_TONE = [
    (("lumbar", "thoracic"), (5.0, 0.9, 1.0)),
    (("neck", "skull"), (4.0, 0.75, 1.0)),
    (("shoulder",), (2.6, 0.55, 1.0)),
    (("elbow",), (3.0, 0.55, 1.0)),
    (("wrist",), (4.0, 0.6, 1.0)),
    (("hip", "knee", "ankle"), (8.0, 0.9, 1.0)),
]
