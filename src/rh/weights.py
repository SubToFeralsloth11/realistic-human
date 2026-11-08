"""Skin weights from anatomy instead of heat diffusion.

1. Region membership: every sculpting leaf carries a bone tag; leaves are grouped into body
   regions (torso, neck, head, arm, hand, leg, foot per side). A vertex's distance to each region
   is the hard union of that region's leaves; regions blend with a soft-min over those distances,
   which puts the blend exactly where the shapes were fused (armpit, hip, wrist, ankle, neck).
2. Inside a region, bones partition the vertex along the chain: spine by height, arm/leg by
   projection on the limb axis around the elbow/knee, hand bones by capsule distance to each
   phalanx, foot/toes by projection past the ball joint, jaw by a head-local region.
3. Top four influences, mesh-graph smoothing, renormalised.
"""
import numpy as np
import scipy.sparse as sp

from . import head as headmod
from .sdf import F, group_distance
from .skeleton import J


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def hats(x, centers):
    """Piecewise-linear partition of unity over sorted centres; returns (n, len(centers))."""
    c = np.asarray(centers, float)
    out = np.zeros((len(x), len(c)))
    xc = np.clip(x, c[0], c[-1])
    idx = np.clip(np.searchsorted(c, xc) - 1, 0, len(c) - 2)
    t = (xc - c[idx]) / (c[idx + 1] - c[idx])
    out[np.arange(len(x)), idx] = 1 - t
    out[np.arange(len(x)), idx + 1] += t
    return out


REGION_OF = {
    "pelvis": "torso", "spine_01": "torso", "spine_02": "torso", "spine_03": "torso", "spine_04": "torso",
    "clavicle_l": "torso", "clavicle_r": "torso", "neck_01": "neck", "head": "head", "jaw": "head",
    "upperarm_l": "arm_l", "upperarm_r": "arm_r", "thigh_l": "leg_l", "thigh_r": "leg_r", "calf_l": "leg_l",
    "calf_r": "leg_r", "foot_l": "foot_l", "foot_r": "foot_r", "ball_l": "foot_l", "ball_r": "foot_r",
}


def region_of(bone):
    if bone is None:
        return None
    if bone in REGION_OF:
        return REGION_OF[bone]
    side = bone[-1]
    if bone.startswith(("hand_", "thumb_", "index_", "middle_", "ring_", "pinky_")):
        return "hand_" + side
    return None


def _seg_dist(P, a, b):
    ab = b - a
    t = np.clip(((P - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(P - (a + t[:, None] * ab), axis=1), t


def compute(verts, faces, sizes, prog, bones, sigma=0.010, smooth_iters=12):
    """Returns (bone_names list, idx uint16 (n,4), w float32 (n,4))."""
    P = np.asarray(verts, F)
    n = len(P)
    names = list(bones.keys())
    col = {b: i for i, b in enumerate(names)}
    regions = {}
    for leaf in prog.leaves():
        r = region_of(leaf.bone)
        if r:
            regions.setdefault(r, []).append(leaf)
    rnames = sorted(regions)
    D = np.stack([group_distance(regions[r], P, band=0.08) for r in rnames], 1).astype(np.float64)
    RW = np.exp(-(D - D.min(axis=1, keepdims=True)) / sigma)
    RW /= RW.sum(axis=1, keepdims=True)
    W = np.zeros((n, len(names)))

    def add(region, bone_w):
        k = rnames.index(region)
        for b, w in bone_w.items():
            W[:, col[b]] += RW[:, k] * w

    z = P[:, 2].astype(np.float64)
    # torso: spine partition by height, clavicles over the shoulder tops
    sp_names = ["pelvis", "spine_01", "spine_02", "spine_03", "spine_04"]
    H = hats(z, [0.92, 1.075, 1.175, 1.29, 1.415])
    clav = smoothstep(0.07, 0.15, np.abs(P[:, 0])) * smoothstep(1.36, 1.44, z)
    tw = {b: H[:, i] * (1 - clav) for i, b in enumerate(sp_names)}
    tw["clavicle_l"] = clav * (P[:, 0] > 0)
    tw["clavicle_r"] = clav * (P[:, 0] <= 0)
    add("torso", tw)
    Hn = hats(z, [1.50, 1.60])
    add("neck", {"neck_01": Hn[:, 0], "neck_02": Hn[:, 1]})
    # head and jaw
    q = P - headmod.O
    jaw = (1 - smoothstep(-0.0745, -0.0690, q[:, 2])) * (1 - smoothstep(0.040, 0.062, q[:, 1]))
    add("head", {"head": 1 - jaw, "jaw": jaw})
    # eyelids: skin on the lid shell between the margins and the folds follows the lid bones
    hi = col["head"]
    for side in ("l", "r"):
        if f"lid_upper_{side}" not in col:
            continue
        v = P.astype(np.float64) - headmod.LANDMARKS[f"eye_{side}"]
        r = np.linalg.norm(v, axis=1)
        R = headmod.LID_R
        shell = (1 - smoothstep(R + 0.0025, R + 0.0065, r)) * smoothstep(R - 0.003, R - 0.001, r)
        front = 1 - smoothstep(-0.002, 0.004, v[:, 1])
        lat = 1 - smoothstep(0.013, 0.018, np.abs(v[:, 0]))
        elev = np.degrees(np.arcsin(np.clip(v[:, 2] / np.maximum(r, 1e-9), -1, 1)))
        upper = smoothstep(8, 13, elev) * (1 - smoothstep(26, 44, elev))
        lower = (1 - smoothstep(-22, -17, elev)) * smoothstep(-42, -30, elev)
        for bone, w in ((f"lid_upper_{side}", upper), (f"lid_lower_{side}", 0.9 * lower)):
            w = w * shell * front * lat
            W[:, col[bone]] += w * W[:, hi]
            W[:, hi] *= (1 - w)
    for s, sg in (("l", 1.0), ("r", -1.0)):
        S = np.array([sg, 1.0, 1.0])
        # arm: elbow blend +-35 mm, forearm twist towards the wrist
        gh, el, wr = J["upperarm_l"] * S, J["lowerarm_l"] * S, J["hand_l"] * S
        ax = (wr - gh) / np.linalg.norm(wr - gh)
        proj = (P - gh) @ ax
        te = np.linalg.norm(el - gh)
        L = np.linalg.norm(wr - gh)
        fore = smoothstep(te - 0.035, te + 0.035, proj)
        twist = smoothstep(te + 0.05, L - 0.02, proj) * 0.5
        add("arm_" + s, {f"upperarm_{s}": 1 - fore, f"lowerarm_{s}": fore - twist * fore,
                         f"lowerarm_twist_01_{s}": twist * fore})
        # leg: knee blend +-40 mm
        hip, kn, an = J["thigh_l"] * S, J["calf_l"] * S, J["foot_l"] * S
        lax = (an - hip) / np.linalg.norm(an - hip)
        lproj = (P - hip) @ lax
        calf = smoothstep(np.linalg.norm(kn - hip) - 0.04, np.linalg.norm(kn - hip) + 0.04, lproj)
        add("leg_" + s, {f"thigh_{s}": 1 - calf, f"calf_{s}": calf})
        # foot / toes
        ball = bones[f"ball_{s}"][0]
        fd = bones[f"foot_{s}"][1] - bones[f"foot_{s}"][0]
        fd = fd / np.linalg.norm(fd)
        toes = smoothstep(-0.012, 0.012, (P - ball) @ fd)
        add("foot_" + s, {f"foot_{s}": 1 - toes, f"ball_{s}": toes})
        # hand: capsule distance to the palm axis and every phalanx
        hb = [b for b in names if b.endswith("_" + s) and (b.startswith(("thumb_", "index_", "middle_", "ring_", "pinky_")) or b == f"hand_{s}")]
        dists = []
        for b in hb:
            h0, t0 = bones[b][0], bones[b][1]
            d, _ = _seg_dist(P.astype(np.float64), h0, t0)
            dists.append(d - (0.028 if b.startswith("hand_") else 0.0))
        dists = np.stack(dists, 1)
        hw = np.exp(-(dists - dists.min(axis=1, keepdims=True)) / 0.0030)
        hw /= hw.sum(axis=1, keepdims=True)
        add("hand_" + s, {b: hw[:, i] for i, b in enumerate(hb)})
    # mesh-graph smoothing
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    pos = np.arange(len(faces)) - np.repeat(starts, sizes)
    last = pos == np.repeat(np.asarray(sizes) - 1, sizes)
    nxt = np.where(last, np.repeat(starts, sizes), np.arange(len(faces)) + 1)
    a, b = faces, faces[nxt]
    A = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr()
    A = ((A + A.T) > 0).astype(np.float64)
    deg = np.asarray(A.sum(axis=1)).ravel()
    Dinv = sp.diags(1.0 / np.maximum(deg, 1))
    M = Dinv @ A
    for _ in range(smooth_iters):
        W = 0.5 * W + 0.5 * (M @ W)
    # top four, renormalised
    idx = np.argsort(-W, axis=1)[:, :4]
    w = np.take_along_axis(W, idx, axis=1)
    w = np.maximum(w, 0)
    w /= np.maximum(w.sum(axis=1, keepdims=True), 1e-9)
    return names, idx.astype(np.uint16), w.astype(np.float32)
