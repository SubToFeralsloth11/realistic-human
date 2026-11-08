"""Head and face.

Head-local coordinates are offsets from HEAD.o, the midpoint between the eyeball centres:
x = his left, y = backwards (the face is at -y), z = up. Frankfort plane horizontal.

Construction:
  1. cranium: frontal, parietal-occipital and temporal masses;
  2. face: the solid behind a smooth bicubic surface y = f(x, z) through a table of soft-tissue
     depths (FACE_TABLE), clipped to a face volume, which gives the facial planes, brow, sockets,
     cheeks, mouth area and chin as one continuous skin;
  3. mandible border and masseter for the jaw line;
  4. features that overhang and cannot live in a heightfield: nose cartilages, lips, eyelids;
  5. soft relief bumps and carved creases (nostrils, mouth slit, lid crease, alar groove).
Midline profile targets (y, z): glabella (-0.027, 0.013), nasion (-0.021, 0.003), pronasale
(-0.049, -0.039), subnasale (-0.029, -0.049), labrale sup. (-0.037, -0.064), stomion (-0.030,
-0.0715), labrale inf. (-0.035, -0.081), sulcus (-0.024, -0.093), pogonion (-0.030, -0.105),
menton (-0.014, -0.118). Head height vertex-menton 0.236, breadth 0.153, length 0.197.
"""
import numpy as np

from .sdf import (Bump, Capsule, Ellipsoid, HeightField, Intersect, Plane, Sphere, Tube, Union,
                  bump_tube, norm, rot_euler, rot_from_axes)
from .skeleton import HEAD

O = HEAD.o
EYE_R = 0.0122
EYE_X = 0.0315


def hp(x, y, z):
    return O + np.array([x, y, z], float)


def sym(*nodes):
    out = []
    for n in nodes:
        out += [n, n.mirrored()]
    return out


LANDMARKS = {
    "eye_l": hp(EYE_X, 0, 0),
    "eye_r": hp(-EYE_X, 0, 0),
    "pronasale": hp(0, -0.049, -0.039),
    "stomion": hp(0, -0.030, -0.0715),
    "menton": hp(0, -0.014, -0.118),
    "vertex": hp(0, 0.05, 0.118),
}

# Soft-tissue depth y (m) of the facial surface at (x, z); nose and lips are added separately,
# so the midline here is the surface beneath them. Columns x = 0..0.07 m, mirrored.
FACE_XS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
FACE_TABLE = [
    # z        x=0      0.01     0.02     0.03     0.04     0.05     0.06     0.07
    (0.045, [-0.0240, -0.0237, -0.0226, -0.0200, -0.0150, -0.0065, 0.0145, 0.0500]),   # lower forehead
    (0.030, [-0.0258, -0.0255, -0.0243, -0.0217, -0.0168, -0.0078, 0.0140, 0.0510]),
    (0.018, [-0.0268, -0.0266, -0.0256, -0.0232, -0.0182, -0.0088, 0.0135, 0.0520]),   # brow ridge
    (0.010, [-0.0268, -0.0258, -0.0245, -0.0220, -0.0172, -0.0082, 0.0140, 0.0530]),   # glabella
    (0.000, [-0.0215, -0.0145, -0.0065, -0.0060, -0.0060, -0.0030, 0.0170, 0.0550]),   # eye level (sockets)
    (-0.010, [-0.0240, -0.0160, -0.0105, -0.0085, -0.0080, -0.0040, 0.0160, 0.0550]),
    (-0.022, [-0.0250, -0.0205, -0.0180, -0.0172, -0.0145, -0.0075, 0.0130, 0.0530]),  # infraorbital / cheekbone
    (-0.035, [-0.0275, -0.0235, -0.0205, -0.0182, -0.0152, -0.0085, 0.0120, 0.0510]),
    (-0.048, [-0.0305, -0.0285, -0.0232, -0.0190, -0.0145, -0.0058, 0.0160, 0.0550]),  # subnasale
    (-0.060, [-0.0320, -0.0312, -0.0272, -0.0200, -0.0136, -0.0042, 0.0190, 0.0600]),  # upper lip skin
    (-0.072, [-0.0280, -0.0270, -0.0228, -0.0156, -0.0096, 0.0005, 0.0240, 0.0650]),   # stomion
    (-0.085, [-0.0262, -0.0255, -0.0222, -0.0156, -0.0080, 0.0042, 0.0320, 0.0700]),
    (-0.093, [-0.0240, -0.0235, -0.0205, -0.0145, -0.0065, 0.0065, 0.0360, 0.0720]),   # mentolabial sulcus
    (-0.105, [-0.0310, -0.0298, -0.0252, -0.0165, -0.0052, 0.0120, 0.0420, 0.0750]),   # pogonion
    (-0.115, [-0.0262, -0.0250, -0.0205, -0.0118, 0.0000, 0.0180, 0.0480, 0.0800]),
    (-0.125, [-0.0100, -0.0090, -0.0060, 0.0000, 0.0100, 0.0250, 0.0540, 0.0850]),
    (-0.135, [0.0100, 0.0110, 0.0140, 0.0200, 0.0280, 0.0400, 0.0600, 0.0900]),
]


# ------------------------------------------------------------------ masses
def cranium():
    parts = [
        Ellipsoid(hp(0, 0.036, 0.048), (0.0640, 0.0605, 0.068)),           # frontal
        Ellipsoid(hp(0, 0.082, 0.040), (0.0765, 0.0880, 0.078)),           # parietal / occipital
        Ellipsoid(hp(0, 0.110, -0.020), (0.0560, 0.0500, 0.046)),          # occipital base
        *sym(Ellipsoid(hp(0.050, 0.032, 0.022), (0.0160, 0.0380, 0.034))),  # temporalis
    ]
    return Union(parts, k=0.024, bone="head", tag="scalp")


def face():
    zs = [r[0] for r in FACE_TABLE][::-1]
    Fv = np.array([r[1] for r in FACE_TABLE][::-1])
    hf = HeightField(O, FACE_XS, zs, Fv, lo=hp(-0.075, -0.06, -0.14), hi=hp(0.075, 0.12, 0.05))
    vol = Ellipsoid(hp(0, 0.045, -0.035), (0.0640, 0.0850, 0.0860))
    return Intersect(vol, hf, k=0.020)


def jaw():
    side = [hp(0.016, -0.0165, -0.1070), hp(0.032, -0.0020, -0.1000), hp(0.045, 0.0215, -0.0905),
            hp(0.0510, 0.0460, -0.0810), hp(0.0520, 0.0560, -0.0530), hp(0.0530, 0.0610, -0.0320)]
    radii = [0.0098, 0.0090, 0.0086, 0.0095, 0.0080, 0.0065]
    left = Tube([hp(0, -0.0200, -0.1085)] + side, [0.0105] + radii)
    return Union([left, left.mirrored()], k=0.006, bone="jaw")


# ------------------------------------------------------------------ nose
def nose():
    dorsum = Tube([hp(0, -0.0165, 0.003), hp(0, -0.0230, -0.008), hp(0, -0.0285, -0.018),
                   hp(0, -0.0345, -0.026), hp(0, -0.0395, -0.032)],
                  [0.0045, 0.0048, 0.0050, 0.0052, 0.0056])
    parts = [
        dorsum,
        *sym(Ellipsoid(hp(0.0068, -0.0200, -0.0230), (0.0050, 0.0090, 0.0170), rot_euler(-22, 0, 14))),  # sidewalls
        Ellipsoid(hp(0, -0.0412, -0.0385), (0.0092, 0.0080, 0.0078)),                         # tip lobule
        *sym(Sphere(hp(0.0040, -0.0436, -0.0370), 0.0048)),                                   # domes
        *sym(Ellipsoid(hp(0.0118, -0.0265, -0.0435), (0.0066, 0.0100, 0.0068), rot_euler(0, 0, -25))),  # alae
        Capsule(hp(0, -0.0405, -0.0455), hp(0, -0.0295, -0.0485), 0.0034),                      # columella
    ]
    return Union(parts, k=0.0045, bone="head", tag="nose")


def nose_carves():
    nostril = Union([
        Ellipsoid(hp(0.0060, -0.0352, -0.0482), (0.0030, 0.0062, 0.0026), rot_euler(0, 0, -28)),
        Ellipsoid(hp(0.0060, -0.0310, -0.0420), (0.0030, 0.0055, 0.0065)),
    ], k=0.003)
    groove = Tube([hp(0.0130, -0.0330, -0.0395), hp(0.0175, -0.0270, -0.0405), hp(0.0185, -0.0225, -0.0455),
                   hp(0.0160, -0.0215, -0.0500)], 0.0010)
    return [(nostril, 0.0015), (nostril.mirrored(), 0.0015), (groove, 0.0022), (groove.mirrored(), 0.0022)]


# ------------------------------------------------------------------ mouth
def _arch(y0, c, xs):
    return y0 + c * xs ** 2


def lips():
    xs = np.linspace(-0.0255, 0.0255, 15)
    ax = np.abs(xs)
    zu = -0.0662 + 0.0008 * np.exp(-((ax - 0.0055) / 0.0030) ** 2) - 0.0052 * (ax / 0.0255) ** 2.2
    ru = 0.0016 + 0.0034 * np.sqrt(np.clip(1 - (xs / 0.0262) ** 2, 0, 1))
    yu = _arch(-0.0300, 26.0, xs)
    upper = Tube(np.stack([xs, yu, zu], 1) + O, ru, samples=3)
    zl = -0.0770 + 0.0052 * (ax / 0.0255) ** 2
    rl = 0.0016 + 0.0042 * np.sqrt(np.clip(1 - (xs / 0.0255) ** 2, 0, 1))
    yl = _arch(-0.0285, 25.0, xs)
    lower = Tube(np.stack([xs, yl, zl], 1) + O, rl, samples=3)
    return [(Union([upper], tag="lip", bone="head"), 0.0024), (Union([lower], tag="lip", bone="jaw"), 0.0026)]


def mouth_carves():
    xs = np.linspace(-0.0250, 0.0250, 13)
    z = -0.0714 - 0.0004 * (np.abs(xs) / 0.025)
    slit = []
    for depth in (0.0, 0.0028, 0.0056):
        sx = xs * (1 - depth * 6)
        pts = np.stack([sx, _arch(-0.0300 + depth, 25.5, sx), z], 1) + O
        slit.append(Tube(pts, 0.00045, samples=3))
    corners = sym(Sphere(hp(0.0262, -0.0138, -0.0717), 0.0013))
    return [(Union(slit), 0.0004), (Union(corners), 0.0020)]


# ------------------------------------------------------------------ eyes
def _plane_through(a, b, c, outside_point):
    n = norm(np.cross(b - a, c - a))
    d = float(n @ a)
    if n @ outside_point < d:
        n, d = -n, -d
    return Plane(n, d)


def eye_parts(side=1):
    E = hp(side * EYE_X, 0, 0)
    R_lid = EYE_R + 0.0022

    def e(x, y, z):
        return E + np.array([side * x, y, z])

    def on_shell(x, z, r=R_lid):
        return e(x, -np.sqrt(r * r - x * x - z * z), z)

    cm = e(-0.0140, -0.0035, -0.0010)   # medial canthus
    cl = e(0.0138, 0.0040, 0.0030)      # lateral canthus (positive canthal tilt)
    tu = on_shell(-0.0010, 0.0047)      # upper lid apex, covers ~1.5 mm of the iris
    tl = on_shell(0.0015, -0.0058)      # lower lid low point at the limbus
    up = _plane_through(cm, cl, tu, E + [0, 0, 0.03])
    lo = _plane_through(cm, cl, tl, E - [0, 0, 0.03])
    # bound the fissure between the canthi so the cut cannot run past the eye corners
    mid_c = 0.5 * (cm + cl)
    ax_c = cl - cm
    bound = Ellipsoid(mid_c, (0.5 * np.linalg.norm(ax_c) + 0.0004, 0.0190, 0.0120),
                      rot_from_axes(x=ax_c, y=[0.0, -1.0, 0.0]))
    wedge = Intersect(Intersect(bound, up), lo)
    lidmass = Sphere(E, R_lid, bone="head", tag="lid")
    caruncle = Ellipsoid(e(-0.0128, -0.0052, -0.0003), (0.0020, 0.0018, 0.0026), bone="head", tag="caruncle")
    crease = Tube([on_shell(x, z, EYE_R + 0.0042) for x, z in
                   ((-0.0115, 0.0088), (-0.0045, 0.0112), (0.0040, 0.0114), (0.0115, 0.0080))], 0.0009)
    cavity = Sphere(E, EYE_R + 0.00025)
    return {"lidmass": lidmass, "wedge": wedge, "caruncle": caruncle, "crease": crease, "cavity": cavity}


# ------------------------------------------------------------------ relief
def relief():
    left = [
        Bump(hp(0.045, -0.012, -0.024), (0.016, 0.012, 0.012), 0.0015),                        # malar eminence
        *bump_tube([hp(0.0180, -0.0245, -0.0470), hp(0.0255, -0.0215, -0.0610), hp(0.0300, -0.0160, -0.0745)],
                   0.0030, 0.006, -0.0010),                                                      # nasolabial fold
        Bump(hp(0.0300, -0.0140, -0.0735), (0.004, 0.004, 0.004), 0.0008),                       # modiolus
        *bump_tube([hp(0.0034, -0.0312, -0.0505), hp(0.0055, -0.0345, -0.0610)], 0.0018, 0.004, 0.0008),  # philtral column
        Bump(hp(0.048, 0.045, -0.062), (0.010, 0.020, 0.024), 0.0030, rot_euler(-15, 0, 0)),      # masseter
        Bump(hp(0.0250, -0.0130, -0.0110), (0.010, 0.006, 0.004), -0.0010),                      # tear trough
    ]
    mid = [
        *bump_tube([hp(0, -0.0335, -0.0510), hp(0, -0.0365, -0.0600)], 0.0018, 0.004, -0.0008),  # philtral groove
        Bump(hp(0, -0.0300, -0.1040), (0.014, 0.008, 0.010), 0.0015),                            # mental protuberance
    ]
    return left + [b.mirrored() for b in left] + mid


def build(prog, k_neck=0.02):
    """Append the head to a program that already holds the neck and body."""
    prog.add(cranium(), k=k_neck)
    prog.add(Union([face()], bone="head", tag="face"), k=0.012)
    prog.add(jaw(), k=0.010)
    prog.add(nose(), k=0.0060)
    for node, k in lips():
        prog.add(node, k=k)
    prog.disp(relief())
    for node, k in nose_carves():
        prog.sub(node, k=k)
    for node, k in mouth_carves():
        prog.sub(node, k=k)
    for e in (eye_parts(1), eye_parts(-1)):
        prog.add(e["lidmass"], k=0.0042)
        prog.sub(e["crease"], k=0.0016)
        prog.sub(e["wedge"], k=0.0007)
        prog.add(e["caruncle"], k=0.0008)
        prog.sub(e["cavity"], k=0.0)
