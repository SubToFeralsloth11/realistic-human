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
                  Loft, bump_tube, norm, rot_euler, rot_from_axes)
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
    (-0.060, [-0.0315, -0.0308, -0.0270, -0.0200, -0.0136, -0.0042, 0.0190, 0.0600]),  # upper lip skin
    (-0.072, [-0.0285, -0.0277, -0.0233, -0.0158, -0.0096, 0.0005, 0.0240, 0.0650]),   # stomion (lips are bumps)
    (-0.085, [-0.0260, -0.0254, -0.0222, -0.0156, -0.0080, 0.0042, 0.0320, 0.0700]),
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
    # face volume: vertical loft whose half-width tapers from the zygomatic arches (0.070) to the
    # jaw angles (0.050) and the chin; its front is far forward so the heightfield defines the face
    z0, z1 = -0.125, 0.050
    rows = [  # z      half-width  back
        (-0.125, 0.012, 0.030), (-0.115, 0.026, 0.050), (-0.100, 0.040, 0.065), (-0.085, 0.050, 0.075),
        (-0.065, 0.056, 0.080), (-0.040, 0.064, 0.085), (-0.022, 0.0695, 0.085), (0.000, 0.0680, 0.085),
        (0.025, 0.0660, 0.085), (0.050, 0.0640, 0.085)]
    sec = [((z - z0) / (z1 - z0), a, a, back, 0.15, 2.2, 0.0, 0.0) for z, a, back in rows]
    vol = Loft(hp(0, 0.030, z0), hp(0, 0.030, z1), (1, 0, 0), sec, round0=0.012, round1=0.012)
    return Intersect(vol, hf, k=0.016)


def jaw():
    side = [hp(0.016, -0.0165, -0.1070), hp(0.032, -0.0020, -0.1000), hp(0.045, 0.0215, -0.0905),
            hp(0.0510, 0.0460, -0.0810), hp(0.0520, 0.0560, -0.0530), hp(0.0530, 0.0610, -0.0320)]
    radii = [0.0098, 0.0090, 0.0086, 0.0095, 0.0080, 0.0065]
    left = Tube([hp(0, -0.0200, -0.1085)] + side, [0.0105] + radii)
    return Union([left, left.mirrored()], k=0.006, bone="jaw")


# ------------------------------------------------------------------ nose
def nose():
    """Bony/cartilaginous pyramid as a loft along the dorsum, alar rims as curved tubes."""
    rows = [  # t     half-w  front   back    n
        (0.00, 0.0055, 0.0045, 0.010, 1.8),   # nasion
        (0.20, 0.0070, 0.0050, 0.012, 1.7),
        (0.40, 0.0085, 0.0055, 0.014, 1.7),   # rhinion
        (0.62, 0.0090, 0.0058, 0.014, 1.9),   # supratip
        (0.80, 0.0100, 0.0070, 0.012, 2.1),   # tip lobule
        (0.92, 0.0095, 0.0070, 0.010, 2.2),
        (1.00, 0.0070, 0.0050, 0.008, 2.2),
    ]
    sec = [(t, w, w, f, b, n) for t, w, f, b, n in rows]
    dorsum = Loft(hp(0, -0.0160, 0.0040), hp(0, -0.0445, -0.0425), (1, 0, 0), sec, round0=0.004, round1=0.006)
    ala = Tube([hp(0.0125, -0.0225, -0.0468), hp(0.0140, -0.0270, -0.0430), hp(0.0125, -0.0330, -0.0400),
                hp(0.0090, -0.0380, -0.0400), hp(0.0050, -0.0415, -0.0412)],
               [0.0035, 0.0045, 0.0044, 0.0038, 0.0030])
    tip = Ellipsoid(hp(0, -0.0418, -0.0390), (0.0085, 0.0075, 0.0072))
    columella = Capsule(hp(0, -0.0425, -0.0455), hp(0, -0.0300, -0.0488), 0.0032)
    return Union([dorsum, tip, ala, ala.mirrored(), columella], k=0.0035, bone="head", tag="nose")


def nose_carves():
    nostril = Union([
        Ellipsoid(hp(0.0062, -0.0345, -0.0482), (0.0030, 0.0062, 0.0026), rot_euler(0, 0, -28)),
        Ellipsoid(hp(0.0062, -0.0310, -0.0420), (0.0030, 0.0055, 0.0065)),
    ], k=0.003)
    return [(nostril, 0.0015), (nostril.mirrored(), 0.0015)]


# ------------------------------------------------------------------ mouth
def _arch(y0, c, xs):
    return y0 + c * xs ** 2


def lip_bumps():
    """Lips as soft swellings along the dental arch: amplitude tapers to the corners."""
    out = []
    xs = np.linspace(-0.0240, 0.0240, 17)
    step = xs[1] - xs[0]
    for x in xs:
        ax = abs(x)
        taper = max(1 - (x / 0.0268) ** 2, 0.0) ** 0.7
        cupid = np.exp(-((ax - 0.0055) / 0.0030) ** 2)
        zu = -0.0663 + 0.0006 * cupid - 0.0045 * (ax / 0.026) ** 2
        ru = (0.0042, 0.0050, 0.0046)
        out.append(Bump(hp(x, _arch(-0.0315, 26.0, x), zu), ru, 0.0045 * taper / (32 / 35 * ru[0] / step), tag="lip"))
        zl = -0.0776 + 0.0050 * (ax / 0.026) ** 2
        rl = (0.0046, 0.0055, 0.0058)
        out.append(Bump(hp(x, _arch(-0.0290, 25.0, x), zl), rl, 0.0050 * taper / (32 / 35 * rl[0] / step), tag="lip"))
    # vermilion border ("white roll") of the upper lip, following the cupid's bow
    xs = np.linspace(-0.0225, 0.0225, 11)
    border = [hp(x, _arch(-0.0352, 26.0, x), -0.0626 + 0.0010 * np.exp(-((abs(x) - 0.0055) / 0.0030) ** 2)
                 - 0.0050 * (abs(x) / 0.026) ** 2) for x in xs]
    out += bump_tube(border, 0.0010, 0.0035, 0.0004)
    return out


def mouth_carves():
    xs = np.linspace(-0.0250, 0.0250, 13)
    z = -0.0716 - 0.0004 * (np.abs(xs) / 0.025)
    slit = []
    for depth in (0.0, 0.0028, 0.0056, 0.0084, 0.0112):
        sx = xs * (1 - depth * 6)
        pts = np.stack([sx, _arch(-0.0365 + depth, 26.0, sx), z], 1) + O
        slit.append(Tube(pts, 0.00045, samples=3))
    # oral vestibule and mouth cavity behind the lip seal (holds the teeth; visible when the jaw opens)
    cavity = Union([Ellipsoid(hp(0, -0.0150, -0.0716), (0.0245, 0.0105, 0.0105), tag="mouth"),
                    Ellipsoid(hp(0, 0.0040, -0.0716), (0.0230, 0.0180, 0.0120), tag="mouth")], k=0.006)
    return [(Union(slit), 0.0004), (cavity, 0.0015)]


# ------------------------------------------------------------------ eyes
def _plane_through(a, b, c, outside_point):
    n = norm(np.cross(b - a, c - a))
    d = float(n @ a)
    if n @ outside_point < d:
        n, d = -n, -d
    return Plane(n, d)


# Lid geometry in eye-local offsets (x mirrored per side): canthi and the two margin apexes.
LID_R = EYE_R + 0.0022
LID_CM = (-0.0145, -0.0035, -0.0010)    # medial canthus
LID_CL = (0.0142, 0.0040, 0.0030)       # lateral canthus (positive canthal tilt)
LID_TU = (-0.0010, 0.0037)              # upper margin apex (x, z): covers ~2 mm of the iris
LID_TL = (0.0015, -0.0059)              # lower margin low point at the limbus


def eye_parts(side=1):
    E = hp(side * EYE_X, 0, 0)
    R_lid = LID_R

    def e(x, y, z):
        return E + np.array([side * x, y, z])

    def on_shell(x, z, r=R_lid):
        return e(x, -np.sqrt(r * r - x * x - z * z), z)

    cm = e(*LID_CM)
    cl = e(*LID_CL)
    tu = on_shell(*LID_TU)
    tl = on_shell(*LID_TL)
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
    crease = [on_shell(x, z, EYE_R + 0.0040) for x, z in
              ((-0.0115, 0.0088), (-0.0045, 0.0112), (0.0040, 0.0114), (0.0115, 0.0080))]
    cavity = Sphere(E, EYE_R + 0.00025)
    return {"lidmass": lidmass, "wedge": wedge, "caruncle": caruncle, "crease": crease, "cavity": cavity}


# ------------------------------------------------------------------ relief
def relief():
    left = [
        *bump_tube([hp(0.0190, -0.0240, -0.0480), hp(0.0255, -0.0210, -0.0590), hp(0.0300, -0.0170, -0.0680)],
                   0.0050, 0.008, -0.0006),                                                      # nasolabial fold
        Bump(hp(0.0300, -0.0140, -0.0735), (0.004, 0.004, 0.004), 0.0006),                       # modiolus
        Bump(hp(0.0262, -0.0136, -0.0718), (0.0025, 0.0025, 0.0025), -0.0007),                   # mouth corner
        Bump(hp(0.048, 0.045, -0.062), (0.010, 0.020, 0.024), 0.0030, rot_euler(-15, 0, 0)),      # masseter
        Bump(hp(0.0250, -0.0130, -0.0110), (0.010, 0.006, 0.004), -0.0008),                      # tear trough
        *bump_tube([hp(0.008, -0.0270, 0.0150), hp(0.024, -0.0245, 0.0170), hp(0.040, -0.0175, 0.0165)],
                   0.0060, 0.010, 0.0010),                                                      # superciliary arch
        Bump(hp(0.0380, -0.0165, -0.0330), (0.020, 0.012, 0.018), 0.0012),                       # malar fat pad
        Bump(hp(0.0480, -0.0020, -0.0480), (0.016, 0.016, 0.018), -0.0010),                      # submalar hollow
    ]
    mid = [
        *bump_tube([hp(0, -0.0335, -0.0510), hp(0, -0.0362, -0.0600)], 0.0020, 0.004, -0.0006),  # philtral groove
        Bump(hp(0, -0.0300, -0.1040), (0.014, 0.008, 0.010), 0.0015),                            # mental protuberance
    ]
    return left + [b.mirrored() for b in left] + mid


def build(prog, k_neck=0.02):
    """Append the head to a program that already holds the neck and body."""
    prog.add(cranium(), k=k_neck)
    prog.add(Union([face()], bone="head", tag="face"), k=0.012)
    prog.add(jaw(), k=0.016)     # soft enough that the mandible border shades into the neck without a groove
    prog.add(nose(), k=0.0050)
    crease = bump_tube([hp(0.0105, -0.0330, -0.0385), hp(0.0160, -0.0265, -0.0405), hp(0.0172, -0.0222, -0.0458)],
                       0.0015, 0.004, -0.0008)                                                  # alar crease
    prog.disp(relief() + lip_bumps() + crease + [b.mirrored() for b in crease])
    for node, k in nose_carves():
        prog.sub(node, k=k)
    for node, k in mouth_carves():
        prog.sub(node, k=k)
    for e in (eye_parts(1), eye_parts(-1)):
        prog.add(e["lidmass"], k=0.0042)
        prog.sub(e["wedge"], k=0.0007)
        prog.add(e["caruncle"], k=0.0008)
        prog.sub(e["cavity"], k=0.0)
    # upper lid fold (supratarsal crease) on the finished lids
    fold = bump_tube(eye_parts(1)["crease"], 0.0016, 0.004, -0.0007)
    prog.disp(fold + [b.mirrored() for b in fold])
