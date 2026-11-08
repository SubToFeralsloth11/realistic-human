"""Neck, torso, arms and legs: smooth lofted skin envelopes plus soft muscle relief.

The envelopes are generalised cylinders whose superelliptic cross-sections follow adult male
anthropometry (1.80 m, athletic): chest 0.35 x 0.25 m, waist 0.29 x 0.20 m, hips 0.34 m,
neck circ. ~0.38 m, upper arm ~0.30 m, forearm ~0.28 m, wrist ~0.17 m, upper thigh ~0.55 m,
calf ~0.37 m. Muscles and bony landmarks are displacement bumps of a few millimetres on top of
the envelope, the way fat and skin soften them on a real body.
World axes: +Z up, facing -Y, left = +X; left parts are mirrored to the right.
"""
import numpy as np

from .sdf import Bump, Capsule, Ellipsoid, Loft, Tube, Union, bump_tube, rot_euler
from .skeleton import FOREARM, J, SHANK, THIGH, UPPERARM


def sym(*nodes):
    out = []
    for n in nodes:
        out += [n, n.mirrored()]
    return out


# ------------------------------------------------------------------ envelopes
def _torso_sections():
    z0, z1 = 0.80, 1.54
    rows = [
        # z      half-width back   front   n     cy
        (0.800, 0.040, 0.050, 0.030, 2.00, 0.040),   # rounded bottom tucked between the thighs
        (0.830, 0.110, 0.100, 0.062, 2.20, 0.025),
        (0.860, 0.150, 0.128, 0.090, 2.40, 0.012),
        (0.900, 0.160, 0.134, 0.100, 2.50, 0.010),
        (0.950, 0.164, 0.122, 0.105, 2.40, 0.006),
        (1.000, 0.160, 0.108, 0.108, 2.25, 0.002),
        (1.050, 0.150, 0.102, 0.106, 2.10, 0.000),
        (1.100, 0.145, 0.100, 0.104, 2.00, 0.000),
        (1.160, 0.150, 0.103, 0.106, 2.00, 0.004),
        (1.220, 0.158, 0.110, 0.112, 2.10, 0.008),
        (1.280, 0.168, 0.117, 0.122, 2.20, 0.008),
        (1.330, 0.175, 0.121, 0.127, 2.30, 0.008),
        (1.380, 0.180, 0.122, 0.120, 2.45, 0.012),
        (1.420, 0.176, 0.112, 0.100, 2.50, 0.018),
        (1.460, 0.150, 0.092, 0.078, 2.30, 0.022),
        (1.500, 0.095, 0.078, 0.066, 2.10, 0.020),
        (1.540, 0.054, 0.056, 0.048, 2.00, 0.014),   # tucked inside the neck: no ledge
    ]
    sec = [((z - z0) / (z1 - z0), a, a, back, front, n, 0.0, cy) for z, a, back, front, n, cy in rows]
    return Loft((0, 0, z0), (0, 0, z1), (1, 0, 0), sec, round0=0.03, round1=0.02)


def _yoke():
    """Trapezius / clavicle shelf across both shoulders, one symmetric loft (no midline seam)."""
    x1 = 0.215
    half = [
        # |x|   up     down  front  back   n
        (0.000, 0.170, 0.060, 0.050, 0.055, 2.0),
        (0.050, 0.140, 0.060, 0.058, 0.062, 2.0),
        (0.080, 0.110, 0.060, 0.062, 0.068, 2.2),
        (0.120, 0.085, 0.060, 0.066, 0.066, 2.4),
        (0.160, 0.068, 0.060, 0.062, 0.056, 2.4),
        (0.190, 0.060, 0.060, 0.055, 0.046, 2.3),
        (0.215, 0.048, 0.060, 0.048, 0.040, 2.2),
    ]
    rows = [(-x, *r) for x, *r in half[:0:-1]] + half
    sec = [((x + x1) / (2 * x1), up, down, front, back, n) for x, up, down, front, back, n in rows]
    return Loft((-x1, 0.02, 1.42), (x1, 0.02, 1.42), (0, 0, 1), sec, round0=0.03, round1=0.03)


def _neck():
    sec = [
        (0.00, 0.068, 0.068, 0.075, 0.062, 2.1),
        (0.25, 0.062, 0.062, 0.065, 0.058, 2.0),
        (0.55, 0.061, 0.061, 0.063, 0.058, 2.0),
        (1.00, 0.060, 0.060, 0.063, 0.055, 2.0),
    ]
    return Loft((0, 0.012, 1.44), (0, -0.002, 1.665), (1, 0, 0), sec, round1=0.03)


def _arm():
    """Shoulder to wrist in one loft (straight rest pose). Elbow at t = 0.30 / 0.558."""
    te = 0.300 / 0.558
    upper = [  # t_u   lat/dorsal  med/palmar  ant/radial  post/ulnar
        (0.00, 0.052, 0.046, 0.054, 0.052), (0.15, 0.050, 0.044, 0.053, 0.051), (0.35, 0.047, 0.042, 0.051, 0.050),
        (0.55, 0.046, 0.043, 0.054, 0.050), (0.75, 0.042, 0.041, 0.047, 0.046), (0.92, 0.041, 0.043, 0.037, 0.041)]
    elbow = [(te, 0.041, 0.042, 0.038, 0.042)]
    fore = [  # t_f
        (0.15, 0.041, 0.042, 0.047, 0.043), (0.35, 0.035, 0.036, 0.042, 0.038), (0.60, 0.027, 0.028, 0.035, 0.032),
        (0.85, 0.020, 0.021, 0.030, 0.029), (1.00, 0.019, 0.019, 0.030, 0.029)]
    rows = [(t * te, *r) for t, *r in upper] + elbow + [(te + t * (1 - te), *r) for t, *r in fore]
    sec = [(t, a, b, c, d, 2.0 if t < 0.8 else 2.2) for t, a, b, c, d in rows]
    return Loft(J["upperarm_l"], J["hand_l"], UPPERARM.ez, sec, round0=0.045, bone="upperarm_l")


def _leg():
    """Hip joint to ankle in one loft. Knee at t = 0.428 / 0.843."""
    tk = 0.428 / 0.843
    thigh = [  # t_th  lat    med    ant    post
        (0.00, 0.082, 0.072, 0.082, 0.092), (0.12, 0.087, 0.082, 0.087, 0.092), (0.25, 0.088, 0.080, 0.088, 0.088),
        (0.45, 0.080, 0.072, 0.083, 0.078), (0.65, 0.070, 0.064, 0.074, 0.067), (0.85, 0.058, 0.057, 0.061, 0.056)]
    knee = [(tk, 0.050, 0.052, 0.049, 0.053)]
    shank = [  # t_sh
        (0.12, 0.048, 0.052, 0.042, 0.060), (0.30, 0.052, 0.058, 0.040, 0.075), (0.50, 0.045, 0.049, 0.034, 0.063),
        (0.70, 0.033, 0.035, 0.028, 0.042), (0.85, 0.027, 0.028, 0.025, 0.031), (1.00, 0.033, 0.033, 0.028, 0.032)]
    rows = [(t * tk, *r) for t, *r in thigh] + knee + [(tk + t * (1 - tk), *r) for t, *r in shank]
    sec = [(t, a, b, c, d, 2.05) for t, a, b, c, d in rows]
    return Loft(J["thigh_l"], J["foot_l"], (1, 0, 0), sec, round0=0.06, bone="thigh_l")


# ------------------------------------------------------------------ relief (left side, mirrored)
def _relief_left():
    U, Fp, T, S = UPPERARM.p, FOREARM.p, THIGH.p, SHANK.p
    b = [
        # chest
        Bump((0.072, -0.112, 1.360), (0.072, 0.035, 0.055), 0.010, rot_euler(0, -12, 0), tag="pec"),
        Bump((0.100, -0.126, 1.313), (0.010, 0.006, 0.010), 0.0010, tag="areola"),
        Bump((0.100, -0.127, 1.313), (0.0028, 0.004, 0.0028), 0.0022, tag="nipple"),
        *bump_tube([(0.020, -0.120, 1.300), (0.080, -0.124, 1.287), (0.135, -0.095, 1.310)], 0.008, 0.014, -0.003),
        *bump_tube([(0.020, -0.075, 1.462), (0.080, -0.078, 1.466), (0.150, -0.040, 1.480)], 0.007, 0.012, 0.0025),  # clavicle
        # abdomen
        *[Bump((0.036, -0.108, z), (0.032, 0.020, 0.034), 0.0025) for z in (1.235, 1.175, 1.115)],
        Bump((0.030, -0.106, 1.045), (0.035, 0.020, 0.040), 0.0020),
        *bump_tube([(0.008, -0.115, 1.205), (0.060, -0.110, 1.208)], 0.006, 0.012, -0.0015),
        *bump_tube([(0.008, -0.113, 1.145), (0.060, -0.109, 1.148)], 0.006, 0.012, -0.0015),
        Bump((0.135, -0.030, 1.080), (0.035, 0.050, 0.050), 0.0060),                         # external oblique
        *bump_tube([(0.125, -0.070, 1.020), (0.090, -0.094, 0.950), (0.040, -0.100, 0.890)], 0.008, 0.016, -0.004),  # inguinal
        *[Bump(c, (0.010, 0.012, 0.008), 0.0020) for c in ((0.158, -0.060, 1.310), (0.160, -0.050, 1.280), (0.160, -0.040, 1.250))],
        # back
        Bump((0.032, 0.115, 1.120), (0.025, 0.030, 0.100), 0.0070),                          # erector spinae
        Bump((0.090, 0.122, 1.370), (0.050, 0.025, 0.070), 0.0050),                          # scapula
        Bump((0.150, 0.060, 1.280), (0.030, 0.060, 0.100), 0.0060),                          # latissimus
        Bump((0.150, 0.070, 0.930), (0.020, 0.020, 0.020), -0.0030),                         # hip dimple
        Bump((0.040, 0.128, 1.022), (0.008, 0.008, 0.008), -0.0025),                         # sacral dimple
        # glutes
        Bump((0.070, 0.135, 0.905), (0.070, 0.040, 0.075), 0.012, tag="glute"),
        *bump_tube([(0.025, 0.128, 0.838), (0.070, 0.132, 0.826), (0.115, 0.115, 0.832), (0.140, 0.090, 0.845)],
                   0.008, 0.016, -0.004),                                                    # gluteal fold
        # neck
        *bump_tube([(0.050, 0.050, 1.632), (0.040, 0.020, 1.580), (0.026, -0.030, 1.512), (0.013, -0.058, 1.468)],
                   0.010, 0.016, 0.0040),                                                    # sternocleidomastoid
        # arm
        Bump(U(0.000, 0.060, 0.045), (0.050, 0.080, 0.030), 0.0080, UPPERARM.R, tag="deltoid"),
        Bump(U(0.048, 0.165, -0.002), (0.025, 0.080, 0.030), 0.0060, UPPERARM.R),            # biceps
        Bump(U(-0.046, 0.130, 0.000), (0.025, 0.090, 0.035), 0.0060, UPPERARM.R),            # triceps long head
        Bump(U(-0.030, 0.120, 0.035), (0.020, 0.070, 0.020), 0.0040, UPPERARM.R),            # triceps lateral head
        Bump(U(-0.040, 0.300, 0.000), (0.015, 0.015, 0.015), 0.0040),                        # olecranon
        Bump(U(0.000, 0.290, -0.042), (0.012, 0.012, 0.012), 0.0030),                        # medial epicondyle
        Bump(Fp(0.040, 0.060, 0.020), (0.020, 0.080, 0.025), 0.0060, FOREARM.R),             # brachioradialis
        Bump(Fp(0.000, 0.080, 0.038), (0.025, 0.080, 0.015), 0.0030, FOREARM.R),             # extensors
        Bump(Fp(-0.028, 0.245, 0.012), (0.008, 0.008, 0.008), 0.0025),                       # ulnar styloid
        # leg
        Bump(T(-0.050, 0.340, -0.030), (0.030, 0.060, 0.030), 0.0060, THIGH.R),              # vastus medialis
        Bump(T(0.070, 0.220, -0.010), (0.030, 0.140, 0.050), 0.0050, THIGH.R),               # vastus lateralis
        Bump(T(0.008, 0.200, -0.075), (0.030, 0.150, 0.020), 0.0040, THIGH.R),               # rectus femoris
        *bump_tube([T(0.030, 0.080, -0.085), T(-0.030, 0.220, -0.070), T(-0.055, 0.360, -0.030)], 0.008, 0.016, -0.0020),
        Bump(T(0.000, 0.420, -0.056), (0.026, 0.030, 0.020), 0.0040, THIGH.R, tag="kneecap"),
        Bump(S(0.000, 0.040, -0.048), (0.010, 0.035, 0.008), 0.0030, SHANK.R),               # patellar tendon
        Bump(T(0.000, 0.440, 0.055), (0.030, 0.040, 0.020), -0.0040, THIGH.R),               # popliteal fossa
        Bump(S(-0.025, 0.140, 0.065), (0.030, 0.080, 0.030), 0.0100, SHANK.R),               # gastrocnemius medial
        Bump(S(0.025, 0.120, 0.060), (0.025, 0.070, 0.025), 0.0070, SHANK.R),                # gastrocnemius lateral
        Bump(S(0.025, 0.150, -0.032), (0.015, 0.120, 0.012), 0.0030, SHANK.R),               # tibialis anterior
        Bump(S(-0.034, 0.405, -0.004), (0.011, 0.011, 0.011), 0.0050),                       # medial malleolus
        Bump(S(0.036, 0.420, 0.006), (0.011, 0.011, 0.011), 0.0050),                         # lateral malleolus
    ]
    return b


def relief():
    left = _relief_left()
    mid = [
        *bump_tube([(0, -0.115, 1.250), (0, -0.112, 1.150), (0, -0.108, 1.060)], 0.006, 0.014, -0.0025),  # linea alba
        *bump_tube([(0, 0.125, 1.050), (0, 0.135, 1.250), (0, 0.130, 1.420)], 0.008, 0.016, -0.0040),     # spinal groove
        *bump_tube([(0, 0.150, 0.980), (0, 0.145, 0.900), (0, 0.130, 0.840)], 0.008, 0.016, -0.0080),     # gluteal cleft
        Bump((0, 0.125, 1.400), (0.070, 0.030, 0.100), 0.0030),                                           # mid trapezius
        Bump((0, -0.068, 1.462), (0.012, 0.012, 0.012), -0.0040),                                         # suprasternal notch
        Bump((0, -0.058, 1.545), (0.008, 0.008, 0.012), 0.0040),                                          # thyroid prominence
    ]
    return left + [b.mirrored() for b in left] + mid


def navel():
    return Union([Ellipsoid((0, -0.1045, 1.075), (0.0065, 0.0100, 0.0085)), Ellipsoid((0, -0.099, 1.074), (0.004, 0.006, 0.004))])


def build(prog):
    """Append neck, torso, arms and legs (hands, feet and head come later)."""
    prog.add(Union([_torso_sections()], bone="spine_02"))
    prog.add(Union([_yoke()], bone="spine_04"), k=0.035)
    prog.add(Union([_neck()], bone="neck_01"), k=0.030)
    for part, k in ((_arm(), 0.040), (_leg(), 0.045)):
        prog.add(part, k=k)
        prog.add(part.mirrored(), k=k)
    # axillary folds: pectoralis and latissimus tendons into the upper arm
    folds = [Capsule((0.120, -0.075, 1.370), UPPERARM.p(0.025, 0.080, -0.006), 0.019, bone="spine_04"),
             Capsule((0.125, 0.085, 1.330), UPPERARM.p(-0.025, 0.075, -0.006), 0.021, bone="spine_04")]
    for f in folds:
        prog.add(f, k=0.030)
        prog.add(f.mirrored(), k=0.030)
    # Achilles tendon
    ach = Capsule(SHANK.p(0, 0.270, 0.036), SHANK.p(0, 0.405, 0.040), 0.0070, bone="calf_l")
    prog.add(ach, k=0.010)
    prog.add(ach.mirrored(), k=0.010)
    prog.add(Ellipsoid((0, -0.068, 0.862), (0.034, 0.030, 0.040), bone="pelvis", tag="groin"), k=0.025)
    prog.disp(relief())
    prog.sub(navel(), k=0.004)
