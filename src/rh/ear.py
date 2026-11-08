"""External ear (auricle): helix, antihelix with its two crura, concha, tragus, antitragus, lobule.

Authored in an ear plane: v = backwards along the ear, u = up along the ear's long axis (tilted
back ~18 deg, parallel to the nose bridge), w = out of the ear plane (lateral). The plane is
swung out from the skull by the cephalo-auricular angle so the helix rim stands off the head.
"""
import numpy as np

from .sdf import Capsule, Ellipsoid, Tube, Union, norm
from .skeleton import axis_rot
from .head import hp

TILT = 18.0        # long axis tilted back from vertical
FLARE = 24.0       # cephalo-auricular angle
ORIGIN = hp(0.0675, 0.0700, -0.0230)   # ear canal entrance, left ear

_u = np.array([0.0, np.sin(np.radians(TILT)), np.cos(np.radians(TILT))])
_v = np.array([0.0, np.cos(np.radians(TILT)), -np.sin(np.radians(TILT))])
_w = np.array([1.0, 0.0, 0.0])
_R = axis_rot(_u, -FLARE)   # swing the back edge (+v) outwards (+x)
V, U, W = _R @ _v, _R @ _u, _R @ _w
if V[0] < 0:
    _R = axis_rot(_u, FLARE)
    V, U, W = _R @ _v, _R @ _u, _R @ _w
ROT = np.stack([V, U, W], axis=1)   # local (v, u, w) -> world


def ep(v, u, w=0.0):
    return ORIGIN + v * V + u * U + w * W


def ell(c, r):
    return Ellipsoid(ep(*c), r, ROT)


def left_ear():
    helix = Tube([ep(0.0040, 0.0040, 0.0008), ep(0.0005, 0.0120, 0.0024), ep(0.0010, 0.0210, 0.0036),
                  ep(0.0080, 0.0280, 0.0040), ep(0.0175, 0.0270, 0.0040), ep(0.0260, 0.0190, 0.0036),
                  ep(0.0295, 0.0070, 0.0032), ep(0.0280, -0.0080, 0.0028), ep(0.0230, -0.0190, 0.0024)],
                 [0.0018, 0.0024, 0.0029, 0.0031, 0.0031, 0.0031, 0.0030, 0.0030, 0.0032])
    antihelix = Tube([ep(0.0150, -0.0130, 0.0030), ep(0.0185, -0.0030, 0.0034), ep(0.0180, 0.0080, 0.0035),
                      ep(0.0165, 0.0170, 0.0033)], [0.0024, 0.0026, 0.0024, 0.0019])
    inferior_crus = Tube([ep(0.0180, 0.0080, 0.0035), ep(0.0120, 0.0130, 0.0033), ep(0.0060, 0.0140, 0.0028)],
                         [0.0022, 0.0019, 0.0015])
    plate = ell((0.0135, 0.0000, -0.0005), (0.0165, 0.0290, 0.0028))
    tragus = ell((-0.0025, -0.0055, 0.0040), (0.0034, 0.0056, 0.0026))
    antitragus = ell((0.0125, -0.0160, 0.0036), (0.0042, 0.0032, 0.0028))
    lobule = ell((0.0135, -0.0255, 0.0004), (0.0086, 0.0092, 0.0028))
    back = ell((0.0100, -0.0040, -0.0065), (0.0085, 0.0120, 0.0060))      # conchal bowl seen from behind
    root = ell((0.0010, 0.0000, -0.0060), (0.0060, 0.0200, 0.0070))       # attachment to the head
    return Union([plate, helix, antihelix, inferior_crus, tragus, antitragus, lobule, back, root],
                 k=0.0026, bone="head", tag="ear")


def left_ear_carves():
    concha = ell((0.0080, -0.0035, 0.0050), (0.0085, 0.0110, 0.0065))
    cymba = ell((0.0090, 0.0085, 0.0045), (0.0065, 0.0030, 0.0040))        # upper concha under the crus
    scapha = Tube([ep(0.0060, 0.0225, 0.0055), ep(0.0150, 0.0235, 0.0055), ep(0.0235, 0.0150, 0.0050),
                   ep(0.0250, 0.0020, 0.0045), ep(0.0225, -0.0110, 0.0040)], 0.0016)
    fossa = ell((0.0120, 0.0175, 0.0050), (0.0040, 0.0035, 0.0018))         # triangular fossa
    canal = Capsule(ep(0.0040, -0.0035, 0.0030), ep(0.0010, -0.0030, -0.0120), 0.0024)
    return [(concha, 0.0025), (cymba, 0.0015), (scapha, 0.0012), (fossa, 0.0012), (canal, 0.0020)]


def build(prog):
    ear = left_ear()
    prog.add(ear, k=0.0035)
    prog.add(ear.mirrored(), k=0.0035)
    for node, k in left_ear_carves():
        prog.sub(node, k=k)
        prog.sub(node.mirrored(), k=k)
