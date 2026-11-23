The finished character as one Blender 5.1 file, generated entirely by the code in this repository (no scans, downloads or motion capture).

**New in v1.1: physics-driven animation**
- Every clip now runs through an articulated rigid-body simulation of the body (17 segments with real masses and inertias, anatomical joint limits, muscles as damped drives with gravity compensation, ground friction, limb collisions), so arms, trunk and head carry weight, lag and follow through instead of moving stiffly.
- New ragdolls: the trip and fall (protective arm reaction, rescue step, hands and knee hit, elbows give) and the hit reaction (flinch, legs give, collapse as the muscle tone fades).
- New pistol routine: sighted isosceles aim with a solved thumbs-forward two-handed grip; each shot is a physical impulse, so muzzle flip and recovery come from the arms; reload carries the spare up the magwell.

**Contents**
- Body: ~980k vertices (face 0.8 mm, hands 0.9 mm, body 2.5 mm), random-walk subsurface skin with procedural pores and colour zones
- Eyes with refractive corneas, 28 teeth with gums and tongue, 110k scalp strands, stubble, brows and lashes (Principled Hair in Cycles)
- T-shirt, five-pocket jeans and leather sneakers at full resolution (~2M vertices), skinned to the rig
- Compact 9 mm pistol with slide, barrel, trigger, slide stop, magazines, cartridges, casings and muzzle flash
- 66-bone armature (Unreal-style names, eyelid bones, jaw, forearm twist)

**Animations** (select `Rig`, then the Action editor):
`idle`, `walk`, `run`, `jump`, `backflip`, `crouch`, `punch`, `fall`, `shot`, and `pistol` (active: draw from the jeans pocket, three shots, reload, scan, put it back; the pistol parts are keyed to this one).

Render with Cycles for the intended look; EEVEE works with simplified outputs for hair, corneas and denim. The file is heavy (about 3M vertices before hair): give Blender 8 GB or more.
