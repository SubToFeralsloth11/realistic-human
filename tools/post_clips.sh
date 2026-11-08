#!/bin/sh
# Attach the rendered web clips in docs/media/clips to their checklist items and push.
#   tools/post_clips.sh            (skips clips that have not been rendered yet)
set -e
cd "$(dirname "$0")/.."
P=tools/progress.py
post() {  # file item caption
	[ -f "docs/media/clips/$1" ] || return 0
	$P media "media/clips/$1" "$3" --item "$2"
	$P item "$2" done
}
post walk.mp4 walk "Walk at 1.2 m/s: heel strike, foot flat, heel-off and toe-off; the pelvis bobs, tilts and turns, the thorax counter-rotates and the arms swing opposite the legs"
post run.mp4 run "Run at 3 m/s: flight phase, forward lean, high knee and heel recovery, bent-arm drive"
post idle.mp4 idle "Standing idle: breathing, slow weight shifts between the legs, glances and blinks"
post jump.mp4 jump "Standing broad jump: counter-movement dip, arm swing, 1.1 m ballistic flight, absorbing landing"
post backflip.mp4 backflip "Standing backflip: dip, arm throw, tuck, full rotation about the centre of mass in 0.72 s of flight, landing"
post fall.mp4 fall "Trip and fall: the swinging foot catches, he pitches forward and braces; the rest is a Verlet ragdoll with muscle tone"
post shot.mp4 shot "Getting shot: a round hits the upper left chest; flinch, pain face (jaw opens, eyes squeeze), legs give, collapse (ragdoll)"
post crouch.mp4 crouch "Extra: squat down, glance around, stand back up"
post punch.mp4 punch "Extra: orthodox stance, guard up; jab, cross (rear heel pivots), left hook, jab, back to neutral"
post pistol_draw.mp4 draw "Draw: right hand into the front jeans pocket, grip, pull, rotate up to a two-handed aim"
post pistol_shoot.mp4 shoot "Shoot: three rounds; trigger press, slide cycling, recoil and muzzle rise, flash, ejected casings"
post pistol_reload.mp4 reload "Reload: thumb the magazine release, empty magazine drops, left hand fetches the spare from the left pocket, seats it, slide release"
post pistol_holster.mp4 holster "Put it back: muzzle down, guide the pistol into the right front pocket, hands relax"
[ -f docs/media/clips/pistol_full.mp4 ] && $P media media/clips/pistol_full.mp4 "Full pistol routine in one take: draw from the pocket, three shots, reload, put it back"
$P push "Post animation clips"
