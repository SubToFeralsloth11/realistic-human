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
post fall.mp4 fall "Trip and fall (rigid-body physics): the left foot catches mid-swing, the arms shoot forward, the caught leg comes through for a rescue step, hands and knee hit, the elbows give, chest down"
post shot.mp4 shot "Getting shot (rigid-body physics): a round hits the upper left chest; flinch and guard, pain face, the legs hold for 0.2 s then give, sitting collapse, fall back as the tone fades"
post crouch.mp4 crouch "Extra: squat down, glance around, stand back up"
post punch.mp4 punch "Extra: orthodox stance, guard up; jab, cross (rear heel pivots), left hook, jab, back to neutral"
post pistol_draw.mp4 draw "Draw: hand into the front jeans pocket, grip with the finger off the trigger, pull, rotate at the hip, meet the support hand at the chest, press out to a sighted isosceles aim"
post pistol_shoot.mp4 shoot "Shoot: three rounds; each is a 3.2 N s impulse on the simulated pistol, so muzzle flip and recovery come from the arms' dynamics; slide cycling, flash, casings, slide locks back"
post pistol_reload.mp4 reload "Reload: pistol into the workspace, eyes down, thumb the release, the empty magazine drops free, spare from the left pocket up the magwell, palm it home, thumb the slide stop, press out"
post pistol_holster.mp4 holster "Scan left and right from low ready, support hand off, muzzle down into the right front pocket, push it home, hand out, relax"
[ -f docs/media/clips/pistol_full.mp4 ] && $P media media/clips/pistol_full.mp4 "Full pistol routine in one take: pocket draw, two-handed thumbs-forward grip, three shots with physical recoil, reload, scan, back in the pocket"
$P push "Post animation clips"
