#!/bin/sh
# Render every animation clip (EEVEE) and the pistol routine, then make web copies in docs/media/clips.
#   tools/render_all.sh [clip ...]      (default: all clips)
set -e
cd "$(dirname "$0")/.."
B=/Applications/Blender.app/Contents/MacOS/Blender
HEAVY="$HOME/.agents/bin/heavy"
OUT=renders/clips
WEB=docs/media/clips
mkdir -p "$OUT" "$WEB"
CLIPS=${*:-"pistol walk run jump backflip fall shot idle crouch punch"}
SAMPLES=${SAMPLES:-24}
for c in $CLIPS; do
	t0=$(date +%s)
	if [ "$c" = pistol ]; then
		"$HEAVY" 5400 "$B" -b --factory-startup --python-exit-code 1 --python blender/pistol_clip.py -- "$OUT" \
			--mesh build/human.npz --res 1280x720 --samples "$SAMPLES" 2>&1 | grep -E "^MP4|Error|Traceback" || true
		for s in full draw shoot reload holster; do
			[ -f "$OUT/pistol_$s.mp4" ] && /opt/homebrew/bin/ffmpeg -y -loglevel error -i "$OUT/pistol_$s.mp4" \
				-c:v libx264 -crf 26 -preset slow -pix_fmt yuv420p -movflags +faststart -an "$WEB/pistol_$s.mp4"
		done
	else
		"$HEAVY" 3600 "$B" -b --factory-startup --python-exit-code 1 --python blender/clip.py -- "$c" "$OUT/$c.mp4" \
			--mesh build/human.npz --res 1280x720 --samples "$SAMPLES" 2>&1 | grep -E "^MP4|Error|Traceback" || true
		[ -f "$OUT/$c.mp4" ] && /opt/homebrew/bin/ffmpeg -y -loglevel error -i "$OUT/$c.mp4" -c:v libx264 -crf 26 \
			-preset slow -pix_fmt yuv420p -movflags +faststart -an "$WEB/$c.mp4"
	fi
	echo "done $c in $(( $(date +%s) - t0 ))s"
done
