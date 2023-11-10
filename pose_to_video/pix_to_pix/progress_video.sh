#!/bin/bash

ffmpeg -framerate 20 -pattern_type glob -i 'training/progress/*.png' \
-vf "drawtext=text='%{n}':fontfile=/usr/share/fonts/google-noto-cjk/NotoSerifCJK-Regular.ttc:x=10:y=10:fontsize=48:fontcolor=black" \
 -y output.mp4
