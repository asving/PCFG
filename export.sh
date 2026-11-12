#!/bin/bash
# Turn the figure pages into PNGs with headless Chrome (2x pixel density). Usage: bash export.sh [folder]
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
cd "$(dirname "$0")/${1:-figures}" || exit 1
shot() {  # name width height
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=2 \
    --user-data-dir=/tmp/chrome-headless-pcfg --window-size="$2,$3" \
    --screenshot="$PWD/$1.png" "file://$PWD/$1.html" > /dev/null 2>&1
  echo "$1.png $(sips -g pixelWidth -g pixelHeight "$1.png" 2>/dev/null | awk '/pixel/{printf "%s ", $2}')"
}
if [ $# -ge 4 ]; then           # bash export.sh FOLDER NAME WIDTH HEIGHT [NAME WIDTH HEIGHT ...]
  shift
  while [ $# -ge 3 ]; do shot "$1" "$2" "$3"; shift 3; done
else
  shot fig0_process 1196 520
  shot fig1 1196 900
  shot fig2 1196 830
  shot fig3 1196 1100
fi
