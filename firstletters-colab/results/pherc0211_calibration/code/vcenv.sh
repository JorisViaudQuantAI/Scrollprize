# vcenv.sh: VC3D AppImage binaries (extracted by wl_030_probe into /content/cache/wl211/vc3d) + GUI libs
VCR=/content/cache/wl211/vc3d/squashfs-root/AppRun
export QT_QPA_PLATFORM=offscreen
if ! ldconfig -p | grep -q libOpenGL.so.0; then
  ( flock 9; apt-get -qq -o DPkg::Lock::Timeout=600 update >/dev/null || true
    apt-get -qq -o DPkg::Lock::Timeout=600 install -y --no-install-recommends libglvnd0 libglx0 libegl1 libopengl0 libgl1 \
      libgl1-mesa-dri libglx-mesa0 libx11-6 libx11-xcb1 libxcb1 libice6 libsm6 libfontconfig1 libfreetype6 libharfbuzz0b \
      libxkbcommon0 libdbus-1-3 >/dev/null ) 9>/tmp/wl_apt.lock
fi
vc() { local t=$1; shift; local s=$(date +%s); mkdir -p /tmp/vccfg; OMP_NUM_THREADS=${VCT:-6} VC3D_CONFIG_DIR=/tmp/vccfg "$VCR" "$t" "$@"; local rc=$?; echo "$t rc=$rc s=$(( $(date +%s)-s ))"; return $rc; }
