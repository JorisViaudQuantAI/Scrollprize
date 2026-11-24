# slot: cpu
# after: wl_001_setup211
# timeout: 2400
# wl lane: probe flatten tools (villa main + pinned) and the current VC3D AppImage
set -u; S=/content/cache/wl211; cd $S
git -C villa grep -l -i "flatboi" 0fb38c45 -- '*.py' '*.cpp' '*.txt' '*.md' 2>/dev/null | head -30 > $OUT/pinned_flatboi.txt
rm -rf $S/villa_main; git clone -q --depth 1 https://github.com/ScrollPrize/villa.git $S/villa_main && git -C $S/villa_main log -1 --format='%H %cd' > $OUT/main_head.txt
cd $S/villa_main; (git ls-files | grep -i -E "flatboi|flatten|render_ink|slim" ) > $OUT/main_flat_files.txt
for f in $(git ls-files | grep -E "spiral-fitting/(render_ink|flatten_spiral_checkpoint)\.py$|flatboi" | head -8); do cp $f $OUT/src__$(echo $f | tr / _) 2>/dev/null; done
git ls-files | grep -E "spiral-fitting/" > $OUT/main_spiral_fitting_files.txt
curl -s https://api.github.com/repos/ScrollPrize/villa/releases > $OUT/releases.json
python3 - <<PY > $OUT/appimages.txt
import json; d=json.load(open('$OUT/releases.json'))
for r in d[:5]:
  for a in r.get('assets',[]):
    if 'AppImage' in a['name']: print(r['tag_name'], a['name'], a['size'], a['browser_download_url'])
PY
U=$(awk 'NR==1{print $4}' $OUT/appimages.txt); echo "url $U"
mkdir -p $S/vc3d.tmp; cd $S/vc3d.tmp; curl -fsSL -o V.AppImage "$U" && chmod +x V.AppImage && ./V.AppImage --appimage-extract >/dev/null && rm V.AppImage && ls squashfs-root/usr/bin > $OUT/vc3d_bins.txt; cat squashfs-root/AppRun | head -40 > $OUT/apprun.txt
cd $S; rm -rf vc3d; mv vc3d.tmp vc3d; echo "$U" > vc3d/url.txt
