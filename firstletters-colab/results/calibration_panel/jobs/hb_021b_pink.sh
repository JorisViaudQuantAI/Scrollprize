# after: hb_020_pren, hb_005c_ink
# timeout: 5400
# panel ink: every item through the same detectors; p846 reuses hb_005c maps (rv3old, s48, s96) and hb_004c villa maps
export WLSC=PHerc0211; source $INPUTS/panel_env.sh; source $INPUTS/hb_read.sh
mkdir -p $PANEL/p846
python -c "
import tifffile, cv2
for d,k in (('forward','fwd'),('reverse','rev')): cv2.imwrite('$PANEL/p846/villa_'+k+'__x_z+0.png', tifffile.imread('$C/villa_our_'+d+'.tif'))"
hb_rv3old p846; hb_s48 p846 asc desc; hb_s96 p846 asc desc; UM=9.362 hb_post p846; stamp p846
for n in p211win p211fw033 p306 p343; do t=$(date +%s); case $n in p306|p343) UM=8.64;; *) UM=9.362;; esac
  hb_villa $n; hb_rv3old $n; hb_s48 $n asc desc; hb_s96 $n asc desc; hb_post $n; stamp ink_$n $t; done
mkdir -p $OUT/z20; for n in p846 p211win p211fw033 p306 p343; do cp $OUT/red/*${n}_*_z20.png $OUT/red/*${n}_ct20.png $OUT/z20/ 2>/dev/null; done
tar czf $OUT/z20.tgz -C $OUT z20 red; ls $OUT | head -50
