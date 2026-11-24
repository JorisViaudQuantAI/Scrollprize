Figé le 2026-11-24T11:41:18.267699+00:00 (UTC).

# PHerc0125 — lecture des deux sorties s48 seules

**Note privée figée avant réouverture des s96, du CT ou du montage de comparaison autorisé.**

## Exposition et repère

J’ai personnellement regardé uniquement les deux nouvelles prédictions raw et snapped au cours de cette étape. Je connais mes observations antérieures et les repères gelés avant le calcul ; ce n’est donc pas un aveugle complet à la région. Aucun avis root ou JSON root-review n’a été consulté.

Les fichiers mesurent **512 pixels horizontalement et 608 verticalement** (tableau H×W = 608×512). Le décalage annoncé vers le plein champ est +432 en x et +80 en y. Le rectangle antérieurement fixé (530,180,840,580) devient ainsi (98,100,408,500) localement. Le contexte complet de ce groupe est annoncé ; sa vérification numérique sera effectuée après ce gel.

Aucune rotation, interpolation, modification de contraste ou annotation n’a été appliquée à ces vues.

## Raw reverse ROI s48

Un élément vertical clair assez détaché occupe environ x=135–215, y=190–330. Il rejoint ou approche, vers le haut, un ensemble courbe/oblique qui traverse le centre supérieur, approximativement x=220–440, y=120–290. Cette zone contient une dépression sombre et plusieurs renflements ; la largeur de la réponse varie beaucoup. Je ne peux pas la diviser en un nombre certain de caractères.

Sous le groupe, une petite masse diffuse ou un prolongement oblique est visible vers x=275–355, y=315–390. Plus bas, deux masses arrondies ressortent vers x=250–350, y=410–505 et x=360–450, y=390–480. Elles ne définissent pas une ligne de base commune évidente avec les éléments supérieurs. La seconde dépasse partiellement la boîte fixée en x ; je ne l’intègre pas arbitrairement à un caractère dans cette boîte.

## Snapped reverse ROI s48

Le montant gauche et le groupe supérieur central sont présents. Une branche oblique descend plus visiblement vers le bas droit, à travers x=285–400, y=285–450, avec une largeur irrégulière et des étranglements. La masse arrondie inférieure gauche reste visible ; à sa droite, le renflement semble davantage lié au prolongement oblique.

La silhouette peut évoquer une structure composée de montants et d’une branche, mais aucun caractère précis n’est assez contraint pour être nommé. Les extrémités et la séparation entre des traits d’un même glyphe et des formes voisines demeurent ambiguës.

## Conclusion avant comparaison

La ressemblance générale entre raw et snapped est visible : montant gauche, ensemble supérieur et masses inférieures. Les différences de connexion dans la partie centrale et basse restent importantes pour une lecture. Ces deux résultats ne constituent pas des preuves indépendantes.

- Hypothèse de forme : fragments allongés et reliés possibles, non attribués à une lettre.
- Glyphe robuste : aucun.
- Séquence : aucune.

Le cadrage, la phase des fenêtres, le pas et le mélange ont changé par rapport aux sorties précédentes. Je n’attribue aucun effet observé à Hann seul, et n’émets pas encore de verdict comparatif avec s96 dans cette note.
