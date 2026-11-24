Figé le 2026-11-24T11:29:37.100560+00:00 (UTC).

# Contrôle public auteur841, mosaïque — revue visuelle séparée

**Contrôle d’exécution et de restitution de formes déjà annotées ; aucune découverte.**

Les avis sur les nouveaux cadrages PHerc0800 ont été figés avant d’examiner ce contrôle. J’ai personnellement vu la sortie Studio de la mosaïque 1024 × 768, la planche des CT préparés et des annotations publiées, la carte d’annotation native, le masque de contexte et un aperçu portant les contours connus. Aucun avis de root ni le fichier `visual_calibration_note.md` n’a été ouvert.

## Nature de la référence

Le reçu identifie le fixture public `841-expanded-layer-03-approved-r000001`, issu du cache auteur fixé à la révision `ad919d288ede1c4daf92725288bb6fe29c558591`. La mosaïque assemble directement six tuiles préparées. Les valeurs d’annotation 0, 1 et 2 signifient respectivement inconnu, fond et encre. Elle contient deux grandes formes annotées ; il ne s’agit pas d’une ligne complète de texte.

Le reçu d’inférence conserve l’ordre préparé de 16 plans de l’auteur. Ni une normale anatomique ni un zéro de profondeur ne sont déduits de ce contrôle. Il ne valide pas une généralisation indépendante sur le rouleau non déchiffré.

## Ce qui est restitué visuellement

**Première forme annotée, à gauche :** l’annotation présente une hampe oblique et deux traverses épaisses à ses extrémités. La prédiction retrouve nettement la traverse inférieure et une partie de la traverse supérieure. La partie centrale de la hampe est moins continue et la limite de la forme est floue. Avec l’annotation visible, plusieurs composantes correspondantes sont reconnaissables ; sans cette indication, je n’aurais pas déduit une lettre complète de ces seules masses.

**Deuxième forme annotée, à droite :** la fourche et les deux branches droites correspondent à une zone fortement prédite. L’extrémité haute de la hampe possède aussi une réponse. En revanche, la partie inférieure gauche et une portion de la hampe sont faibles ou interrompues : le caractère entier ne ressort pas avec la topologie nette du tracé connu. Des masses lumineuses voisines se confondent visuellement avec ses branches.

Il existe des débordements de réponse dans le voisinage déclaré comme fond, ainsi que des masses hors de la zone annotée. Les dernières ne peuvent pas être déclarées fausses positives, puisque la valeur 0 signifie inconnu. Les raccords du rendu sans mélange et le flou contribuent aux limites de lisibilité ; cette revue ne sépare pas quantitativement leur part des limites du modèle.

## Contexte vérifié

Le masque strict vaut vrai dans (96,96,960,672), soit 63,28125 % du canevas. Un comptage direct confirme que les 67 113 pixels d’encre annotée et les 111 310 pixels de fond annoté sont tous à l’intérieur de ce contexte complet. Les interruptions des traits connus ne peuvent donc pas être expliquées simplement par le fait que ces annotations seraient au bord d’un contexte invalide.

## Conséquence pour l’exploration

Ce contrôle montre que le pipeline peut répondre à des portions importantes de formes connues, mais que la sortie actuelle au pas 96 ne restitue pas toujours leur structure entière. Il serait excessif d’exiger qu’un vrai caractère inconnu soit nécessairement parfaitement net dans ce rendu. Réciproquement, une ressemblance en boucle ou en angle ne devient pas fiable au seul motif qu’elle est lumineuse.

La mosaïque est un meilleur contrôle visuel que la seule tuile de 512 pixels car elle inclut les deux formes publiées en entier. Elle reste une référence connue et ne démontre pas une découverte, une lecture indépendante ni la bonne orientation des entrées PHerc0800.

## Figure d’audit

`author841_mosaic_prediction_known_ink_outline.png` conserve la prédiction native en niveaux de gris. Seule une bordure intérieure rouge d’un pixel, calculée depuis le label d’encre déjà publié, a été ajoutée. Le rouge n’est ni une sortie du modèle ni un trait découvert. Aucun ajustement spatial, contraste ou interpolation n’a été appliqué.
