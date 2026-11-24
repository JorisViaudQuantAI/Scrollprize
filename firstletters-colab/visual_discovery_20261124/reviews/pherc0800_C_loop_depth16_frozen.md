Figé le 2026-11-24T11:37:03.154326+00:00 (UTC).

# PHerc0800 C — contour faible suivi sur les seize profondeurs

**Avis privé sur le même rectangle, sans redéfinition du contour.**

## Source et exposition

J’ai personnellement inspecté la planche `root-review/0800_C_loop_all16depths.png`, seul fichier de ce dossier explicitement autorisé pour cette tâche. Aucun JSON, compte rendu ou fichier voisin n’a été ouvert. Elle montre les indices 8 à 23, en uint8 0–255, pour le rectangle fixé (32,176,336,416) du cadrage gauche de 512 pixels. Chaque cellule représente le même champ local. Je connais mes observations antérieures des prédictions, des cartes officielles et de quelques profondeurs ; je ne revendique pas une lecture aveugle à ces images.

Aucun nouveau contraste, filtre, annotation ou calcul de modèle n’a été appliqué. Les nouvelles sorties à contexte étendu et les sorties PHerc0125 stride48 n’ont pas été examinées avant ce gel.

## Observation de l’évolution

### Indices 8–10

Le champ est surtout strié verticalement. Une structure transversale légèrement courbe occupe la partie supérieure du centre, avec de nombreux petits détails du support autour. Le côté droit comporte une limite plus verticale, encore étroite et irrégulière. Je ne distingue pas un contour fermé de largeur relativement cohérente qui isolerait la boucle suggérée par la prédiction.

### Indices 11–13

Des bandes horizontales deviennent plus marquées à travers la région médiane. La limite verticale droite se creuse visuellement et rejoint une zone sombre arrivant du haut. Des régions claires et sombres se réorganisent autour de cette limite. La structure qui paraît former un côté du motif appartient à un ensemble prolongé vers le bord supérieur, pas seulement au contour d’un petit anneau.

### Indices 14–17

Une plage sombre grossit dans le centre gauche ; une autre, au-dessus et à droite, s’étend le long de la limite verticale. Plusieurs lignes auparavant fines ou discontinues deviennent les bords de régions sombres larges. Les traverses et les montants supposés ne conservent pas une frontière de boucle unique à travers cette évolution. Les changements sont progressifs sur plusieurs indices, et non un seul pixel ou une seule vue aberrante.

### Indices 18–23

La plage supérieure devient une large zone sombre qui se prolonge hors du rectangle. La plage centrale gauche prend elle aussi une forme épaisse et arrondie. Les stries verticales restent visibles dans les régions de matière situées entre et sous ces plages. Les limites sombres se déplacent ou s’élargissent tandis que certains détails transversaux s’atténuent. Je ne retrouve pas un O isolé dont les deux côtés et les traverses resteraient identifiables comme un même tracé d’encre.

Ces observations portent sur le contraste et la forme. Le noir du CT n’est pas étiqueté ici comme de l’encre ; je n’attribue pas une cause chimique à une valeur de gris.

## Ce que les seize profondeurs permettent de distinguer

| Question | Conclusion étayée par ces images | Limite |
|---|---|---|
| Une boucle d’encre est-elle directement identifiable ? | **Non.** Je ne peux pas suivre un contour fermé distinct comme une trace d’encre à travers la série. | Cela ne prouve pas qu’il n’y ait pas d’encre ni qu’un signal d’encre ne puisse être extrait du volume. |
| Existe-t-il une frontière matérielle organisée ? | **Oui, visuellement :** une longue limite droite et des plages reliées évoluent progressivement et dépassent le contour supposé. | « Matérielle » désigne ici une structure de contraste et de texture du CT ; sa nature précise — pli, vide, décollement, limite de fibre ou autre — n’est pas identifiée. |
| La coupe change-t-elle de couche ? | **Possible, mais non établi.** La variation des bandes et des stries est compatible avec la traversée d’une structure en épaisseur ou d’un relief. | Cette planche ne fournit pas le repère géométrique ni la continuité anatomique nécessaires pour identifier une feuille précise ou démontrer un saut vers une autre feuille. |

## Mise à jour de l’hypothèse faible

La compatibilité en forme de O appartient toujours aux **prédictions**, particulièrement à l’ordre k08→k23 et à certaines cartes officielles. La série CT complète rend plus concrète une explication concurrente : le modèle peut répondre à des régions et frontières organisées du support dont l’intégration en profondeur produit une silhouette en boucle.

Je ne transforme pas cette explication en diagnostic certain. Cependant, les seize profondeurs **ne renforcent pas la qualification de glyphe** ; elles renforcent l’obstacle matériel déjà noté. Mon statut reste : hypothèse de silhouette faible, aucun glyphe robuste, aucune séquence.

La future extension du contexte pourra tester si la fermeture gauche est un effet de bord de l’inférence. Même si la boucle survit à cette correction, cela établira sa stabilité de rendu, pas son origine comme encre. Une distinction plus forte entre écriture et support demanderait de relier les traits proposés à la géométrie locale et à un signal qui ne soit pas expliqué par ces frontières de matière. Aucune acquisition ni inférence supplémentaire n’est lancée par cette note.
