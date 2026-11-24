Figé le 2026-11-24T11:45:27.286202+00:00 (UTC).

# PHerc0800 C — nouvelle carte à contexte réel, lecture seule

**Avis privé figé avant comparaison directe à l’ancien cadrage.**

## Périmètre et exposition

J’ai personnellement ouvert la nouvelle carte `0800_C_left_context96_k08_to_k23_s96.preview.png` et son masque de contexte rendu en noir/blanc. Je connais mes précédentes observations, y compris l’hypothèse faible de boucle et les seize profondeurs CT. Cette lecture n’est pas aveugle à la région. Aucun avis root ni JSON root-review n’a été lu. L’ancienne carte n’a pas encore été rouverte pendant cette étape.

La nouvelle carte fait 608 pixels de large sur 512 de haut. Le mapping annoncé est global = local + (804,650). Le contour fixé occupe maintenant le rectangle local (128,176,432,416). Il n’a pas été redéfini après observation.

## Observation de la nouvelle carte

Le groupe présente plusieurs espaces sombres superposés entourés par des régions claires. Dans le rectangle fixé, un intérieur sombre ovalaire est entouré d’une traverse supérieure, d’une traverse inférieure et de deux côtés clairs. Le côté gauche est visuellement présent ; la bordure qui l’entoure ne paraît pas interrompue par un grand intervalle sombre à cette résolution. Je peux donc suivre une **fermeture visuelle de type boucle** dans cette prédiction.

Cette boucle est épaisse, assez aplatie et liée à des éléments au-dessus et au-dessous. Le côté droit se prolonge verticalement ; le côté gauche rejoint également d’autres masses. La boîte isole un motif utile à examiner, mais ne prouve pas que ce motif constitue un caractère séparé du reste.

La silhouette reste compatible avec une forme de type O, à titre d’hypothèse faible. Aucun autre caractère et aucune séquence ne sont proposés. Les grandes masses extérieures au rectangle ne sont pas ajoutées à la lecture pour compléter une lettre.

## Contexte

Le chargement du masque confirme que **tous les pixels du rectangle (128,176,432,416) sont dans le contexte complet**. Le masque a pour boîte englobante (96,96,576,480), avec des particularités locales en dehors du contour ; la validité de toute l’image n’est pas présumée. Sa fraction globale est 59,2105 %.

La fermeture gauche se trouve donc, dans cette nouvelle sortie, dans une région réellement admissible selon le masque fourni. Ce fait lève une limitation technique pour le contour examiné, sans attribuer une origine d’encre à la réponse.

## Verdict avant comparaison directe

- Morphologie : une boucle est visible dans le rectangle fixé.
- Compatibilité avec une lettre : forme de type O, hypothèse faible.
- Glyphe robuste : aucun, en raison des attaches à une structure plus grande et de l’explication concurrente par les frontières de matière déjà observées dans le CT.
- Séquence : aucune.

L’ancienne sortie sera comparée sur le mapping exact nouvelle[:,96:608] ↔ ancienne. Cette note ne contient pas encore un verdict sur sa stabilité entre les deux calculs. Aucun modèle, traitement de contraste, interpolation ou nouvelle acquisition n’a été exécuté par cette lecture.
