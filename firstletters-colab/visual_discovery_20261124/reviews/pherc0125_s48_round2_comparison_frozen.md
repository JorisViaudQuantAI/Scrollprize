Figé le 2026-11-24T11:43:21.571103+00:00 (UTC).

# PHerc0125 — comparaison s48 / s96 et CT sur le groupe préfixé

**Avis privé, après gel de la lecture des deux sorties s48 seules.**

## Séparation des étapes et sources

La première lecture s48 est figée dans `pherc0125_s48_round1_frozen.md`, SHA-256 `6d595f2b26f291032b572c71e4623af220407f06ba0349ad5b205e54b30eb0ee`. Les repères utilisés ci-dessous proviennent du compte rendu antérieur à Hann, `pherc0125_raw_snapped_round3_comparison_frozen.md`, SHA-256 `86864fe43301cb19d81262a05bd7fc60307a537711ecec350e8dabb6345ab2f6`.

Après le gel initial, j’ai ouvert le seul montage de comparaison expressément autorisé, `deliverables/PHerc0125_raffinement_central.png`. Il juxtapose les CT raw/snapped au plan 15, les prédictions s96 et les nouvelles s48 pour les mêmes pixels du groupe. Les CT à d’autres profondeurs avaient été vus dans des étapes précédentes ; aucun nouveau diagnostic anatomique n’en est déduit ici. Aucun JSON root-review ni autre avis root n’a été consulté.

Les nouvelles cartes mesurent 512 × 608 pixels en largeur × hauteur, avec le décalage global (+432,+80). Le groupe fixé local (98,100,408,500) correspond exactement au groupe global (530,180,840,580).

## Mise à jour générale

**Le rendu s48 est visuellement plus continu aux raccords, mais aucune lettre individuelle ni séquence n’est identifiée.** Les grandes formes persistent. Les différences de connexion entre géométries, particulièrement dans la liaison supérieure et le prolongement inférieur, subsistent. Le test stabilise donc une partie de la morphologie visible sans résoudre l’interprétation comme écriture.

## Vérification des quatre repères gelés avant calcul

| Repère, coordonnées globales | Observation après s48 | Conséquence |
|---|---|---|
| Montant gauche (550,270,650,430) | Présent dans raw et snapped. Les transitions rectangulaires des fenêtres s96 sont nettement moins apparentes ; la masse reste épaisse et arrondie. | La présence du montant ne dépend pas seulement d’un raccord s96, mais ses extrémités ne définissent pas à elles seules un caractère. |
| Liaison supérieure (650,200,840,350) | La large liaison plus lumineuse de raw persiste. Dans snapped, l’ensemble est moins uniformément relié et descend davantage en oblique. | Le choix entre des éléments séparés et un contour connecté reste sensible à la géométrie. |
| Prolongement inférieur droit (720,340,800,470) | Le prolongement est plus continu et plus visible dans snapped ; raw conserve plutôt une réponse diffuse ou morcelée dans cette portion. | L’écart observé avant Hann ne disparaît pas avec le nouveau rendu. Pas de hampe commune nettement terminée. |
| Masses inférieures (650,490,840,580) | Les masses restent repérables. Leurs portions contenues dans le rectangle ne composent ni une ligne de base certaine, ni un second caractère complet. | Persistance de réponses locales, sans gain décisif de lecture. Ne pas élargir le contour au-delà du groupe fixé pour compléter une lettre. |

Ces comparaisons sont qualitatives, à coordonnées fixes. Aucune nouvelle étiquette alphabétique n’a été choisie après observation du résultat.

## Correspondance avec le CT

Dans les deux géométries, la région du montant prédit est voisine d’une limite verticale de matière et de faisceaux adjacents. Les branches et masses plus à droite se trouvent dans une région riche en croisements de stries horizontales et verticales. Les CT du montage ne révèlent pas une séparation de caractères qui résoudrait les jonctions variables des prédictions.

La nouvelle continuité du rendu ne change pas cette ambiguïté physique : une frontière ou une texture du support peut également produire une réponse stable. La présence possible d’encre n’est pas exclue, et l’inspection de ces CT n’est pas traitée comme une vérité terrain d’absence d’encre.

## Contexte et limites causales

Le chargement direct des deux tableaux `.context_valid.npy` confirme que **le groupe entier et les quatre repères sont intégralement dans le contexte complet**. Sur l’ensemble du nouveau canevas, le contexte valide représente 42,763 % ; la validité du groupe ne doit donc pas être étendue à toute l’image par défaut.

Les reçus indiquent le même modèle et les mêmes 16 profondeurs ordonnées 8→23, sans changement d’intensité ou rééchantillonnage spatial dans la préparation du sous-volume. En revanche, plusieurs paramètres ont changé ensemble : **cadrage, phase des fenêtres, pas 96→48 et mélange Hann**. La différence constatée est l’effet de cet ensemble de changements. Elle ne permet pas d’estimer l’effet causal du seul mélange Hann.

Raw et snapped restent deux préparations très proches des mêmes données, évaluées par le même modèle. Leur concordance est un contrôle de sensibilité, pas une réplication indépendante.

## Décision

- Hypothèse de forme : montant et branches persistants, sans lettre nommée.
- Glyphe robuste : aucun.
- Séquence : aucune.
- Gain acquis : les principaux éléments ne disparaissent pas lorsque le rendu atténue les raccords.
- Incertitude restante : limites entre éléments, connexions sensibles à la géométrie et cause physique de la réponse.

Il n’est pas justifié de convertir ce gain de continuité visuelle en annonce de découverte. Ce résultat s’ajoute aux étapes conservées ; il ne remplace pas leurs observations et n’est pas mélangé à l’investigation de la boucle PHerc0800.
