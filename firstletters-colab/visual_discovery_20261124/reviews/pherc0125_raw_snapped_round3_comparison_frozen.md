Figé le 2026-11-24T11:15:48.798106+00:00 (UTC).

# PHerc0125 — troisième lecture, comparaison raw / snapped au pas 96

**Avis privé gelé après une première lecture séparée des deux cartes raw.**

## Séquence et portée

La première observation raw a été enregistrée dans `pherc0125_raw_round3_initial_frozen.md`, SHA-256 `cd7b9ffccd4ea59e1c6a33d1e06389c7fcfb71ff2fe7f4441ebc4252ca8ccdc9`, avant de rouvrir les cartes snapped. Les quatre aperçus ont ensuite été personnellement examinés. Un montage du rectangle central préexistant (530,180,840,580) affiche les pixels originaux des quatre cartes à la même échelle, sans contraste ni interpolation ; l’égalité avec les pixels sources a été vérifiée.

Je n’ai pas lu root-review ni les conclusions d’autres agents. J’ai connaissance de mes CT et observations précédents et du projet annoncé de comparaison stride48/Hann. Aucune sortie Hann n’a été examinée avant ce gel. Aucun des rapports antérieurs n’est modifié.

## Verdict comparatif

**Plusieurs masses prédites persistent entre les deux géométries. Aucune ne devient une lettre identifiée par cette concordance.** La localisation de ces formes paraît donc moins dépendante du seul ajustement raw/snapped, mais leur cause peut rester liée au support ou à une réponse commune du modèle. Les mêmes données de départ, le même modèle et un calcul très proche ne constituent pas une réplication indépendante.

### Ordre desc16 : forte ressemblance des grandes masses

La masse centrale presque triangulaire vers (550,450) apparaît dans raw et snapped, de taille générale et d’emplacement très proches. Elle reste largement pleine, sans contour interne ou succession de traits permettant une identification. Le groupe gauche vers x=100–230, y=310–570 conserve son organisation en masse supérieure et prolongement inférieur.

Le grand groupe droit, avec une dépression sombre interne et une extension latérale, est lui aussi conservé dans les deux cartes. Mais sa contrainte de contexte reste entière : la boîte initiale n’est valide qu’à environ 64 % et ne permet pas de retenir son contour entier. Une concordance entre deux résultats également incomplets ne rétablit pas le contexte manquant.

Une différence visible concerne la petite masse au-dessus du triangle central, vers x=550–680, y=180–320 : elle est plus étendue et lumineuse dans snapped. Des éléments faibles proches du centre gauche changent également d’intensité. Ces variations ne font apparaître aucune ligne.

### Ordre reverse16 : emplacement stable, connexions variables

Le montant du centre gauche du groupe, autour de x=550–650, y=270–430, persiste. Le groupe oblique/courbe à sa droite est présent dans les deux géométries, ainsi que les masses arrondies plus basses vers x=650–870, y=490–580. Ce sont les correspondances les plus utiles à vérifier dans le rendu corrigé.

Cependant, dans raw, la partie supérieure droite de ce groupe paraît plus reliée et plus large, particulièrement vers x=680–840, y=220–340. Dans snapped, cette région est moins uniformément reliée et un prolongement plus vertical vers x=720–800, y=340–470 ressort davantage. La frontière entre deux formes séparées et une forme fusionnée change donc précisément là où une lecture aurait besoin de connaître les jonctions.

À gauche du canevas, le petit montant vers x=165–220, y=470–585 est nettement plus brillant dans raw et moins complet dans snapped. Son intensité n’est pas un repère stable. Le groupe central ne s’étend pas en une suite lisible grâce aux nouveaux éléments de raw.

## Contexte et provenance contrôlés

Les masques `.context_valid.npy` et `.source_valid.npy` sont respectivement identiques pixel par pixel entre raw et snapped pour les deux ordres. Le contexte valide reste 66,149 % du canevas ; la présence source reste 98,967 %. Le rectangle central (530,180,840,580) reste entièrement valide. La présence source n’est pas une certification d’appartenance à la même feuille.

Les quatre inférences partagent le même hash de poids, le pas 96, l’absence de mélange des fenêtres, la forme d’entrée 16 × 801 × 1061 et le noyau [20,781,20,1041]. Les ordres de profondeur sont respectivement 23→8 et 8→23. Les raccords déjà mesurés sur snapped sont visuellement présents dans le montage comparatif. Le raccordement des prédictions doit donc encore être corrigé avant une interprétation des connexions fines.

Le montage central reste une vue des mêmes données ; il ne crée aucune observation indépendante et n’augmente pas la résolution.

## Points précis à suivre dans le test stride48/Hann annoncé

Ces repères sont fixés avant toute vue des nouvelles sorties. Coordonnées du canevas de prédiction d’origine :

| Repère | Zone xyxy approximative | Question purement morphologique |
|---|---|---|
| Montant central gauche | (550,270,650,430) | Sa longueur, ses extrémités et sa séparation du voisin survivent-elles aux nouveaux raccords ? |
| Liaison supérieure du groupe reverse16 | (650,200,840,350) | La connexion présente surtout dans raw demeure-t-elle, ou résultait-elle de limites de fenêtres ? |
| Prolongement inférieur droit | (720,340,800,470) | Le prolongement plus marqué dans snapped demeure-t-il comme élément distinct ? |
| Deux masses sous le groupe | (650,490,840,580) | Restent-elles séparées et stables, sans acquérir artificiellement une barre de liaison ? |

Ces quatre zones sont des **repères de contrôle de morphologie**, sans étiquette alphabétique. Si une forme persiste après correction des raccords, elle devra encore être distinguée des limites de matière vues dans le CT. L’expérience ne doit pas être jugée selon sa ressemblance avec une lettre attendue.

## Conclusion de cette troisième étape

La concordance raw/snapped justifie de continuer le contrôle ciblé du groupe central reverse16. Elle n’améliore pas, à ce stade, ma confiance dans l’identification d’un caractère particulier. Aucune transcription, lettre validée ou découverte nouvelle n’est revendiquée.
