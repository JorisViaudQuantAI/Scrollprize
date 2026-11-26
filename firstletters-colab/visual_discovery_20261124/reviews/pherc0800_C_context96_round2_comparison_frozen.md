Figé le 2026-11-24T11:47:55.597258+00:00 (UTC).

# PHerc0800 C — test final de contexte réel sur la boucle fixée

**Comparaison gelée après lecture séparée de la nouvelle carte. Dernière étape de la campagne visuelle.**

## Étapes préservées et comparaison exacte

La nouvelle carte et son masque ont été examinés seuls puis figés dans `pherc0800_C_context96_round1_frozen.md`, SHA-256 `48eccc774bffdb0a50b8b4a54ee5702fb6a5af72260bf5e26a9761991b85001d`. Ce premier avis est inchangé.

L’ancienne prédiction a ensuite été rouverte. La correspondance utilisée est exactement `nouvelle[:,96:608] ↔ ancienne[:, :]`. Le rectangle ancien (32,176,336,416) devient (128,176,432,416) dans la nouvelle carte. Les pixels ont été affichés côte à côte sans rééchantillonnage, modification de contraste ou tracé de lettre. Aucun avis root ni JSON de root n’a été consulté pour cette comparaison.

## Résultat visuel

**Le contour en boucle persiste. Il ne se rompt pas après l’ajout de contexte réel.** Dans la nouvelle carte, l’intérieur sombre ovalaire reste entouré d’une région claire par les quatre côtés. Les variations visibles se concentrent dans le côté gauche ; elles ne suppriment pas la fermeture apparente à cette résolution. Les traverses supérieure et inférieure ainsi que le côté droit conservent leur organisation.

La boucle demeure épaisse et attachée à un réseau de montants et de compartiments. La stabilisation de son côté gauche ne suffit pas à décider qu’il s’agit d’un caractère individuel. L’ensemble n’est pas devenu une séquence lisible et aucun mot n’a été proposé.

## Ce que le nouveau calcul a effectivement testé

Le masque confirme que les **72 960 pixels** du rectangle sont maintenant admissibles. Le calcul direct des différences entre fichiers de probabilité, dans le repère commun, donne :

| Partie du rectangle fixé | Pixels | Résultat numérique |
|---|---:|---|
| Région déjà valide dans l’ancien cadrage | 57 600 | Valeurs strictement identiques bit à bit. |
| Bande gauche nouvellement valide, ancien x=32:96 et y=176:416 | 15 360 | Changement absolu moyen de probabilité 0,042616 ; maximum 0,180886. |
| Rectangle complet | 72 960 | Changement absolu moyen 0,008972. |

Ces valeurs ne mesurent pas la probabilité qu’un glyphe soit authentique. Elles localisent l’effet du contexte dans les sorties du modèle. La partie déjà valide n’apporte pas une nouvelle confirmation indépendante, puisqu’elle est identique numériquement. **L’information nouvelle est que la fermeture proposée reste visible dans la bande gauche lorsque celle-ci est évaluée avec un contexte admissible réel.**

Le pas 96 et sa phase ont été conservés, selon le protocole annoncé, avec le même ordre des profondeurs et la même intensité. Je ne déduis pas de l’identité de certaines valeurs une identité anatomique de feuille ni une origine chimique du motif.

## Mise à jour du niveau d’hypothèse

| Niveau | Verdict final |
|---|---|
| Motif visuel | Boucle persistante, désormais entièrement dans le contexte complet du modèle. |
| Compatibilité avec une lettre individuelle | **Forme de type O, hypothèse faible mieux documentée.** La réserve du bord gauche hors contexte est levée. |
| Glyphe robuste | **Aucun.** L’attache à une structure plus grande et l’explication concurrente par la matière restent non résolues. |
| Séquence ou mot | **Aucun.** |

Le résultat améliore donc la solidité de l’observation de forme, pas suffisamment celle de l’attribution à une lettre. Il serait désormais inexact d’écarter ce contour au seul motif que son bord gauche manquerait de contexte ; il serait tout aussi inexact d’en faire une lettre confirmée du fait de sa stabilité.

## Pourquoi la réserve physique demeure

L’examen gelé des seize profondeurs avait montré une longue limite et des plages matérielles progressives qui dépassent le contour supposé. Les nouvelles prédictions ne changent pas ces observations CT. Une réponse stable du modèle peut suivre une structure stable du support. Les cartes officielles similaires ne sont pas présumées indépendantes de ces réponses.

Le CT seul ne prouve pas l’absence d’encre. L’hypothèse d’écriture reste ouverte, mais aucune preuve positive permettant de séparer l’encre de ces frontières matérielles n’a été établie au cours de cette campagne.

## État final du dossier visuel

La campagne laisse un candidat de silhouette précisément localisé et une chaîne de contrôles conservée : lecture initiale, comparaison officielle, série CT complète, extension de contexte réel et comparaison exacte. Aucun modèle ou acquisition supplémentaire n’a été exécuté par cette revue finale. Tous les rapports antérieurs restent inchangés ; le dossier PHerc0125 est séparé.

La figure `pherc0800_C_context96_views/fixed_loop_old_new_native.png` permet de revoir le même rectangle ancien/nouveau sans aucun trait ajouté. Le fichier `comparison_metrics.json` contient le mapping, la méthode et les différences numériques.
