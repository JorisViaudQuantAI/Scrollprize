# Recherche visuelle de lettres sur papyrus

## Campagne locale du 24 novembre 2026

**Statut : aucune nouvelle lettre ni séquence de texte validée dans cette passe.**

Cette campagne répond à la demande de participer directement à une découverte : les images ont été examinées personnellement par l’agent principal, des régions ont été choisies pour leur intérêt visuel, de nouvelles prédictions ont été calculées sur du CT réel, et les formes suspectes ont été confrontées aux profondeurs, aux masques et à une lecture indépendante. Une hypothèse de forme n’est pas transformée en transcription.

Le résultat comprend une piste visuelle encore ambiguë sur PHerc0800, un examen neuf d’une fenêtre de PHerc0125, un raffinement du rendu sur cette dernière, et une résolution numérique très forte du repère d’un volume de surface public. Ce dossier public partage les observations et leurs limites, sur demande explicite de l’utilisateur. Aucune nouvelle lettre ni transcription n’est revendiquée.

## 1. Ce qui a réellement été examiné

| Périmètre | Travail effectué | Portée |
|---|---|---|
| PHerc0800, six surfaces publiques A–F | Lecture de 18 cartes existantes : ink9um, reader-v3 et Argus-v3 ; vues natives et détails | Ces surfaces se recouvrent largement en 3D. Elles ne sont pas six régions indépendantes. |
| PHerc0800, régions A et C | Acquisition de plusieurs profondeurs CT, distinction entre source manquante et valeur stockée nulle | La présence de matière ne certifie pas une même feuille anatomique. |
| PHerc0800 C, deux fenêtres 512² et une reprise avec contexte supplémentaire | Quatre cartes Studio initiales, puis une carte avec 96 pixels CT réels ajoutés à gauche | Les deux premières fenêtres se recouvrent sur 80 224 pixels, soit 30,60 % de chacune. La reprise réutilise leur région. |
| PHerc0125, fenêtre du maillage z14144_w025 | Nouveau rendu CT natif, géométrie brute et géométrie recalée, puis quatre cartes Studio | Fenêtre choisie sur la géométrie avant consultation du CT ou des scores d’encre. |
| PHerc0125, groupe central | Deux recalculs avec pas 48 et mélange Hann, sur une fenêtre fixée après la lecture initiale | Raffinement exploratoire ; le recadrage change aussi la phase des fenêtres. |
| Contrôle public PHerc0841 | Un tile puis une mosaïque avec les deux formes déjà annotées | Contrôle connu, jamais compté comme découverte nouvelle. |

Au total, les reçus enregistrent **13 cartes locales**, **1078 fenêtres modèle** et **13 contrôles d’identité supplémentaires**, soit **1091 passages modèle**. Les 18 cartes publiques PHerc0800 ont été téléchargées et inspectées ; elles ne sont pas comptées comme des calculs nouveaux. Tous les nouveaux modèles ont été exécutés localement sur CPU, en FP32, avec deux processus de quatre threads au maximum.

Ce périmètre ne constitue pas un examen de tous les papyrus ni de toutes les surfaces des deux rouleaux. La fenêtre PHerc0125 mesure 801 × 1061 pixels avec son halo ; son noyau choisi mesure 761 × 1021 pixels. Les deux cadrages PHerc0800 initiaux couvrent une union de 444 064 pixels du même canevas ; l’extension ajoute 49 152 pixels de contexte. Les ordres de profondeur ne sont pas additionnés comme des surfaces nouvelles.

## 2. PHerc0800 : une forme intéressante, encore insuffisante pour lire une lettre

### Sélection visuelle

Les cartes C ont retenu l’attention de l’agent principal avant les calculs nouveaux : plusieurs traverses, des montants et des compartiments sont visibles dans la moitié droite. Le lecteur indépendant a également classé C en tête de ses priorités, sans lire les observations de l’agent principal. Les identités des modèles étaient visibles dans les planches ; cette indépendance porte sur les interprétations des lecteurs, pas sur l’identité des données.

Les cartes A montrent de grandes bandes amalgamées. B possède des jonctions anguleuses mais aucune séquence isolable. D, E et F montrent surtout des masses ou des fragments isolés. Les motifs répétitifs sur les bords de tuiles, hors empreinte du papyrus, ont été exclus des candidats.

### Deux cadrages fixés et de nouvelles prédictions

La fenêtre centrale couvre `y794:1306, x1194:1706` du canevas C. Un second cadrage, `y650:1162, x900:1412`, couvre davantage le groupe de compartiments à gauche. Ce second choix est explicitement exploratoire et visuel ; il est antérieur à ses sorties Studio.

Pour chaque fenêtre, les entrées contiennent les mêmes 16 plans, soit `k23→k08`, soit leur inversion exacte `k08→k23`. Les pixels sont des copies uint8 du volume de surface public, sans interpolation supplémentaire ni normalisation préalable. Les masques source sont complets. Un reçu séparé documente la conversion des masques centraux uint8 en bool.

L’ordre `k23→k08` produit une activation très large, qui fusionne plusieurs traits. L’ordre inverse restitue plus nettement, dans la fenêtre gauche, un montant relié à trois traverses ou compartiments. Une ressemblance faible avec une forme de type epsilon après miroir a été notée dans le journal personnel, **sans attribution de caractère**. Les autres orientations exactes ont également été conservées. Une image miroir permet d’examiner une hypothèse d’orientation ; elle ne confirme pas la lecture.

Le CT offre une explication concurrente : les formes s’appuient sur un réseau de limites matérielles, de fibres et de zones sombres qui changent avec la profondeur. La grande bande sombre supérieure s’élargit sensiblement entre les plans 15 et 23. Cette observation fragilise une lecture directe, sans démontrer l’absence d’encre : les contrôles positifs comportent eux aussi des structures fibreuses et sombres.

Plusieurs extrémités du motif gauche sont hors du masque de contexte strict des nouvelles cartes. Elles disposent de valeurs CT, mais certaines fenêtres modèle contributrices utilisent du contexte réfléchi au bord du crop. Le contour entier ne peut donc pas être présenté comme un glyphe robuste. Les cartes officielles reader-v3 et Argus-v3 sont morphologiquement proches ; leur accord n’a pas été assimilé à deux preuves indépendantes.

Le lecteur indépendant a proposé une hypothèse individuelle faible : une boucle compatible avec une forme de type O, dans la boîte locale `[32,176,336,416]` du cadrage gauche. Il a gelé cette observation avant de comparer les recadrages officiels. Sa description et celle de l’agent principal ne convergent pas vers un caractère identifié : elles décrivent des parties d’une structure connectée. Les seize profondeurs ont ensuite été examinées séparément par les deux lecteurs. Elles renforcent l’alternative d’une frontière matérielle ; aucun tracé d’encre fermé distinct n’a été suivi avec assurance.

### Vérification du contour avec du vrai contexte supplémentaire

Une entrée élargie de 96 pixels à gauche a été acquise : `y650:1162, x804:1412`, soit 512 × 608 pixels et 16 plans dans l’ordre `k08→k23`. Le bloc initial est identique bit à bit. Les nouvelles requêtes ont reçu 2 625 108 octets, sans chunk manquant. Le décalage exact de 96 pixels conserve la phase de la grille s96.

La boîte proposée par le lecteur indépendant, `x932:1236, y826:1066`, contient 72 960 pixels. Elle passe de 78,95 % à 100 % de contexte strict complet. Cette reprise teste la dépendance du contour au bord de l’entrée, sans modifier modèle, ordre, intensités ou emplacement physique de la région.

**La boucle et sa fermeture gauche persistent avec ce contexte réel.** L’agent principal et le lecteur indépendant ont gelé leurs observations avant échange de leurs conclusions. La réserve technique du bord est levée pour la boîte fixée, sans promotion en glyphe robuste.

Le contrôle de phase est particulièrement utile : les 416 colonnes de l’ancien cadre qui n’avaient pas besoin du nouveau contexte restent identiques bit à bit. Dans la boîte candidate, 57 600 pixels auparavant admissibles ne changent donc pas. Les 15 360 pixels du bord gauche nouvellement admissible constituent la nouvelle vérification ; leur différence absolue moyenne de probabilité est 0,042616. Il serait incorrect de présenter les 72 960 pixels comme une seconde observation indépendante.

Le lecteur conserve finalement **une silhouette de type O faible, mieux documentée, sans lettre validée et sans séquence**. L’attachement à une structure plus grande, l’explication matérielle des CT et les raccords encore présents dans cette carte s96 empêchent une conclusion plus forte.

**Conclusion visuelle pour C :** conserver une piste morphologique précise et ses alternatives explicatives, sans proposer de mot ou de transcription. Les deux fenêtres et leurs ordres ne sont pas des réplications indépendantes.

## 3. PHerc0125 : nouveau rendu, quatre cartes initiales, puis correction des raccords

### Choix et rendu

Le maillage public `z14144_w025` a été sélectionné après comparaison au travail déjà visible dans le dépôt et aux tâches annoncées. À l’état du dépôt examiné (`5ce5fde`), son nom ne figurait pas dans les 15 rendus exécutés ni parmi les trois cibles Tier1B revendiquées. Cela réduit un doublon de cible nommée ; une intersection physique avec d’autres surfaces n’a pas été exclue.

Le maillage est déjà aplati selon ses métadonnées. Aucun second aplatissement n’a été appliqué automatiquement. La fenêtre a été choisie à partir de la géométrie, avant lecture du CT, du volume m7 ou d’une prédiction d’encre.

Le CT natif est uint8, au pas de 9,362 µm. Le renderer a été appelé explicitement avec `strict_uint8=True`. Deux géométries ont été conservées : brute, puis recalée par deux passes utilisant m7. La hausse du score m7 après recalage n’est pas une validation indépendante puisque ce même signal pilote le déplacement.

Trente et un plans ont été rendus, puis une bande de 16 plans `23…8` et son inversion exacte `8…23` ont été préparées dans chaque géométrie. Les ordres partagent les mêmes échantillons au sein d’une géométrie. Les géométries brute et recalée échantillonnent des positions physiques différentes.

Le premier essai a atteint son plafond de téléchargement. Le budget a été augmenté avant toute lecture des sorties d’encre, sans changer la fenêtre ni les quatre conditions. Les sorties partielles et l’essai de récupération du cache sont conservés comme diagnostics ; seuls les fichiers `0125-w025-v2` alimentent les inférences principales.

### Lecture des quatre cartes s96

Les cartes descendantes montrent de grandes masses isolées, dont une forme triangulaire centrale et une masse anguleuse à droite. Les cartes inversées comportent un montant central, une liaison supérieure et des fragments sous-jacents. Certaines formes persistent entre les géométries brute et recalée, mais leurs jonctions et prolongements varient.

L’agent principal a examiné les sorties natives, les détails, les miroirs horizontaux et des rotations exactes, ainsi que des affichages à probabilité linéaire, logit fixe et seuil fixe. Aucun affichage ne permet une séquence de lettres défendable. Le lecteur indépendant a gelé séparément sa lecture des prédictions, sa confrontation au CT, puis sa comparaison des géométries.

Le groupe central de l’ordre inverse, `x530:840, y180:580`, possède un contexte complet. Le grand groupe droit de l’ordre descendant n’en possède que 64,06 % sur la boîte évaluée ; son contour entier a été écarté comme candidat robuste. Les PNG CT de revue initiaux étaient recadrés au noyau : leur correspondance avec le plein canevas de prédiction nécessite de retirer 20 pixels sur chaque axe. Les comparaisons finales et l’atlas utilisent les tableaux source pour éviter cette ambiguïté.

### Risque concret des raccords

La lecture indépendante a repéré et mesuré des discontinuités sur la grille de pas 96. Dans les paires de pixels à contexte complet des cartes snapped, les sauts moyens atteignent environ 17 à 26 niveaux sur 255 aux raccords, contre environ 0,65 à 0,70 ailleurs. Ces chiffres décrivent les raccords de prédiction ; ils ne mesurent pas l’encre.

Pour vérifier les formes après réduction de ce défaut, deux nouvelles entrées ont été copiées sans modification depuis l’ordre inverse, avec une fenêtre fixe `y80:688, x432:944` de 608 × 512 pixels. Les deux géométries ont été recalculées à pas 48 avec mélange Hann. Les 124 000 pixels du groupe suspect disposent tous d’un contexte strict complet dans cette configuration et dans la comparaison au plein champ initial.

Le recalcul rend les contours nettement plus continus. Sur le groupe fixé et les mêmes 2 750 paires de pixels traversant les anciens raccords, le saut moyen passe de **34,04 à 1,08 niveaux sur 255** en géométrie brute, et de **33,82 à 1,01** après recalage. Ailleurs, les différences moyennes restent proches de 1 niveau. Ces mesures utilisent les previews à échelle fixe, sans test d’hypothèse statistique.

Le montant gauche, la liaison supérieure et les masses inférieures persistent. La liaison supérieure reste plus fusionnée dans raw ; un prolongement inférieur ressort davantage dans snapped. Les deux lecteurs ne retiennent **aucune lettre nommée ni glyphe robuste** dans ces sorties plus continues. Le rendu a été amélioré, mais l’origine des formes reste ambiguë face aux frontières et croisements de fibres du CT.

Le changement de crop modifie aussi la phase de grille. Une éventuelle amélioration ne peut donc pas être attribuée aux seuls poids Hann. Cette expérience sert à réexaminer une morphologie avec un rendu plus continu ; elle n’est pas un test statistique indépendant d’une hypothèse de lettre.

## 4. Résultat technique solide : retrouver le repère réel du volume PHerc0800 C

La comparaison aux profondeurs CT aurait peu de valeur si les pixels affichés ne correspondaient pas aux bonnes positions 3D. Un sous-agent a donc repris le CT natif et le maillage public, puis essayé des conventions géométriques définies avant les acquisitions et les comparaisons correspondantes.

La recette qui reproduit presque exactement le volume public utilise :

1. une origine au demi-pixel : `gy=(y+0,5)×0,05`, `gx=(x+0,5)×0,05` ;
2. une position interpolée bilinéairement dans le maillage ;
3. une normale centrale calculée au sommet le plus proche, sans interpolation de cette normale ;
4. le point d’échantillonnage `X(k)=P−(k−15)N` ;
5. l’interpolation trilinéaire du CT natif, puis la troncature en uint8.

Cette convention correspond au code historique consulté, dont la normale diffère de celle du renderer bicubique actuel. La date de modification de l’objet S3 a seulement guidé le choix d’une version historique plausible ; elle n’est pas présentée comme une date de calcul certifiée.

Sur six petites zones de 9 × 9 pixels et 16 profondeurs, **7 744 des 7 776 valeurs** sont reproduites exactement. Les 32 différences restantes valent un seul niveau uint8. L’agent principal a recalculé ces comptes à partir des comparaisons individuelles et vérifié les identités de troncature et de différence signée.

Trois zones supplémentaires, distinctes de celles de l’exploration initiale, ont été fixées avant leur comparaison numérique. Sur ces zones :

| Convention | Erreur absolue moyenne uint8 | Valeurs exactement égales |
|---|---:|---:|
| Recette historique, float64 | 0,00386 | 99,6142 % |
| Même recette, origine zéro | 3,54475 | 9,6965 % |
| Même recette, signe opposé | 27,76440 | 7,0988 % |
| Même recette, arrondi au plus proche | 0,48868 | 51,1317 % |
| Géométrie bicubique actuelle | 0,83745 | 45,8333 % |

Le simple coefficient de corrélation aurait été trompeur : l’origine zéro conserve une corrélation de 0,98117 malgré une erreur nettement plus grande. Le contrôle porte sur les valeurs et les coordonnées, pas sur la ressemblance d’une image.

Cette vérification soutient fortement le repère et le sens des plans **aux points testés**. Elle ne prouve ni le hash du renderer historique, ni l’identité globale de tous les sommets, ni le suivi d’une même feuille anatomique, ni l’alignement exact de tous les TIFF d’encre. Elle ne constitue pas une découverte de texte. Le coût supplémentaire de ce replay est de 17 053 061 octets reçus ; aucun modèle d’encre n’y est exécuté.

## 5. Contrôles et limites de la chaîne d’inférence

Chaque job doit d’abord reproduire exactement une sortie historique sur un patch public PHerc0211 de 128². Les 13 sentinelles de cette campagne ont réussi l’égalité bit à bit. Le modèle, son code source et ses poids sont épinglés ; les entrées restent uint8 et le modèle applique une seule fois `clip(0,200)/255` en FP32.

Les fenêtres de prédiction utilisent une entrée 128², dont le centre 96² est conservé. Le pas 96 ne mélange pas les sorties ; le pas 48 utilise le mélange Hann du code de référence. Le masque `valid` du modèle, le support des 16 plans source et le contexte complet sont sauvegardés séparément. Aucun de ces masques ne certifie l’identité anatomique d’une feuille.

Le contrôle public PHerc0841 a été élargi d’un tile partiel à une mosaïque 768 × 1024 copiée depuis six tiles adjacents. Son recouvrement avec le tile antérieur est exactement identique. Les tiles étant jointifs sans recouvrement, aucun contrôle des joints par pixels communs n’est revendiqué.

La mosaïque contient les 67 113 pixels d’encre et 111 310 pixels de fond annotés du cache ; tous ont un contexte complet. Les 608 009 pixels inconnus sont exclus de l’évaluation. Sur ces annotations connues, la probabilité moyenne est 0,609 sur l’encre et 0,302 sur le fond ; l’AUC est 0,8332 et l’AP 0,7882. Ces mesures sont des diagnostics de restitution sur un contrôle public connu, pas une validation indépendante ni un résultat de découverte.

Visuellement, les sorties de ce contrôle restent fragmentaires et diffuses par rapport aux annotations. Les deux formes connues ne deviennent pas une ligne de texte proprement lisible. Cette limite interdit de conclure à l’absence d’encre à partir d’une carte peu lisible sur un nouveau papyrus. Un panneau publié PHerc0211 montrant des lettres connues plus nettes a également servi de référence visuelle ; ses annotations étaient visibles.

Une erreur d’orchestration est survenue après les deux premiers jobs : PyTorch refuse de régler une seconde fois le nombre de threads interop dans le même processus. Les sorties terminées ont été conservées. Les deux jobs bruts ont été relancés dans des processus neufs ; aucun passage modèle brut n’avait eu lieu avant cet échec. Les snapshots des deux versions du wrapper correspondent exactement aux hashes consignés dans leurs reçus.

## 6. Dossier public et vérification

Cette publication compacte conserve sept figures originales, les lectures figées, les reçus expurgés, les comparaisons numériques du replay, des crops exacts de probabilités et de masques, et les scores de tous les pixels annotés du contrôle PHerc0841. Les figures ne contiennent pas de tracé de lettre ajouté ; les contours rouges du contrôle 0841 proviennent exclusivement des annotations publiques connues.

Le script `verify_evidence.py` recalcule les métriques sur les valeurs publiées, sans modèle, GPU ni réseau. Il contrôle aussi les empreintes de chaque fichier public. Les SHA-256 des originaux figurent séparément des SHA-256 publiés lorsqu’un reçu a été expurgé ou un document adapté. Le dossier original est la source de cette extraction ; la publication ne prétend pas que les fichiers expurgés sont identiques bit à bit à leurs originaux.

Le paquet ne contient pas les volumes CT complets, les cartes entières, l’atlas HTML, les gros caches ni le checkpoint de 893 MB. Les vérifications autonomes portent donc sur les valeurs incluses. La stabilité des 57 600 pixels déjà admissibles du rectangle C est recalculable ; l’identité des 416 colonnes du plein cadre reste une observation documentée dans le reçu original. Refaire les acquisitions et les inférences demanderait les sources épinglées et des téléchargements supplémentaires ; ce vérificateur ne les réexécute pas.

`build_evidence.py` documente et reproduit l’extraction depuis le dossier complet original. Le [README](README.md) donne la commande de vérification ; [PROVENANCE.md](PROVENANCE.md) décrit les périmètres et les liens vers les sources.

### Sources principales

- Dépôt de coordination étudié : <https://github.com/JorisViaudQuantAI/Scrollprize>, état d’intégration `5ce5fde` au début de cette passe ; fil <https://github.com/JorisViaudQuantAI/Scrollprize/issues/1>.
- Modèle public : <https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt>, commit `bf66f753176792a966bca17822629fb8821f20c8`.
- Poids `studio-softloss-20261023-ema.safetensors`, SHA-256 `96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90`.
- Maillages PHerc0125 : <https://github.com/pscamillo/vesuvius-eligible-meshes>, commit `66000516c97e043b696ce874f145e29638109c91`.
- Géométrie historique : <https://github.com/ScrollPrize/villa/blob/4386f625e1646dd67b6ce27597be434877fb96d7/volume-cartographer/core/src/QuadSurface.cpp> ; code source téléchargé, épinglé et conservé dans le dossier du replay.
- Tutoriel officiel de la chaîne de lecture et de l’inspection visuelle, page archivée : <https://scrollprize.org/tutorial>. Les conventions réellement utilisées ici proviennent des sources épinglées et des expériences décrites, pas d’une supposition de compatibilité avec toutes les versions actuelles.
- Les URL exactes des objets CT, TIFF, maillages, ETag, réponses et empreintes figurent dans les reçus de chaque acquisition.

## 7. Décision scientifique à l’issue de cette passe

La piste la plus précisément documentée est le contour gauche de PHerc0800 C : sa fermeture résiste au remplacement du contexte réfléchi par du CT réel, mais sa ressemblance avec un O reste faible et n’est pas séparée de l’explication matérielle. C’est un candidat de travail non confirmé, sans découverte de texte établie. Sa forme était déjà présente dans certaines prédictions publiques ; l’apport nouveau de cette campagne est sa sélection argumentée, les calculs locaux, les vérifications de contexte et la confrontation aux profondeurs.

PHerc0125 apporte un nouveau rendu et une comparaison propre des géométries et des ordres. Le raffinement a levé un défaut de raccordement réel, sans faire apparaître de glyphe défendable dans la fenêtre étudiée. Cette absence de lecture n’est pas assimilée à une absence d’encre.

Les contributions réutilisables sont le replay du repère historique, les entrées et masques correctement distingués, les cartes conservées pour chaque condition, la comparaison de contexte à phase inchangée, les lectures gelées et le dossier de reproduction. Le prochain contrôle discriminant du candidat C serait une vérification anatomique de la feuille, accompagnée d’un rendu continu sur davantage de contexte de ligne. Le contrôle PHerc0841 rappelle qu’il faut aussi distinguer une limite du modèle d’une limite des données.

Le travail suivant doit partir de ces images et de ces limites documentées : conserver les candidats comme hypothèses, améliorer le contexte ou la géométrie lorsque le rendu limite la lecture, et rechercher une organisation de traits qui survive à ces vérifications. Une forte probabilité d’encre, une ressemblance isolée ou un accord de modèles apparent ne remplacent pas une lecture soutenue par les données.
