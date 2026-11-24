# Provenance et périmètre de la publication

## Deux niveaux de preuve

La campagne originale du 24 novembre 2026 a acquis du CT public, calculé treize cartes Studio, comparé des cadres fixes et conservé les sorties, les masques et les reçus. Ce dossier public en extrait les éléments utiles à la discussion scientifique et au triage industriel. La publication n’ajoute aucun forward modèle.

Le **vérificateur compact** contrôle les fichiers présents et recalcule des métriques à partir de valeurs enregistrées. Le **replay intégral** demanderait de récupérer les entrées, le modèle et certains caches exclus. Les deux opérations doivent être distinguées dans toute citation de ce dossier.

| Vérification autonome dans ce dossier | Ce qui reste une observation du reçu original |
|---|---|
| Treize reçus, leurs empreintes publiées, somme 1 078 + 13 = 1 091 | Nouvelle exécution des sentinelles et de chaque carte avec PyTorch. |
| Crops exacts de six cartes, leurs masques et leur rattachement aux hashes d’inférence | Carte complète et entrée CT complète de chaque job. |
| Identité des 57 600 pixels déjà admissibles de la boîte C ; effet sur les 15 360 nouveaux | Identité des 416 colonnes de tout l’ancien cadre. |
| Sauts aux mêmes 2 750 paires de pixels du groupe PHerc0125 | Toutes les autres régions des quatre cartes complètes. |
| AUC/AP sur tous les pixels déclarés connus par l’extraction, indices et labels conservés | Retour aux six tuiles CT originales et indépendance d’entraînement du contrôle. |
| Troncature des 7 776 valeurs interpolées conservées ; comparaison aux valeurs officielles | Nouvelle interpolation des sept chunks CT natifs avec le maillage entier. |

Les empreintes cryptographiques fixent des octets et facilitent la traçabilité. Elles ne prouvent pas la vérité physique des sources ni l’identité anatomique d’une feuille.

## Références épinglées

| Composant | Source et version |
|---|---|
| Coordination | [Scrollprize](https://github.com/JorisViaudQuantAI/Scrollprize), état d’intégration `5ce5fde` lors de la sélection initiale ; [fil des agents](https://github.com/JorisViaudQuantAI/Scrollprize/issues/1). |
| Modèle Studio | [vesuvius-studio-multiscroll-cpt](https://github.com/priyankaranke/vesuvius-studio-multiscroll-cpt/tree/bf66f753176792a966bca17822629fb8821f20c8), commit `bf66f753176792a966bca17822629fb8821f20c8`. |
| Poids | `studio-softloss-20261023-ema.safetensors`, SHA-256 `96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90`, 893 443 724 octets ; exclus de cette contribution. |
| Code modèle | `model.py`, SHA-256 `efbb390b6208c480fa8e8681137690b7df897c3e406956b9e804fa66645316d7`. |
| Chemin d’inférence | [studio_t4.py](../studio_t4/studio_t4.py), snapshot historique SHA-256 `348260f45e882dc3574b334d3c76aae42b2655d9b37354d26a0c57b24bda6d28`. La version actuellement consultée doit être comparée à ce hash avant replay. |
| Maillage PHerc0125 | [vesuvius-eligible-meshes](https://github.com/pscamillo/vesuvius-eligible-meshes/tree/66000516c97e043b696ce874f145e29638109c91), commit `66000516c97e043b696ce874f145e29638109c91`, cible `z14144_w025`. |
| CT PHerc0125 | [volume natif](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr/.zattrs), uint8, pas 9,362 µm. |
| Surface PHerc0800 C | [métadonnées du volume de surface](https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0800/segments/20251028220955-auto_grown_20251028220955262/surface-volumes/8.64um-1.2m-116keV-volume-20250521135224.zarr/.zattrs), segment `20251028220955-auto_grown_20251028220955262`, pas natif 8,64 µm. |
| Géométrie historique | [QuadSurface.cpp](https://github.com/ScrollPrize/villa/blob/4386f625e1646dd67b6ce27597be434877fb96d7/volume-cartographer/core/src/QuadSurface.cpp), commit `4386f625e1646dd67b6ce27597be434877fb96d7`. La date S3 a motivé une hypothèse de version ; elle ne certifie pas le binaire utilisé pour produire le volume. |
| Contrôle PHerc0841 | [cache auteur public](https://huggingface.co/datasets/paranke/vesuvius-814-841-9b-studio-training-data/tree/ad919d288ede1c4daf92725288bb6fe29c558591/donor-studio-resunet-scratch-20260926/prepared/841-expanded-layer-03-approved-r000001), révision `ad919d288ede1c4daf92725288bb6fe29c558591`. Les URL exactes et empreintes sont dans [0841_mosaic.json](receipts/acquisition/0841_mosaic.json). |

L’inférence historique utilise CPU FP32, batch un, `clip(0,200)/255` appliqué une fois, une entrée 128 × 128 et un centre conservé de 96 × 96. Le pas 96 ne mélange pas ; le pas 48 utilise Hann. Le wrapper v2 lance un processus neuf pour chaque job afin de ne pas réinitialiser les threads interop de PyTorch dans un processus déjà utilisé. Les SHA exacts des wrappers v1/v2 figurent dans chaque reçu.

## Recette géométrique documentée

Pour les coordonnées de surface `(y,x)` et l’indice de plan `k` :

```text
gy = (y + 0.5) * 0.05
gx = (x + 0.5) * 0.05
P  = interpolation bilinéaire des positions du maillage
N  = normalize(cross(tangente_colonne, tangente_ligne))
     au sommet le plus proche, par différences centrales
X(k) = P - (k - 15) * N
valeur = troncature uint8(interpolation trilinéaire du CT natif à X(k))
```

La normale est celle du sommet le plus proche, pas une normale interpolée. Le signe, l’origine demi-pixel et la troncature sont essentiels. Le CSV conserve la valeur interpolée float64 avant troncature, la valeur officielle et leur différence signée ; les profils NPZ conservent aussi les variantes de convention. Les six zones de 9 × 9 pixels ne représentent pas une validation de l’ensemble du maillage.

## Extraction des tableaux

[crop_manifest.json](data/crop_manifest.json) conserve le nom de chaque tableau source, son SHA-256 complet, la boîte découpée dans ce tableau, la boîte globale, la forme et le type. Les douze tableaux publiés sont six crops de probabilités float32 et les six masques de contexte bool correspondants. Aucune interpolation, quantification ou normalisation n’est appliquée pendant l’extraction. Les previews utilisées pour les sauts PHerc0125 sont recalculées par `numpy.rint(p * 255)` ; cette quantification a été comparée bit à bit aux quatre previews originales avant publication.

[0841_manifest.json](data/0841_manifest.json) décrit l’extraction des 178 423 pixels annotés. Les indices aplatis suivent l’ordre C d’une image de 768 × 1024. Les labels 1 et 2 signifient fond et encre ; les zéros inconnus sont exclus. Le tableau de scores est float32, les étiquettes uint8, les indices int32 et le contexte bool. Les calculs AUC et AP regroupent explicitement les scores ex æquo ; aucun seuil n’est sélectionné.

## Empreintes originales et fichiers publics

[manifest.json](manifest.json) distingue deux ensembles :

- `files` décrit les **octets publics réels**, leur taille et leur SHA-256. Le manifeste s’exclut lui-même pour éviter une empreinte circulaire.
- `origins` décrit les **fichiers historiques sources** et leur transformation éventuelle. Pour une copie exacte, le vérificateur exige l’égalité des empreintes. Pour un document adapté ou un JSON expurgé, l’empreinte originale est conservée mais n’est pas présentée comme celle du fichier public.

Les reçus remplacent les chemins du poste de calcul par `artifact://campaign/…` ou `artifact://workspace/…`, y compris les chemins apparaissant comme clés JSON. Ces identifiants désignent des artefacts historiques. Ils ne prétendent pas que les fichiers correspondants sont contenus dans ce paquet. Les liens des treize reçus dans le résumé de campagne sont rebâtis, avec les champs distincts `historical_receipt_sha256` et `published_receipt_sha256`.

Les sept figures sont copiées bit à bit. Les lectures figées sont également conservées telles quelles ; une mention de dossier privé décrit leur contexte historique de rédaction. Le rapport complet est adapté pour remplacer l’ancien statut de diffusion et la description de l’archive intégrale par le périmètre public réel. Sa source comptait 3 542 mots selon `str.split()`.

## Reproduire l’extraction depuis la campagne complète

Le détenteur du dossier original complet peut régénérer les preuves dérivées :

```bash
python3 firstletters-colab/visual_discovery_20261124/build_evidence.py /chemin/vers/campagne_originale
python3 firstletters-colab/visual_discovery_20261124/verify_evidence.py
```

L’exporteur vérifie les treize empreintes de reçus originaux et les sorties déclarées avant d’extraire les crops. Il recopie les figures, expurge les reçus, calcule les tableaux numériques et rafraîchit le manifeste. Il n’effectue aucun téléchargement ni forward. Le dossier original doit être distinct du dossier public.

Le code d’export et le vérificateur sont des outils ajoutés pour cette publication. Ils ne sont pas les scripts qui ont produit les sorties d’inférence historiques. Aucun fichier du dossier d’origine n’est modifié.
