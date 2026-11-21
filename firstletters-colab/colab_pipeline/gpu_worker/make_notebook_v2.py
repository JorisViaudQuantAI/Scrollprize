"""Writes vesuvius_gpu_worker_v2.ipynb: French instructions, nvidia-smi check, and the stable bootstrap.

The bootstrap loads the highest worker_vNNN.py from the Drive root and runs it; the worker itself is
never inside the notebook, so updating it never needs a new notebook.
Usage: python make_notebook_v2.py --root-id ID --jobs-id ID --results-id ID --inputs-id ID [--out NOTEBOOK]
       (the four ids are those of your own Drive folders vesuvius-gpu, vesuvius-gpu/jobs, /results, /inputs;
        without them the notebook keeps <...> placeholders and the worker stops with an error)
The test suite execs BOOTSTRAP with a __name__ other than "__main__" to exercise lancer_ouvrier locally.
"""
import json
import os

INSTRUCTIONS = """# Vesuvius — ouvrier GPU (version 2)

Ce notebook transforme cette session Colab en « ouvrier » : il surveille le dossier Google Drive `vesuvius-gpu/jobs`, exécute sur le GPU les scripts qui y sont déposés, et renvoie les résultats dans `vesuvius-gpu/results`.

Cette version se met à jour toute seule : vous n'aurez plus besoin de remplacer ce notebook.

**Mode d'emploi**

1. Menu **Exécution → Modifier le type d'exécution** : vérifiez qu'un **GPU** est sélectionné (T4 suffit pour commencer ; L4 ou A100 si vous avez des unités de calcul).
2. Menu **Exécution → Tout exécuter**.
3. Acceptez la demande d'accès à Google Drive quand elle apparaît (elle sert uniquement à lire les scripts et à écrire les résultats).
4. Laissez cet onglet ouvert. La dernière cellule tourne en continu : c'est normal.

Si la session s'est arrêtée, refaites simplement **Exécution → Tout exécuter** : le travail déjà terminé n'est pas refait.

Si un message dit qu'un autre ouvrier est encore actif : menu **Exécution → Gérer les sessions**, arrêtez l'autre session, puis refaites **Exécution → Tout exécuter**.

Pour arrêter : menu **Exécution → Interrompre l'exécution**, ou fermez l'onglet.
"""

GPU_CHECK = """!nvidia-smi || echo 'AUCUN GPU : choisissez un GPU dans Exécution > Modifier le type d execution, puis relancez.'
!df -h /content | tail -1
"""

BOOTSTRAP = '''# Amorce : charge depuis Google Drive la dernière version de l'ouvrier (worker_vNNN.py) et la lance.
ROOT_ID = "<ROOT_FOLDER_ID>"     # dossier Drive « vesuvius-gpu »
JOBS_ID = "<JOBS_FOLDER_ID>"     # vesuvius-gpu/jobs
RESULTS_ID = "<RESULTS_FOLDER_ID>"  # vesuvius-gpu/results
INPUTS_ID = "<INPUTS_FOLDER_ID>"   # vesuvius-gpu/inputs
import re, time, traceback

def lancer_ouvrier(lister, lire, ids, pause=60, **opts):
    casses, echecs = set(), {}
    while True:
        v = None
        try:
            fichiers = {int(m.group(1)): f["id"] for f in lister() if (m := re.fullmatch(r"worker_v(\\d{3})\\.py", f["name"]))}
            bons = sorted(k for k in fichiers if k not in casses)
            if not bons:
                print("ARRÊT : aucun fichier worker_vNNN.py utilisable dans le dossier vesuvius-gpu de Google Drive. "
                      "Prévenez la personne qui vous a confié ce notebook.")
                return None
            v = bons[-1]
            ns = {"__name__": "gpu_worker", "LOADED_VERSION": v, "SKIP_VERSIONS": set(casses)}
            exec(compile(lire(fichiers[v]), f"worker_v{v:03d}.py", "exec"), ns)
            r = ns["main"](*ids, **opts)
            if r != "update":
                return r
        except KeyboardInterrupt:
            raise
        except Exception:
            traceback.print_exc()
            echecs[v] = echecs.get(v, 0) + 1
            if v is not None and echecs[v] >= 3:
                casses.add(v)
            print("Erreur (sans doute passagère) : nouvel essai dans une minute. Rien à faire.")
            time.sleep(pause)

if __name__ == "__main__":
    from google.colab import auth
    from googleapiclient.discovery import build
    auth.authenticate_user()
    drive = build("drive", "v3", cache_discovery=False)
    lister = lambda: drive.files().list(q=f"'{ROOT_ID}' in parents and trashed=false",
                                        fields="files(id,name)", pageSize=1000).execute()["files"]
    lire = lambda i: drive.files().get_media(fileId=i).execute().decode("utf-8")
    lancer_ouvrier(lister, lire, (ROOT_ID, JOBS_ID, RESULTS_ID, INPUTS_ID))
'''


def bootstrap(ids=None):
    """BOOTSTRAP with the placeholders replaced by real Drive folder ids (dict with root/jobs/results/inputs)."""
    b = BOOTSTRAP
    for key, ph in (("root", "<ROOT_FOLDER_ID>"), ("jobs", "<JOBS_FOLDER_ID>"),
                    ("results", "<RESULTS_FOLDER_ID>"), ("inputs", "<INPUTS_FOLDER_ID>")):
        if ids and ids.get(key):
            b = b.replace(ph, ids[key])
    return b


def cell(kind, src):
    c = {"cell_type": kind, "metadata": {}, "source": src.splitlines(keepends=True)}
    if kind == "code":
        c.update(execution_count=None, outputs=[])
    return c


def notebook(ids=None):
    return {
        "cells": [cell("markdown", INSTRUCTIONS), cell("code", GPU_CHECK), cell("code", bootstrap(ids))],
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Write the Colab worker notebook (bootstrap + instructions).")
    for k in ("root", "jobs", "results", "inputs"):
        ap.add_argument(f"--{k}-id", help=f"Google Drive folder id of vesuvius-gpu{'' if k == 'root' else '/' + k}")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "vesuvius_gpu_worker.ipynb"))
    a = ap.parse_args()
    ids = {k: getattr(a, f"{k}_id") for k in ("root", "jobs", "results", "inputs")}
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(notebook(ids), f, ensure_ascii=False, indent=1)
        f.write("\n")
    missing = [k for k, v in ids.items() if not v]
    print("wrote", a.out, "- bootstrap lines:", len(BOOTSTRAP.splitlines()),
          ("- WARNING: placeholders left for " + ", ".join(missing)) if missing else "")
