import os
import json
from collections import defaultdict
from joblib import Parallel, delayed, parallel_backend
import cv2
from concurrent.futures import ThreadPoolExecutor

# Directory dei dati
BASE_DIR = "dataset"
TRAIN_DIR = os.path.join(BASE_DIR, "images_thermal_train")
VAL_DIR   = os.path.join(BASE_DIR, "images_thermal_val")

'''
Carica il file COCO JSON e restituisce il dizionario delle annotazioni.
split_dir: path alla cartella del split (train o val).
Un dizionario Python esattamente corrispondente alla struttura del COCO JSON, con i campi:
    - "images": lista di informazioni su ciascuna immagine (file_name, width, height, id)
    - "annotations": lista di annotazioni, ognuna con un image_id, una bbox nel formato [xmin, ymin, width, height] e un category_id
    - "categories": lista di classi disponibili con id e name
'''
def load_coco_annotations(split_dir):
    coco_path = os.path.join(split_dir, "coco.json")
    if not os.path.isfile(coco_path):
        raise FileNotFoundError(f"Annotations file not found: {coco_path}")
    with open(coco_path, 'r') as f:
        coco = json.load(f)
    return coco

def process_single_sample(img_info, annotation_per_image, split_dir):
    img_id = img_info['id']
    img_file = img_info['file_name']
    img_path = os.path.join(split_dir, img_file)

    if not os.path.isfile(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")

    bboxes = []
    labels = []
    for annotation in annotation_per_image[img_id]:
        x, y, w, h = annotation['bbox']
        bboxes.append([x, y, x + w, y + h])
        labels.append(annotation['category_id'])

    return {
        'img_path': img_path,
        'bboxes': bboxes,
        'labels': labels
    }

'''
Trasformare il dizionario COCO in una lista di “sample” pronta per il training/dataloader.
Per ciascuna immagine in coco["images"] recuperiamo le sue annotazioni, ne ricaviamo tutte le box e tutte le classi, poi creiamo un oggetto:
[{ 'img_path': ..., 'bboxes': [...], 'labels': [...] }, ...]
'''
def build_samples(coco, split_dir):
    annotation_per_image = defaultdict(list)
    # Suddivido le annotazioni in base alle immagini
    for annotation in coco['annotations']:
        annotation_per_image[annotation['image_id']].append(annotation)

    samples = []
    # Suddivido le informazioni alle corrispondenti immagini e creo un sample che contiene tutto
    for img_info in coco['images']:
        info = process_single_sample(img_info, annotation_per_image, split_dir)
        samples.append(info)
    return samples

'''
Funzione parallela per costruire i samples
Da notare che al momento non vengono effettivamente lette le immagini da disco
ma vengono solo costruiti i sample con le info delle annotazioni
'''
def build_samples_parallel(coco, split_dir, nCores):
    annotation_per_image = defaultdict(list)
    for annotation in coco['annotations']:
        annotation_per_image[annotation['image_id']].append(annotation)

    num_threads = nCores
    print(f"[INFO] Using {num_threads} cores for loading samples")

    with parallel_backend('threading', n_jobs=num_threads):
        samples = Parallel(verbose=0)(
            delayed(process_single_sample)(img_info, annotation_per_image, split_dir)
            for img_info in coco['images']
        )
    return samples

'''
Carica un intero split ('train' o 'val') e restituisce la lista dei sample
'''
def load_split(split, parallel, nCores = 1):
    if split == 'training':
        split_dir = TRAIN_DIR
    elif split == 'validation':
        split_dir = VAL_DIR
    else:
        raise ValueError("Split non recognized, use 'training' or 'validation'")

    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Directory not found: {split_dir}")

    coco = load_coco_annotations(split_dir)

    if parallel:
        samples = build_samples_parallel(coco, split_dir, nCores)
    else:
        samples = build_samples(coco, split_dir)
    return samples

'''
Carica il dataset e stampa il numero di samples
'''
def load_data(split, parallel, nCores = 1):
    samples = load_split(split, parallel, nCores)
    print(f"\n=== Split: {split} ===")
    print(f"N° samples: {len(samples)}")

    return samples

'''
Carica l'immagine da disco dal dict 'sample' e restituisce un array NumPy H×W×1.
'''
def sample_to_image_array(sample):
    img = cv2.imread(sample['img_path'], cv2.IMREAD_GRAYSCALE)      # restituisce un array NumPy bidimensionale di forma (H, W) con valori uint8 (0–255), dove H e W sono altezza e larghezza dell’immagine.
    if img is None:
        raise FileNotFoundError(f"Image not found: {sample['img_path']}")
    # Aggiunge la dimensione canale: da (H, W) a (H, W, 1), Albumentations richiede sempre un array 3D (H, W, C), anche se C=1 (canali) per le immagini in scala di grigi.
    return img[:, :, None]