import albumentations as A
import cv2
import numpy as np
import os

from load_dataset import sample_to_image_array

output_dir = "augmented_images"

normalization_mean = 0.5
normalization_std = 0.5

# Definisci la pipeline di augmentation
transform = A.Compose([
    # 1) Crop safe + resize
    A.RandomSizedBBoxSafeCrop(height=512, width=512, p=0.5),  # Fa il crop dell'immagine tenendo conto dei bboxes

    # 2) Trasformazioni geometriche
    A.HorizontalFlip(p=0.7),
    A.Rotate(limit=45, p=0.3),

    # 3) Occlusion / dropout
    A.OneOf([  # Per non rendere l'immagine troppo corrotta ha senso applicare solo una delle due
        A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(0.1, 0.2), hole_width_range=(0.1, 0.2), p=1.0),
        # Rimuove casualmente regioni rettangolari dell'immagine, simula occlusione
        A.GridDropout(ratio=0.5, unit_size_range=(16, 32), p=1.0),
        # creare una griglia di celle mancanti, un pattern regolare simile a uno schermo bucato
    ], p=0.2),

    # 4) Trasformazioni di color features
    # In questo caso ho solo immagini termiche in scala di grigi, quindi non ha senso aggiungerne

    # 5) Trasformazioni Affine
    A.Affine(scale=(0.8, 1.2), translate_percent=(0.0, 0.05), shear=(-16, 16), p=0.7),

    # 6) Aggiunge rumore e contrasto
    A.OneOf([
        A.RandomBrightnessContrast(p=1.0),  # Modifica luminosità e contrasto
        A.GaussNoise(p=1.0),  # Aggiunge un rumore gaussiano
    ], p=0.3),

    # 7) Normalizzazione
    # Si usa per mettere tutti i valori di input su una scala normalizzata, in modo da accellerare la convergenza in caso di training e rendere più stabile l'ottimizzazione
    A.Normalize(mean=(normalization_mean,), std=(normalization_std,)),
    # Per ogni immagine calcola una propria media e deviazione standard
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'], min_visibility=0.1))

'''
Converte un intervallo di tempo (float in secondi) in stringa con minuti, secondi e millisecondi
'''
def printTime(time):
    minutes = int(time // 60)
    seconds = int(time % 60)
    milliseconds = int((time - minutes*60 - seconds) * 1000)
    return f"{minutes} minutes {seconds} seconds {milliseconds} milliseconds"

def process_sample(sample_and_idx):
    idx, sample = sample_and_idx

    # Immagine originale
    img = sample_to_image_array(sample)
    bboxes = sample['bboxes']
    labels = sample['labels']

    # Immagine augmentata
    out = transform(image=img, bboxes=bboxes, labels=labels)
    img_t = out['image']
    bboxes_t = out['bboxes']
    labels_t = [int(x) for x in out['labels']]

    # Portiamo da float32 [0,1] a uint8 [0,255]
    img_uint8 = unnormalize_img(img_t.squeeze())
    img_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    output_path = os.path.join(output_dir, f"aug_{idx}.jpg")
    cv2.imwrite(output_path, img_bgr)

    return idx, img, bboxes, labels, img_t, bboxes_t, labels_t

# Le immagini normalizzate servono per il training, per mostrarle con matplot servono non normalizzate
def unnormalize_img(img_norm):
    # img_norm: H×W×1 o H×W float in [-1,1+]
    img = img_norm * normalization_std + normalization_mean
    img = np.clip(img, 0, 1)
    # se vuoi in [0,255] per uint8:
    img = (img * 255).astype(np.uint8)

    return img

def ask_user_input(min_value, max_value):
    while True:
        try:
            user_input = int(input("Insert number: "))
            if min_value <= user_input <= max_value:
                return user_input
            else:
                print(f"Error, insert a valid number (between {min_value} and {max_value}).")
        except ValueError:
            print("Error, insert a number.")

'''
# Per profilare la funzione process_sample

import cProfile
import pstats
from io import StringIO

def profile_process_sample(sample):
    pr = cProfile.Profile()
    pr.enable()

    process_sample(sample)

    pr.disable()
    s = StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumtime')
    ps.print_stats(20)  # Mostra le 20 funzioni più lente
    print(s.getvalue())
    
profile_process_sample((0,train_samples[0]))
'''