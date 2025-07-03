import math
import albumentations as A
import matplotlib.pyplot as plt
import time
import shutil

from load_dataset import *
from images_handling import *
from utils import *

from concurrent.futures import ProcessPoolExecutor
from joblib import Parallel, delayed, parallel_backend

import cv2
cv2.setNumThreads(0)       # Senza questo, Albumentations di default esegue OpenCV a 12 threads
cv2.ocl.setUseOpenCL(False)

if __name__ == "__main__":
    # Mappa delle categorie, id -> nome categoria
    categoryID_to_name = {
        1: 'person', 2: 'bike', 3: 'car', 4: 'motor', 6: 'bus',
        7: 'train', 8: 'truck', 10: 'light', 11: 'hydrant', 12: 'sign',
        17: 'dog', 37: 'skateboard', 73: 'stroller', 77: 'scooter', 79: 'other vehicle'
    }

    output_dir = "augmented_images"
    parallel_load = True
    parallel_augment = True
    nCores = os.cpu_count()
    max_visualize = 3       # N° di immagini da visualizzare
    visuals = []

    # Se la cartella esiste, svuota; altrimenti crea
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("Select the dataset to use:")
    print(" - 1: Small (1000 images)")
    print(" - 2: Medium (5000 images)")
    print(" - 3: Big (10000 images)")

    datasetSize = ask_user_input(1, 3)

    print("\nSelect the parallel version to use:")
    print(" - 0: Serial execution")
    print(" - 1: ProcessPoolExecutor")
    print(" - 2: Joblib with threading")
    print(" - 3: Joblib with loky")

    parallelVersion = ask_user_input(0, 3)

    if parallelVersion == 0:
        parallel_load = False
        parallel_augment = False
    else:
        not_valid = True
        while not_valid:
            try:
                nCores = int(input("Select number of cores to use: "))
                if 1 <= nCores <= os.cpu_count():
                    not_valid = False
                else:
                    print(f"Error, insert a valid number (between 1 and {os.cpu_count()}).")
            except ValueError:
                print("Error, insert a number.")
        print("")

    start_time = time.perf_counter()

    # Carica tutti i sample di training (e validation)
    train_samples = load_data('training', parallel_load, nCores)
    #val_samples = load_data('validation', parallel_load, nCores)

    end_time = time.perf_counter()
    print(f"\nTotal time to load dataset: {printTime(end_time - start_time)}")

    # Indici delle immagini da processare
    if datasetSize == 1:
        sample_indices = list(range(1, 1001))
    elif datasetSize == 2:
        sample_indices = list(range(1, 5001))
    elif datasetSize == 3:
        sample_indices = list(range(1, len(train_samples)))

    start_time = time.perf_counter()

    # Loop sui samples da augmentare
    if parallel_augment:
        # num_processes = 2      # TODO: testare con diversi numeri di processi
        print(f"\n[INFO] Using {nCores} cores for augmentation")
        print("Parallel Augmentation...")

        if parallelVersion == 1:
            args = [(idx, train_samples[idx]) for idx in sample_indices]

            # Calcolo dinamico di un chunksize
            # ad es. tot_samples diviso (n_processi * fattore)
            tot = len(args)
            factor = 25
            chunkSize = max(1, math.ceil(tot / (nCores * factor)))
            print(f"\n[INFO] Chunksize = {chunkSize}")

            with ProcessPoolExecutor(max_workers=nCores) as executor:
                results = list(executor.map(process_sample, args, chunksize=chunkSize))

        elif parallelVersion == 2:
            # Usa il backend 'threading'
            with parallel_backend('threading', n_jobs=nCores):
                results = Parallel(verbose=10)(
                    delayed(process_sample)((idx, train_samples[idx])) for idx in sample_indices
                )

        elif parallelVersion == 3:
            # Usa il backend 'loky'
            with parallel_backend('loky', n_jobs=nCores):
                results = Parallel(verbose=10)(
                    delayed(process_sample)((idx, train_samples[idx])) for idx in sample_indices
                )

        visuals = results[:max_visualize]  # solo le prime 3 per visualizzazione
    else:
        print("\nSequential Augmentation...")
        for i, idx in enumerate(sample_indices, start=1):
            sample = train_samples[idx]
            results = process_sample((idx, sample))

            if idx < (max_visualize+1):
                visuals.append(results)

    end_time = time.perf_counter()
    print(f"\nTotal time to augment dataset: {printTime(end_time - start_time)}")
    print(f'({(end_time - start_time):0.3f} s)')

    # Prepara la figura: due righe per ogni indice (originale + augmented)
    nSamples = len(visuals)
    nCols = 2
    nRows = nSamples
    plt.figure(figsize=(6 * nCols, 4 * nRows))

    for i, (idx, img_o, b_o, l_o, img_a, b_a, l_a) in enumerate(visuals, start=1):
        # Immagine originale
        visualize_with_boxes(
            img_o, b_o, l_o, categoryID_to_name,
            n_row=nRows, n_col=nCols, index=2 * i - 1,
            title=f"Orig idx={idx}"
        )
        # Immagine augmentata
        visualize_with_boxes(
            unnormalize_img(img_a.squeeze()), b_a, l_a, categoryID_to_name,
            n_row=nRows, n_col=nCols, index=2 * i,
            title="Augmented"
        )
    plt.tight_layout()
    plt.show()
