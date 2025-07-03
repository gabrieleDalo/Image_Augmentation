''' Esempio di semplice Transform
    # Prendi il primo sample
    first = train_samples[0]
    image = sample_to_image_array(first)  # converte l'immagine in un array Numpy per usarla in Albumentations
    print(f"image.shape = {image.shape}    # (altezza, larghezza, canali)")

    # Visualizza con albumentations
    transform = A.HorizontalFlip(p=1.0)     # Crea un oggetto Albumentations che ruota orizzontalmente ogni immagine che gli passi,, p=1.0 significa: applicalo sempre
    out = transform(image=image)        # Ottiene l'immagine trasformata
    flipped = out['image']          # In questo caso applico la trasformazione solo all'immagine, se avessimo fornito anche bboxes ci troveremmo dentro anche out['bboxes'] aggiornate

    # Faccio il plot
    plt.subplot(1, 2, 1)        # Dice a Matplotlib di voler creare una griglia di 1 riga e 2 colonne di grafici, e di concentrarsi sul primo di questi (l’indice 1). In pratica: “voglio due grafici affiancati, e ora disegno nel primo”
    plt.imshow(image.squeeze(), cmap='gray')    # image.squeeze() trasforma l’array da (H, W, 1) a (H, W) eliminando la dimensione di lunghezza 1, così Matplotlib può interpretarlo correttamente come immagine bidimensionale. cmap='gray' imposta la colormap su scala di grigi, altrimenti Matplotlib cercherebbe di colorarlo
    plt.title("Original")
    plt.subplot(1, 2, 2)        # Ora dice di disegnare nel secondo grafico
    plt.imshow(flipped.squeeze(), cmap='gray')
    plt.title("Flipped")
    plt.show()
'''

''' Esempio di semplice Pipeline usando Compose
    # Prendi il primo sample
    first = train_samples[0]
    image = sample_to_image_array(first)  # converte l'immagine in un array Numpy per usarla in Albumentations
    print(f"image.shape = {image.shape}    # (altezza, larghezza, canali)")

    # Visualizza con albumentations
    transform = A.Compose([
        A.HorizontalFlip(p=0.5), # 50% chance to flip
        A.Rotate(p=0.8), # 80% chance to adjust brightness/contrast
        A.VerticalFlip(p=0.3), # 30% chance to blur
    ])
    out = transform(image=image)
    flipped = out['image']

    # Faccio il plot
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(), cmap='gray')
    plt.title("Original")
    plt.subplot(1, 2, 2)
    plt.imshow(flipped.squeeze(), cmap='gray')
    plt.title("Composed")
    plt.show()
'''

''' Esempio con pipeline e bboxes
    # 1) Carica i primi 2 sample del training
    samples = train_samples[:2]

    # 2) Definisci la pipeline di augmentation
    pipeline = A.Compose([
        A.HorizontalFlip(p=1.0),
        A.RandomBrightnessContrast(p=0.7),
        A.RandomScale(scale_limit=0.2, p=0.5),
    ],
        bbox_params=A.BboxParams(
            format='pascal_voc',  # le nostre box sono [xmin, ymin, xmax, ymax]
            label_fields=['labels']  # indica ad Albumentations dove trovare le labels
        ))

    # 3) Per ciascun sample, applica e visualizza
    for idx, sample in enumerate(samples):
        # a) Carica immagine e annotazioni
        image = sample_to_image_array(sample)  # array H×W×1
        original_bboxes = sample['bboxes']
        labels = sample['labels']

        # b) Applica la trasformazione
        out = pipeline(image=image, bboxes=original_bboxes, labels=labels)
        transformed_image = out['image']
        transformed_bboxes = out['bboxes']

        # c) Visualizza l'originale
        plt.figure(figsize=(5, 5))
        plt.title(f"Sample {idx} – Original")
        plt.imshow(image.squeeze(), cmap='gray')
        ax = plt.gca()
        for (x1, y1, x2, y2) in original_bboxes:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, linewidth=2)
            ax.add_patch(rect)
        plt.axis('off')
        plt.show()

        # d) Visualizza il trasformato
        plt.figure(figsize=(5, 5))
        plt.title(f"Sample {idx} – Transformed")
        plt.imshow(transformed_image.squeeze(), cmap='gray')
        ax = plt.gca()
        for (x1, y1, x2, y2) in transformed_bboxes:
            rect = plt.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 fill=False, linewidth=2)
            ax.add_patch(rect)
        plt.axis('off')
        plt.show()
'''

''' Esempio con pipeline e bboxes con loro visualizzazione
        # Dizionario id->nome categoria
    category_id_to_name = {
        1: 'person', 2: 'bike', 3: 'car', 4: 'motor', 6: 'bus', 7: 'train', 8: 'truck', 10: 'light', 11: 'hydrant', 12: 'sign', 17: 'dog', 37: 'skateboard', 73: 'stroller', 77: 'scooter', 79: 'other vehicle'
    }

    start_time = time.perf_counter()

    # Verifica e carica split di training e di validation
    train_samples = load_data('training')
    val_samples = load_data('validation')

    # Ora train_samples e val_samples contengono le liste di dizionari con img_path, bboxes, labels
    print(f"\nEsempio di sample train[0]: {train_samples[0]}")

    # Prendi il primo sample
    sample = train_samples[0]
    image = sample_to_image_array(sample)  # converte l'immagine in un array Numpy per usarla in Albumentations
    bboxes = sample['bboxes']  # lista di [xmin, ymin, xmax, ymax]
    labels = sample['labels']  # lista di category_id

    # Visualizza l'immagine originale
    #visualize_without_boxes(image, 1, 2, 1, "Original")
    visualize_with_boxes(image, bboxes, labels, category_id_to_name, 1, 2, 1, "Original")

    # Definisci transform includendo bbox e labels, crea un oggetto Albumentations che ruota orizzontalmente ogni immagine che gli passi,, p=1.0 significa: applicalo sempre
    transform = A.Compose([
        A.HorizontalFlip(p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    # Applica la transformazione
    transformed = transform(image=image, bboxes=bboxes, labels=labels)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    transformed_labels = transformed['labels']

    # Visualizza l'immagine trasformata
    #visualize_without_boxes(transformed_image, 1, 2, 2, "Transformed (Flipped)")
    visualize_with_boxes(transformed_image, transformed_bboxes, transformed_labels, category_id_to_name, 1, 2, 2, "Transformed (Flipped)")
    plt.show()

    # Stampa dettagli
    print(f"Original shape: {image.shape}, Transformed shape: {transformed_image.shape}")
    print(f"Original bboxes: {bboxes}")
    print(f"Transformed bboxes: {transformed_bboxes}")
    print(f"Original labels: {labels}")
    print(f"Transformed labels: {transformed_labels}")

    end_time = time.perf_counter()

    elapsed_time = end_time - start_time
    print("")
    print(f"Tempo trascorso: {printTime(elapsed_time)}")
'''

'''
    # Mappa delle categorie, id->nome categoria
    category_id_to_name = {
        1: 'person', 2: 'bike', 3: 'car', 4: 'motor', 6: 'bus',
        7: 'train', 8: 'truck', 10: 'light', 11: 'hydrant', 12: 'sign',
        17: 'dog', 37: 'skateboard', 73: 'stroller', 77: 'scooter', 79: 'other vehicle'
    }

    start_time = time.perf_counter()

    # Carica tutti i sample di training e validation
    train_samples = load_data('training', parallel=True)
    #val_samples = load_data('validation', parallel=False)

    end_time = time.perf_counter()

    total_time_sequential = end_time - start_time
    print("")
    print(f"Sequential time to load dataset: {printTime(total_time_sequential)}")

    start_time = time.perf_counter()

    # Indici delle immagini da processare
    #sample_indices = list(range(1, 6))  # range(1, 101) sono gli indici da 1 a 100
    sample_indices = list(range(1, len(train_samples)))

    # Definisci la pipeline di augmentation
    transform = A.Compose([
        A.RandomSizedBBoxSafeCrop(height=512, width=512, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.Rotate(limit=45, p=1.0),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    print("")
    visuals = []
    # Loop sui samples da augmentare
    for i, idx in enumerate(sample_indices, start=1):
        print(f"Image n° {idx}")
        sample = train_samples[idx]

        # Immagine originale
        img = sample_to_image_array(sample)  # converte l'immagine in un array Numpy per usarla in Albumentations
        bboxes = sample['bboxes']
        labels = sample['labels']

        # Immagine augmentata
        out = transform(image=img, bboxes=bboxes, labels=labels)
        img_t = out['image']
        bboxes_t = out['bboxes']
        labels_t = [int(x) for x in out['labels']]  # Garantisce che le label trasformate siano ancora int

        if idx < 4:  # Faccio il plot solo delle prime 3 immagini (serve per i test durante lo sviluppo)
            visuals.append((idx, img, bboxes, labels, img_t, bboxes_t, labels_t))

    end_time = time.perf_counter()

    # TODO: alla fine, rimuovere le parti per visualizzare le immagini?
    # Prepara la figura: due righe per ogni indice (originale + augmented)
    nSamples = len(visuals)
    nCols = 2
    nRows = nSamples
    plt.figure(figsize=(6 * nCols, 4 * nRows))

    for i, (idx, img_o, b_o, l_o, img_a, b_a, l_a) in enumerate(visuals, start=1):
        # Immagine originale
        visualize_with_boxes(
            img_o, b_o, l_o, category_id_to_name,
            n_row=nRows, n_col=nCols, index=2 * i - 1,
            title=f"Orig idx={idx}"
        )
        # Immagine augmentata
        visualize_with_boxes(
            img_a, b_a, l_a, category_id_to_name,
            n_row=nRows, n_col=nCols, index=2 * i,
            title="Augmented"
        )
    plt.tight_layout()
    plt.show()

    total_time_sequential = end_time - start_time
    print("")
    print(f"Sequential time to augment dataset: {printTime(total_time_sequential)}")
'''