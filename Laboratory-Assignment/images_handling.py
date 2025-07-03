import cv2
import matplotlib.pyplot as plt

'''
Visualizza una singola bounding box sull'immagine.
bbox in formato [xmin, ymin, width, height]
'''
def show_bbox(img, bbox, class_name, color=(0, 255, 0), thickness=2, text_color=(255,255,255)):
    # Calcola le coordinate del bbox
    x_min, y_min, w, h = bbox       # Prende le coordinate in formato COCO e le converte in Pascal_Voc
    x_min, x_max = int(x_min), int(x_min + w)
    y_min, y_max = int(y_min), int(y_min + h)

    # Disegna il rettangolo del bbox
    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color=color, thickness=thickness)

    # Disegna il rettangolo della label
    ((text_width, text_height), _) = cv2.getTextSize(class_name, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)
    cv2.rectangle(img,
                  (x_min, y_min - int(1.3 * text_height)),
                  (x_min + text_width, y_min),
                  color, thickness=-1)
    cv2.putText(img, class_name,
                (x_min, y_min - int(0.3 * text_height)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.35, text_color, lineType=cv2.LINE_AA)

    return img

'''
Visualizza un'immagine con bounding boxes in una griglia n_row x n_col alla posizione index
'''
def visualize_with_boxes(image, bboxes, category_ids, category_id_to_name, n_row, n_col, index, title=None):
    # image.squeeze() trasforma l’array da shape (H, W, 1) a (H, W)
    # cv2.cvtColor(..., COLOR_GRAY2BGR) duplica quel singolo canale su tre canali, restituendo un array (H, W, 3). Questo ti dà un’immagine ancora in scala di grigi ma con spazio colore BGR, necessario per OpenCV
    img = cv2.cvtColor(image.squeeze(), cv2.COLOR_GRAY2BGR)
    # Ciclo su ogni bounding box nell'immagine
    for bbox, cid in zip(bboxes, category_ids):     # Per ogni bbox tu hai le coordinate absolute dell’angolo in alto a sinistra (x1,y1) e in basso a destra (x2,y2)
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1     # Converti queste in [xmin, ymin, width, height] per il wrapper visualize_bbox
        class_name = category_id_to_name.get(cid, str(cid))     # Recupera il nome della classe
        img = show_bbox(img, [x1, y1, w, h], class_name)       # disegna sul tuo array BGR il rettangolo e l’etichetta
    plt.subplot(n_row, n_col, index)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

'''
Visualizza un'immagine senza bounding boxes in una griglia n_row x n_col alla posizione index
'''
def visualize_without_boxes(image, n_row, n_columns, index, title=None):
    plt.subplot(n_row, n_columns, index)
    plt.axis('off')
    if title:
        plt.title(title)
    plt.imshow(image.squeeze(), cmap='gray')

