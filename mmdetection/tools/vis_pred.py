import os
import cv2
import numpy as np

from mmdet.apis import init_detector
from mmengine.visualization import Visualizer


# =========================
# 1. load model
# =========================
def build_model(config, checkpoint, device='cuda'):
    model = init_detector(config, checkpoint, device=device)
    model.eval()
    return model


# =========================
# 2. inference (关键修复点)
# =========================
from mmdet.apis import init_detector, inference_detector


def inference(model, img_path):
    result = inference_detector(model, img_path)
    return result


# =========================
# 3. parse results
# =========================
def parse(result):
    pred = result.pred_instances
    boxes = pred.bboxes.cpu().numpy()
    scores = pred.scores.cpu().numpy()
    return boxes, scores


# =========================
# 4. draw + save
# =========================
def draw_and_save(img, boxes, scores, save_path):
    img = img.copy()

    for b, s in zip(boxes, scores):
        x1, y1, x2, y2 = map(int, b)

        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(
            img,
            f"{s:.2f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            1
        )

    cv2.imwrite(save_path, img)


# =========================
# 5. run
# =========================
def run(config, ckpt, img_list, save_dir):

    import os
    import cv2
    os.makedirs(save_dir, exist_ok=True)

    model = init_detector(config, ckpt, device='cuda')

    for i, img_path in enumerate(img_list):

        if not os.path.exists(img_path):
            print(f"[ERROR] image not found: {img_path}")
            continue

        # ===== inference =====
        result = inference_detector(model, img_path)

        boxes = result.pred_instances.bboxes.cpu().numpy()
        scores = result.pred_instances.scores.cpu().numpy()

        # ===== load image =====
        img = cv2.imread(img_path)

        # ===== draw =====
        for b, s in zip(boxes, scores):
            x1, y1, x2, y2 = map(int, b)

            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{s:.2f}", (x1, y1-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 0, 255), 1)

        save_path = os.path.join(save_dir, f"vis_{i}.jpg")
        cv2.imwrite(save_path, img)

        print(f"[OK] saved: {save_path}")


# =========================
# 6. entry
# =========================
if __name__ == "__main__":

    config = "/home/hl/my_data/zyr/ETS/mmdetection/configs/grounding_dino/CDFSOD/GroudingDINO-few-shot-SwinB.py"
    checkpoint = "/home/hl/my_data/zyr/ETS/work_dirs/clipart_10_newbu/exp36/epoch_10.pth"

    img_list = [
        "/home/hl/my_data/zyr/ETS/data/clipart1k/test/118378507.jpg",
        "/home/hl/my_data/zyr/ETS/data/clipart1k/test/125196156.jpg",
        "/home/hl/my_data/zyr/ETS/data/clipart1k/test/132858229.jpg"
    ]

    save_dir = "./vis_results"

    run(config, checkpoint, img_list, save_dir)