import base64
import json
import sys

import cv2
import numpy as np


def main():
    if len(sys.argv) != 4:
        raise SystemExit("usage: inpaint.py input-image corrections-json output-image")

    input_path, corrections_path, output_path = sys.argv[1:]
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("image could not be decoded")

    with open(corrections_path, "r", encoding="utf-8") as handle:
        corrections = json.load(handle)

    image_height, image_width = image.shape[:2]
    full_mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for correction in corrections:
        if correction.get("action") != "replace":
            continue
        target = next(
            (anchor for anchor in correction.get("anchors", []) if anchor.get("relation") == "target"),
            None,
        )
        if not target:
            continue

        x = int(target["x"] / 1000 * image_width)
        y = int(target["y"] / 1000 * image_height)
        width = max(1, int(target["width"] / 1000 * image_width))
        height = max(1, int(target["height"] / 1000 * image_height))
        pad_x = max(3, int(width * 0.06))
        pad_y = max(3, int(height * 0.12))
        x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
        x2, y2 = min(image_width, x + width + pad_x), min(image_height, y + height + pad_y)

        region = image[y1:y2, x1:x2]
        blue, green, red = cv2.split(region)
        blue_ink = (
            (blue.astype(np.int16) - red.astype(np.int16) > 14)
            & (blue.astype(np.int16) - green.astype(np.int16) > 3)
            & (blue < 225)
        )
        region_mask = (blue_ink.astype(np.uint8) * 255)
        kernel_size = max(3, int(round(min(width, height) * 0.07)) | 1)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        region_mask = cv2.dilate(region_mask, kernel, iterations=1)
        full_mask[y1:y2, x1:x2] = cv2.max(full_mask[y1:y2, x1:x2], region_mask)

    if np.any(full_mask):
        radius = max(2, int(round(min(image_width, image_height) * 0.002)))
        image = cv2.inpaint(image, full_mask, radius, cv2.INPAINT_TELEA)

    if not cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise SystemExit("output image could not be written")


if __name__ == "__main__":
    main()
