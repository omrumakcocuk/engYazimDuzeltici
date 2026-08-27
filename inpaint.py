import base64
import json
import math
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
        if correction.get("action") not in ("replace", "rewrite_line"):
            continue
        targets = [
            anchor for anchor in correction.get("anchors", [])
            if anchor.get("relation") == "target"
        ]

        for target in targets:
            try:
                values = [float(target[key]) for key in ("x", "y", "width", "height")]
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in values):
                continue

            x = int(values[0] / 1000 * image_width)
            y = int(values[1] / 1000 * image_height)
            width = max(1, int(values[2] / 1000 * image_width))
            height = max(1, int(values[3] / 1000 * image_height))
            # Vision boxes can end on the centre of the outer pen stroke,
            # especially with cursive writing. Give the mask enough room to
            # remove those strokes, then clip it to the whitespace shared with
            # neighbouring words below.
            pad_x = max(3, int(width * 0.12))
            pad_y = max(3, int(height * 0.15))
            x1 = min(image_width, max(0, x - pad_x))
            y1 = min(image_height, max(0, y - pad_y))
            x2 = min(image_width, max(0, x + width + pad_x))
            y2 = min(image_height, max(0, y + height + pad_y))
            try:
                safe_top = float(target.get("safeTop", 0)) / 1000 * image_height
                safe_bottom = float(target.get("safeBottom", 1000)) / 1000 * image_height
            except (TypeError, ValueError):
                safe_top, safe_bottom = 0, image_height
            if math.isfinite(safe_top) and math.isfinite(safe_bottom) and safe_top < safe_bottom:
                y1 = max(y1, int(math.ceil(safe_top)))
                y2 = min(y2, int(math.floor(safe_bottom)))
            try:
                safe_left = float(target.get("slotX", 0)) / 1000 * image_width
                safe_right = safe_left + float(target.get("slotWidth", 1000)) / 1000 * image_width
            except (TypeError, ValueError):
                safe_left, safe_right = 0, image_width
            if math.isfinite(safe_left) and math.isfinite(safe_right) and safe_left < safe_right:
                x1 = max(x1, int(math.ceil(safe_left)))
                x2 = min(x2, int(math.floor(safe_right)))
            if x1 >= x2 or y1 >= y2:
                continue

            region = image[y1:y2, x1:x2]
            if region.size == 0 or region.ndim != 3 or region.shape[2] != 3:
                continue
            blue, green, red = cv2.split(region)
            blue16 = blue.astype(np.int16)
            green16 = green.astype(np.int16)
            red16 = red.astype(np.int16)
            gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            paper_level = float(np.percentile(gray, 82))
            background_kernel = max(9, int(round(min(region.shape[:2]) * 0.65)) | 1)
            background = cv2.morphologyEx(
                gray,
                cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (background_kernel, background_kernel)),
            )
            local_contrast = background.astype(np.int16) - gray.astype(np.int16)
            # Dark-blue handwriting varies greatly with lighting. Use both
            # colour dominance and local darkness, while keeping the mask
            # restricted to the OCR word box so neighbouring lines survive.
            blue_ink = (
                ((blue16 - red16 > 4) & (blue16 - green16 > -3) & (gray < paper_level - 5))
                | ((blue16 - red16 > 2) & (gray < paper_level - 22))
                | (gray < paper_level - 48)
                | ((local_contrast > 18) & (gray < paper_level - 4))
            )
            region_mask = (blue_ink.astype(np.uint8) * 255)
            # Gürültü noktalarını değil kalem darbelerini temizle.
            component_count, labels, stats, _ = cv2.connectedComponentsWithStats(region_mask, 8)
            filtered = np.zeros_like(region_mask)
            minimum_area = max(2, int(width * height * 0.0008))
            for component in range(1, component_count):
                if stats[component, cv2.CC_STAT_AREA] >= minimum_area:
                    filtered[labels == component] = 255
            region_mask = filtered
            kernel_size = max(3, int(round(min(width, height) * 0.10)) | 1)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
            region_mask = cv2.morphologyEx(region_mask, cv2.MORPH_CLOSE, kernel)
            region_mask = cv2.dilate(region_mask, kernel, iterations=1)
            full_mask[y1:y2, x1:x2] = cv2.max(full_mask[y1:y2, x1:x2], region_mask)

    if np.any(full_mask):
        radius = max(2, int(round(min(image_width, image_height) * 0.002)))
        image = cv2.inpaint(image, full_mask, radius, cv2.INPAINT_TELEA)

    if not cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise SystemExit("output image could not be written")


if __name__ == "__main__":
    main()
