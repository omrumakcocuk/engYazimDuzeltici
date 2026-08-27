import base64
import json
import math
import sys

import cv2
import numpy as np


def detect_ink_mask(image, x1, y1, x2, y2, width, height):
    region = image[y1:y2, x1:x2]
    if region.size == 0 or region.ndim != 3 or region.shape[2] != 3:
        return None
    blue, green, red = cv2.split(region)
    blue16 = blue.astype(np.int16)
    green16 = green.astype(np.int16)
    red16 = red.astype(np.int16)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # A tightly padded box around a dense multi-word phrase can be mostly
    # covered by ink, which pulls a percentile computed from the box itself
    # too far down - the "82nd percentile is paper" assumption needs enough
    # actual paper visible in the sample. Sample the paper level from a wider
    # neighbourhood instead (word plus roughly its own size of margin on
    # every side), which almost always has enough visible paper regardless
    # of how densely the target box itself is filled; otherwise the whole
    # box gets misread as ink and inpainting a solid block instead of just
    # the pen strokes is what left the visible blurry patch on ruled paper.
    sample_x1 = max(0, x1 - (x2 - x1))
    sample_y1 = max(0, y1 - (y2 - y1))
    sample_x2 = min(image.shape[1], x2 + (x2 - x1))
    sample_y2 = min(image.shape[0], y2 + (y2 - y1))
    sample_gray = cv2.cvtColor(image[sample_y1:sample_y2, sample_x1:sample_x2], cv2.COLOR_BGR2GRAY)
    paper_level = float(np.percentile(sample_gray, 75))
    # A kernel this close to the box's own size cannot recover a true local
    # "paper" estimate once ink covers much of a dense multi-letter/word box
    # (the close operation never reaches a fully white patch to report), so
    # local_contrast stays high across far more of the box than the actual
    # pen strokes - this is what turned a bold phrase's mask into a solid
    # block instead of just its letter shapes, and inpainting a solid block
    # is what left a visible blurry patch on ruled paper.
    background_kernel = max(9, int(round(min(region.shape[:2]) * 0.35)) | 1)
    background = cv2.morphologyEx(
        gray,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (background_kernel, background_kernel)),
    )
    local_contrast = background.astype(np.int16) - gray.astype(np.int16)
    # Dark-blue handwriting varies greatly with lighting. Use both colour
    # dominance and local darkness, while keeping the mask restricted to the
    # supplied box so neighbouring lines/words survive.
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
    return region_mask


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
            # especially with cursive writing, so start from a small padded
            # box around the box Vision reported. A capital letter or an
            # ascender (b/d/h/k/l/t...) is ordinary and already sits inside
            # Vision's own box, right up against its top edge - that is not
            # a sign of anything cut off, unlike a descender (g/j/p/q/y),
            # which Vision's box more often does clip. Padding the top less
            # than the bottom means a normal word's ink usually stops well
            # short of the top edge, instead of touching it (and triggering
            # the growth loop below) essentially every time.
            pad_x = max(3, int(width * 0.12))
            pad_top = max(3, int(height * 0.22))
            pad_bottom = max(3, int(height * 0.15))
            core_x1 = min(image_width, max(0, x - pad_x))
            core_y1 = min(image_height, max(0, y - pad_top))
            core_x2 = min(image_width, max(0, x + width + pad_x))
            core_y2 = min(image_height, max(0, y + height + pad_bottom))

            # The safe zone (safeTop/safeBottom, slotX/slotWidth) is the
            # actual whitespace shared with the neighbouring line/word and is
            # the true hard limit on how far the mask may ever grow.
            max_x1, max_y1, max_x2, max_y2 = core_x1, core_y1, core_x2, core_y2
            try:
                safe_top = float(target.get("safeTop", 0)) / 1000 * image_height
                safe_bottom = float(target.get("safeBottom", 1000)) / 1000 * image_height
            except (TypeError, ValueError):
                safe_top, safe_bottom = 0, image_height
            if math.isfinite(safe_top) and math.isfinite(safe_bottom) and safe_top < safe_bottom:
                max_y1 = max(0, int(math.ceil(safe_top)))
                max_y2 = min(image_height, int(math.floor(safe_bottom)))
            try:
                safe_left = float(target.get("slotX", 0)) / 1000 * image_width
                safe_right = safe_left + float(target.get("slotWidth", 1000)) / 1000 * image_width
            except (TypeError, ValueError):
                safe_left, safe_right = 0, image_width
            if math.isfinite(safe_left) and math.isfinite(safe_right) and safe_left < safe_right:
                max_x1 = max(0, int(math.ceil(safe_left)))
                max_x2 = min(image_width, int(math.floor(safe_right)))
            if max_x1 >= max_x2 or max_y1 >= max_y2:
                continue

            x1 = max(core_x1, max_x1)
            y1 = max(core_y1, max_y1)
            x2 = min(core_x2, max_x2)
            y2 = min(core_y2, max_y2)
            if x1 >= x2 or y1 >= y2:
                continue

            region_mask = detect_ink_mask(image, x1, y1, x2, y2, width, height)
            if region_mask is None:
                continue

            # Jumping straight to the full safe zone the instant a stroke
            # touches the padded edge (rather than only when it is actually
            # cut off) used to erase a much bigger area than the ink needed,
            # and OpenCV's inpaint then smeared that mostly-blank extra space
            # into a visible blurry patch - most visible over ruled/grid
            # paper, where inpainting a large hole cannot reconstruct the
            # fine repeating grid lines.
            #
            # Growing by half of whatever gap remains to the safe zone breaks
            # down for the first/last physical line on a page: there is no
            # real neighbouring line there, so that "gap" is the distance to
            # the page edge - often hundreds of pixels - and even one halving
            # step already erases most of the way to the edge. Step by a
            # fixed amount tied to the word's own size instead, so a
            # descender/ascender that only needed a little more room stops
            # after a small step regardless of how far away the true safe-zone
            # ceiling happens to be, and only ink that keeps touching after
            # several such steps ever reaches that ceiling.
            step_y = max(1, int(round(height * 0.4)))
            step_x = max(1, int(round(width * 0.4)))
            # An ordinary capital or ascender already sits right against the
            # padded top edge as a matter of course, so "still touching
            # after growing" is not the reliable cut-off signal upward that
            # it is downward (only a real descender keeps touching the
            # bottom edge step after step). Left/right and bottom get the
            # full run of steps; top gets at most one small nudge.
            top_steps_allowed = 1
            top_steps_used = 0
            for _ in range(4):
                touches_top = (
                    top_steps_used < top_steps_allowed
                    and y1 > max_y1 and bool(region_mask[0, :].any())
                )
                touches_bottom = y2 < max_y2 and bool(region_mask[-1, :].any())
                touches_left = x1 > max_x1 and bool(region_mask[:, 0].any())
                touches_right = x2 < max_x2 and bool(region_mask[:, -1].any())
                if not (touches_top or touches_bottom or touches_left or touches_right):
                    break
                grown_y1 = max(max_y1, y1 - step_y) if touches_top else y1
                grown_y2 = min(max_y2, y2 + step_y) if touches_bottom else y2
                grown_x1 = max(max_x1, x1 - step_x) if touches_left else x1
                grown_x2 = min(max_x2, x2 + step_x) if touches_right else x2
                if grown_x1 == x1 and grown_y1 == y1 and grown_x2 == x2 and grown_y2 == y2:
                    break
                grown_mask = detect_ink_mask(image, grown_x1, grown_y1, grown_x2, grown_y2, width, height)
                if grown_mask is None:
                    break
                if touches_top:
                    top_steps_used += 1
                x1, y1, x2, y2, region_mask = grown_x1, grown_y1, grown_x2, grown_y2, grown_mask

            full_mask[y1:y2, x1:x2] = cv2.max(full_mask[y1:y2, x1:x2], region_mask)

    if np.any(full_mask):
        radius = max(2, int(round(min(image_width, image_height) * 0.002)))
        image = cv2.inpaint(image, full_mask, radius, cv2.INPAINT_TELEA)

    if not cv2.imwrite(output_path, image, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
        raise SystemExit("output image could not be written")


if __name__ == "__main__":
    main()
