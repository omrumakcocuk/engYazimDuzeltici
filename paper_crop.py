import json
import sys

import cv2
import numpy as np


def order_points(points):
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def polygon_score(quad, contour_area, image):
    height, width = image.shape[:2]
    image_area = float(height * width)
    quad_area = abs(cv2.contourArea(quad.astype(np.float32)))
    area_ratio = quad_area / image_area
    if area_ratio < 0.18 or area_ratio > 0.97:
        return None

    ordered = order_points(quad)
    top, right, bottom, left = (
        np.linalg.norm(ordered[1] - ordered[0]),
        np.linalg.norm(ordered[2] - ordered[1]),
        np.linalg.norm(ordered[2] - ordered[3]),
        np.linalg.norm(ordered[3] - ordered[0]),
    )
    paper_width = max(top, bottom)
    paper_height = max(left, right)
    if min(paper_width, paper_height) < 180:
        return None
    aspect = paper_width / max(1.0, paper_height)
    if aspect < 0.35 or aspect > 2.8:
        return None

    rectangularity = min(1.0, contour_area / max(1.0, quad_area))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillConvexPoly(mask, ordered.astype(np.int32), 255)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = cv2.mean(gray, mask=mask)[0] / 255.0
    if brightness < 0.32:
        return None
    return area_ratio * 3.0 + rectangularity + brightness * 0.6


def centered_paper_mask(image):
    height, width = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    x1, x2 = int(width * 0.38), int(width * 0.62)
    y1, y2 = int(height * 0.45), int(height * 0.70)
    sample = lab[y1:y2, x1:x2].reshape(-1, 3)
    reference = np.median(sample, axis=0)
    # Paper lighting can vary substantially from top to bottom, therefore
    # luminance is weighted less than colour channels.
    delta = lab - reference
    distance = np.sqrt(delta[:, :, 0] ** 2 * 0.20 +
                       delta[:, :, 1] ** 2 * 1.5 +
                       delta[:, :, 2] ** 2 * 1.5)
    mask = (distance < 28).astype(np.uint8) * 255
    kernel_size = max(7, int(round(min(height, width) * 0.012)) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    return mask


def find_paper(image):
    height, width = image.shape[:2]
    detection_scale = min(1.0, 1600.0 / max(height, width))
    if detection_scale < 1.0:
        small = cv2.resize(image, None, fx=detection_scale, fy=detection_scale,
                           interpolation=cv2.INTER_AREA)
    else:
        small = image.copy()

    small_height, small_width = small.shape[:2]
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    edges = cv2.Canny(blurred, 28, 95)
    edge_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, edge_kernel, iterations=2)

    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    light = cv2.inRange(hsv, np.array((0, 0, 75)), np.array((180, 145, 255)))
    mask_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 17))
    light = cv2.morphologyEx(light, cv2.MORPH_CLOSE, mask_kernel, iterations=2)

    colour = centered_paper_mask(small)
    candidates = []
    fallback_contours = []
    for binary in (edges, light, colour):
        contours, _ = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            contour_area = abs(cv2.contourArea(contour))
            if contour_area < small_height * small_width * 0.16:
                continue
            fallback_contours.append((contour_area, contour))
            hull = cv2.convexHull(contour)
            perimeter = cv2.arcLength(hull, True)
            for epsilon in (0.012, 0.018, 0.025, 0.035, 0.05):
                polygon = cv2.approxPolyDP(hull, epsilon * perimeter, True)
                if len(polygon) != 4 or not cv2.isContourConvex(polygon):
                    continue
                quad = polygon.reshape(4, 2).astype(np.float32)
                score = polygon_score(quad, contour_area, small)
                if score is not None:
                    candidates.append((score, quad))
                break

    if not candidates:
        # Torn or folded sheets often do not form an exact four-corner contour.
        # A minimum-area rectangle is a safe fallback only for a large central
        # light component, and is still rejected by polygon_score when implausible.
        centre = np.array((small_width * 0.5, small_height * 0.58))
        for contour_area, contour in sorted(fallback_contours, reverse=True, key=lambda item: item[0]):
            if cv2.pointPolygonTest(contour, tuple(centre), False) < 0:
                continue
            quad = cv2.boxPoints(cv2.minAreaRect(contour)).astype(np.float32)
            score = polygon_score(quad, contour_area, small)
            if score is not None:
                candidates.append((score - 0.25, quad))
                break
    if not candidates:
        return None
    _, best = max(candidates, key=lambda item: item[0])
    best = order_points(best / detection_scale)
    margin = max(height, width) * 0.012
    touches = sum((
        np.min(best[:, 0]) <= margin,
        np.max(best[:, 0]) >= width - margin,
        np.min(best[:, 1]) <= margin,
        np.max(best[:, 1]) >= height - margin,
    ))
    # A shape that follows several image borders is usually the desk, keyboard,
    # or complete photograph rather than the sheet itself.
    return None if touches >= 2 else best


def crop_around_handwriting(image):
    blue, green, red = cv2.split(image)
    blue_ink = ((blue.astype(np.int16) - red.astype(np.int16) > 14) &
                (blue.astype(np.int16) - green.astype(np.int16) > 3) &
                (blue < 225))
    # Siyah/kurşun kalem için yalnız renk farkına güvenemeyiz. Açık kâğıt
    # üzerinde yerel arka plandan belirgin biçimde koyu ince darbeleri ekle;
    # siyah masa ve klavye bu parlaklık kapısı sayesinde aday olmaz.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    local_background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    )
    dark_ink = ((local_background.astype(np.int16) - gray.astype(np.int16) > 22)
                & (local_background > 125))
    # Renkli kalem yeterince belirginse daha geniş koyu-nesne maskesini
    # karıştırma; aksi halde klavye tuşları el yazısı bandına dönüşebilir.
    ink_source = blue_ink if np.count_nonzero(blue_ink) >= 80 else dark_ink
    ink = (ink_source.astype(np.uint8) * 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_OPEN, kernel)
    # A pen, logo or isolated scribble outside the sheet must not enlarge the
    # crop. Join nearby text rows vertically and retain the strongest multi-line
    # handwriting band.
    row_strength = np.count_nonzero(ink, axis=1)
    active_rows = (row_strength >= max(3, int(image.shape[1] * 0.003))).astype(np.uint8) * 255
    close_size = max(31, int(round(image.shape[0] * 0.055)) | 1)
    joined_rows = cv2.morphologyEx(
        active_rows.reshape(-1, 1),
        cv2.MORPH_CLOSE,
        np.ones((close_size, 1), dtype=np.uint8)
    ).reshape(-1)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (joined_rows > 0).astype(np.uint8).reshape(-1, 1), 8
    )
    bands = []
    for label in range(1, count):
        band_y = int(stats[label, cv2.CC_STAT_TOP])
        band_height = int(stats[label, cv2.CC_STAT_HEIGHT])
        band_points = cv2.findNonZero(ink[band_y:band_y + band_height])
        if band_points is None:
            continue
        _, _, band_width, _ = cv2.boundingRect(band_points)
        strength = int(row_strength[band_y:band_y + band_height].sum())
        # A large blue logo can contain more ink than a short handwritten
        # sentence. Handwriting normally spreads horizontally, whereas logos
        # and isolated marks are more compact. Score the horizontal span and
        # accept single-line text instead of requiring a tall multi-line band.
        if (band_height >= max(20, image.shape[0] * 0.025)
                and band_width >= image.shape[1] * 0.25):
            bands.append((band_width * np.sqrt(band_height), band_y, band_height))
    if bands:
        _, band_y, band_height = max(bands, key=lambda item: item[0])
        band_mask = np.zeros_like(ink)
        band_mask[band_y:band_y + band_height] = ink[band_y:band_y + band_height]
        ink = band_mask
    points = cv2.findNonZero(ink)
    if points is None or len(points) < 80:
        return image
    x, y, width, height = cv2.boundingRect(points)
    image_height, image_width = image.shape[:2]
    pad_x = max(int(image_width * 0.065), int(width * 0.07))
    pad_top = max(int(image_height * 0.07), int(height * 0.10))
    pad_bottom = max(int(image_height * 0.10), int(height * 0.15))
    x1 = max(0, x - pad_x)
    x2 = min(image_width, x + width + pad_x)
    y1 = max(0, y - pad_top)
    y2 = min(image_height, y + height + pad_bottom)
    # Yazı sayfanın üstündeyse kâğıdın kalan boş kısmını kesmeyelim. Sonuç
    # yalnız yazı şeridi değil, üzerinde düzeltme yapılacak gerçek sayfa olsun.
    below_start = min(image_height, y + height + max(8, int(image_height * .02)))
    if below_start < image_height:
        centre_left, centre_right = int(image_width * .3), int(image_width * .7)
        below = gray[below_start:image_height, centre_left:centre_right]
        if below.size and np.median(below) > 115:
            y2 = image_height
    if (x2 - x1) < 300 or (y2 - y1) < 300:
        return image
    # Never return a narrow off-centre strip. This means the colour detector
    # locked onto a logo or one thick word and would remove OCR context.
    crop_width_ratio = (x2 - x1) / image_width
    crop_centre_x = (x1 + x2) / 2.0
    if crop_width_ratio < 0.48 or abs(crop_centre_x - image_width / 2.0) > image_width * 0.24:
        return image
    return image[y1:y2, x1:x2]


def warp_paper(image, quad):
    top_left, top_right, bottom_right, bottom_left = quad
    output_width = int(round(max(
        np.linalg.norm(top_right - top_left),
        np.linalg.norm(bottom_right - bottom_left),
    )))
    output_height = int(round(max(
        np.linalg.norm(bottom_left - top_left),
        np.linalg.norm(bottom_right - top_right),
    )))
    if output_width < 300 or output_height < 300:
        return image

    destination = np.array([
        [0, 0],
        [output_width - 1, 0],
        [output_width - 1, output_height - 1],
        [0, output_height - 1],
    ], dtype=np.float32)
    transform = cv2.getPerspectiveTransform(quad.astype(np.float32), destination)
    warped = cv2.warpPerspective(image, transform, (output_width, output_height),
                                 flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)

    # Leave a small safety margin inside the detected paper edge. This removes
    # obvious desk/keyboard pixels without risking handwritten edge content.
    inset_x = max(1, int(round(output_width * 0.004)))
    inset_y = max(1, int(round(output_height * 0.004)))
    if output_width - inset_x * 2 > 300 and output_height - inset_y * 2 > 300:
        warped = warped[inset_y:output_height - inset_y, inset_x:output_width - inset_x]
    return warped


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: paper_crop.py input-image output-image")
    input_path, output_path = sys.argv[1:]
    image = cv2.imread(input_path, cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit("image could not be decoded")
    quad = find_paper(image)
    output = warp_paper(image, quad) if quad is not None else crop_around_handwriting(image)
    cropped = quad is not None or output.shape[:2] != image.shape[:2]
    # Full-resolution PNG photographs make every OCR/AI request unnecessarily
    # large. Preserve enough handwriting detail while capping transfer size.
    max_dimension = 2200
    scale = min(1.0, max_dimension / max(output.shape[:2]))
    if scale < 1.0:
        output = cv2.resize(output, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    parameters = [cv2.IMWRITE_JPEG_QUALITY, 90] if output_path.lower().endswith((".jpg", ".jpeg")) else [cv2.IMWRITE_PNG_COMPRESSION, 3]
    if not cv2.imwrite(output_path, output, parameters):
        raise SystemExit("output image could not be written")
    print(json.dumps({"cropped": bool(cropped)}))


if __name__ == "__main__":
    main()
