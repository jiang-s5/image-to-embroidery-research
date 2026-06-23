from __future__ import annotations

import argparse
import csv
import io
import json
import shutil
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


CANVAS_SIZE = 256


OPENCLIPART_ITEMS = [
    ("ocp_001", "flower", "306377", "simple-flower-silhouette-shape"),
    ("ocp_002", "butterfly", "317916", "delicate-butterfly-silhouette"),
    ("ocp_003", "cat", "1198", "cat-silhouette"),
    ("ocp_004", "koi_fish", "336165", "koi-fish-silhouette"),
    ("ocp_005", "public_domain_logo", "211358", "public-domain-logo"),
    ("ocp_006", "butterfly_2", "245816", "butterfly-silhouette-2"),
    ("ocp_007", "sitting_cat", "169097", "sitting-cat-silhouette"),
    ("ocp_008", "fish", "323907", "fish-silhouette-monochrome-clip-art"),
    ("ocp_009", "flower_2", "245575", "stylized-flower-silhouette"),
    ("ocp_010", "thought_bubble", "218059", "thought-bubble"),
]


OPENMOJI_ITEMS = [
    ("omj_001", "heart", ("2764-FE0F", "2764")),
    ("omj_002", "star", ("2B50",)),
    ("omj_003", "butterfly", ("1F98B",)),
    ("omj_004", "fish", ("1F41F",)),
    ("omj_005", "sun", ("2600-FE0F", "2600")),
    ("omj_006", "cherry_blossom", ("1F338",)),
    ("omj_007", "ribbon", ("1F380",)),
    ("omj_008", "paw_prints", ("1F43E",)),
    ("omj_009", "lady_beetle", ("1F41E",)),
    ("omj_010", "seedling", ("1F331",)),
]


QUICKDRAW_ITEMS = [
    ("qd_001", "cat"),
    ("qd_002", "dog"),
    ("qd_003", "fish"),
    ("qd_004", "flower"),
    ("qd_005", "bicycle"),
    ("qd_006", "bird"),
    ("qd_007", "moon"),
    ("qd_008", "star"),
    ("qd_009", "tree"),
    ("qd_010", "umbrella"),
]


TEXT_ITEMS = [
    ("txt_001", "AI"),
    ("txt_002", "DST"),
    ("txt_003", "WKU"),
    ("txt_004", "2026"),
    ("txt_005", "HELLO"),
]


SOURCE_NOTES = {
    "Openclipart": {
        "license": "CC0-1.0",
        "url": "https://openclipart.org/share",
        "needs_attribution": "false",
        "attribution_text": "",
        "notes": "Public-domain/CC0 clipart. Fixed Openclipart IDs are recorded in manifest.csv.",
    },
    "OpenMoji": {
        "license": "CC BY-SA 4.0",
        "url": "https://openmoji.org/",
        "needs_attribution": "true",
        "attribution_text": "OpenMoji graphics, CC BY-SA 4.0",
        "notes": "OpenMoji color PNG assets from the official GitHub export.",
    },
    "QuickDraw": {
        "license": "CC BY 4.0",
        "url": "https://github.com/googlecreativelab/quickdraw-dataset",
        "needs_attribution": "true",
        "attribution_text": "Quick, Draw! Dataset by Google, CC BY 4.0",
        "notes": "Deterministic rendered vector drawings selected from simplified ndjson.",
    },
    "Oxford-IIIT Pet": {
        "license": "CC BY-SA 4.0 or dataset-specific attribution terms",
        "url": "https://www.robots.ox.ac.uk/~vgg/data/pets/",
        "needs_attribution": "true",
        "attribution_text": "Oxford-IIIT Pet Dataset",
        "notes": "Use image + trimap annotations as a real-photo mask benchmark when available locally.",
    },
    "Rendered text": {
        "license": "Font-specific OFL/Apache; generated text images are local artifacts.",
        "url": "https://fonts.google.com/",
        "needs_attribution": "true",
        "attribution_text": "Rendered text using the font recorded in manifest.csv",
        "notes": "Use a specific open font path for a formal public release.",
    },
}


@dataclass
class ManifestRow:
    sample_id: str
    phase: str
    source_name: str
    source_url: str
    license: str
    license_notes: str
    selector_type: str
    source_identifier: str
    category: str
    subtype: str
    raw_path: str
    image_png_path: str
    mask_path: str
    edge_sobel_path: str
    edge_canny_path: str
    dt_path: str
    skeleton_path: str
    width_px: int
    height_px: int
    has_external_mask: str
    needs_attribution: str
    attribution_text: str
    target_split: str
    notes: str


def request_bytes(url: str, timeout: float = 30.0) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "image-to-embroidery-research/benchmark-builder"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read()


def ensure_rgba_square(image: Image.Image, size: int = CANVAS_SIZE) -> Image.Image:
    image = image.convert("RGBA")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    offset = ((size - image.width) // 2, (size - image.height) // 2)
    canvas.alpha_composite(image, offset)
    return canvas


def alpha_or_nonwhite_mask(rgba: Image.Image) -> np.ndarray:
    rgba = rgba.convert("RGBA")
    alpha = np.asarray(rgba.getchannel("A"), dtype=np.uint8)
    rgb = np.asarray(rgba.convert("RGB"), dtype=np.uint8)
    nonwhite = (np.any(rgb < 245, axis=2).astype(np.uint8) * 255)
    mask = np.maximum(alpha, nonwhite)
    return (mask > 8).astype(np.uint8) * 255


def refine_mask(mask: np.ndarray) -> np.ndarray:
    mask = (mask > 0).astype(np.uint8) * 255
    open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    clean = np.zeros_like(mask)
    for label in range(1, num_labels):
        if stats[label, cv2.CC_STAT_AREA] >= 16:
            clean[labels == label] = 255
    return clean


def skeletonize_cv(mask: np.ndarray) -> np.ndarray:
    img = (mask > 0).astype(np.uint8) * 255
    skel = np.zeros_like(img)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(img) > 0:
        eroded = cv2.erode(img, element)
        opened = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
    return skel


def write_sample_artifacts(
    output_dir: Path,
    sample_id: str,
    subset_dir: str,
    image: Image.Image,
    raw_bytes: bytes | None = None,
    raw_suffix: str = ".png",
) -> dict[str, str]:
    sample_dir = output_dir / subset_dir
    prior_dir = output_dir / "priors" / sample_id
    raw_dir = output_dir / "raw" / subset_dir
    sample_dir.mkdir(parents=True, exist_ok=True)
    prior_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    rgba = ensure_rgba_square(image, CANVAS_SIZE)
    mask = refine_mask(alpha_or_nonwhite_mask(rgba))

    rgb = Image.new("RGB", rgba.size, (255, 255, 255))
    rgb.paste(rgba.convert("RGB"), mask=rgba.getchannel("A"))

    image_rel = Path(subset_dir) / f"{sample_id}.png"
    mask_rel = Path(subset_dir) / f"{sample_id}_mask.png"
    sobel_rel = Path("priors") / sample_id / "sobel.png"
    canny_rel = Path("priors") / sample_id / "canny.png"
    dt_rel = Path("priors") / sample_id / "dt.npy"
    skeleton_rel = Path("priors") / sample_id / "skeleton.png"
    raw_rel = Path("raw") / subset_dir / f"{sample_id}{raw_suffix}"

    rgb.save(output_dir / image_rel)
    Image.fromarray(mask).save(output_dir / mask_rel)

    gray = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    sy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(sx, sy)
    if mag.max() > 0:
        mag = (mag / mag.max() * 255).astype(np.uint8)
    else:
        mag = np.zeros_like(gray, dtype=np.uint8)
    canny = cv2.Canny(blur, 64, 160)
    dt = cv2.distanceTransform((mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    skeleton = skeletonize_cv(mask)

    Image.fromarray(mag).save(output_dir / sobel_rel)
    Image.fromarray(canny).save(output_dir / canny_rel)
    np.save(output_dir / dt_rel, dt)
    Image.fromarray(skeleton).save(output_dir / skeleton_rel)

    if raw_bytes is not None:
        (output_dir / raw_rel).write_bytes(raw_bytes)
    else:
        rgba.save(output_dir / raw_rel)

    return {
        "raw_path": raw_rel.as_posix(),
        "image_png_path": image_rel.as_posix(),
        "mask_path": mask_rel.as_posix(),
        "edge_sobel_path": sobel_rel.as_posix(),
        "edge_canny_path": canny_rel.as_posix(),
        "dt_path": dt_rel.as_posix(),
        "skeleton_path": skeleton_rel.as_posix(),
    }


def make_row(
    sample_id: str,
    phase: str,
    source_name: str,
    source_url: str,
    selector_type: str,
    source_identifier: str,
    category: str,
    subtype: str,
    paths: dict[str, str],
    notes: str,
) -> ManifestRow:
    info = SOURCE_NOTES[source_name]
    return ManifestRow(
        sample_id=sample_id,
        phase=phase,
        source_name=source_name,
        source_url=source_url,
        license=info["license"],
        license_notes=info["notes"],
        selector_type=selector_type,
        source_identifier=source_identifier,
        category=category,
        subtype=subtype,
        raw_path=paths["raw_path"],
        image_png_path=paths["image_png_path"],
        mask_path=paths["mask_path"],
        edge_sobel_path=paths["edge_sobel_path"],
        edge_canny_path=paths["edge_canny_path"],
        dt_path=paths["dt_path"],
        skeleton_path=paths["skeleton_path"],
        width_px=CANVAS_SIZE,
        height_px=CANVAS_SIZE,
        has_external_mask="true",
        needs_attribution=info["needs_attribution"],
        attribution_text=info["attribution_text"],
        target_split="public_benchmark_v1",
        notes=notes,
    )


def openclipart_image_url(clip_id: str) -> str:
    return f"https://openclipart.org/image/2000px/{clip_id}"


def openclipart_detail_url(clip_id: str, slug: str) -> str:
    return f"https://openclipart.org/detail/{clip_id}/{slug}"


def download_openclipart(output_dir: Path, limit: int) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for sample_id, category, clip_id, slug in OPENCLIPART_ITEMS[:limit]:
        image_url = openclipart_image_url(clip_id)
        detail_url = openclipart_detail_url(clip_id, slug)
        try:
            payload = request_bytes(image_url)
            image = Image.open(io.BytesIO(payload))
        except Exception as exc:
            print(f"warning: failed to download Openclipart {clip_id}: {exc}")
            continue
        paths = write_sample_artifacts(output_dir, sample_id, "openclipart_10", image, payload, ".png")
        rows.append(
            make_row(
                sample_id=sample_id,
                phase="core" if len(rows) < 5 else "full",
                source_name="Openclipart",
                source_url=detail_url,
                selector_type="fixed_openclipart_id",
                source_identifier=clip_id,
                category=category,
                subtype="vector_silhouette",
                paths=paths,
                notes=f"downloaded PNG render from {image_url}; detail page keeps the fixed source ID",
            )
        )
    return rows


def openmoji_url(codepoint: str) -> str:
    return f"https://raw.githubusercontent.com/hfg-gmuend/openmoji/master/color/618x618/{codepoint}.png"


def download_openmoji(output_dir: Path, limit: int, timeout: float = 20.0) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for sample_id, category, codepoints in OPENMOJI_ITEMS[:limit]:
        payload = None
        url = ""
        last_error = ""
        for codepoint in codepoints:
            url = openmoji_url(codepoint)
            try:
                payload = request_bytes(url, timeout)
                break
            except Exception as exc:
                last_error = str(exc)
        if payload is None:
            print(f"warning: failed to download OpenMoji {sample_id}/{category}: {last_error}")
            continue
        image = Image.open(io.BytesIO(payload))
        paths = write_sample_artifacts(output_dir, sample_id, "openmoji_10", image, payload, ".png")
        rows.append(
            make_row(
                sample_id=sample_id,
                phase="core" if len(rows) < 5 else "full",
                source_name="OpenMoji",
                source_url=url,
                selector_type="fixed_openmoji_codepoint",
                source_identifier="+".join(codepoints),
                category=category,
                subtype="multicolor_icon",
                paths=paths,
                notes="downloaded from OpenMoji GitHub color PNG export; attribution required",
            )
        )
    return rows


def drawing_bbox(drawing: list[list[list[int]]]) -> tuple[int, int, int, int]:
    xs: list[int] = []
    ys: list[int] = []
    for stroke in drawing:
        if len(stroke) >= 2:
            xs.extend(stroke[0])
            ys.extend(stroke[1])
    if not xs or not ys:
        return (0, 0, 0, 0)
    return (min(xs), min(ys), max(xs), max(ys))


def bbox_area_ratio(box: tuple[int, int, int, int], canvas: int = 256) -> float:
    x0, y0, x1, y1 = box
    return max(0, x1 - x0) * max(0, y1 - y0) / float(canvas * canvas)


def quickdraw_url(category: str) -> str:
    return f"https://storage.googleapis.com/quickdraw_dataset/full/simplified/{category}.ndjson"


def select_quickdraw(category: str, max_lines: int = 30000, select_min_key: bool = False) -> dict:
    url = quickdraw_url(category)
    best: dict | None = None
    request = urllib.request.Request(url, headers={"User-Agent": "image-to-embroidery-research/benchmark-builder"})
    with urllib.request.urlopen(request, timeout=60.0) as response:
        for line_no, raw in enumerate(response, start=1):
            if line_no > max_lines:
                break
            try:
                obj = json.loads(raw.decode("utf-8"))
            except Exception:
                continue
            if not obj.get("recognized", False):
                continue
            drawing = obj.get("drawing", [])
            if not (2 <= len(drawing) <= 8):
                continue
            ratio = bbox_area_ratio(drawing_bbox(drawing))
            if not (0.30 <= ratio <= 0.85):
                continue
            key_id = int(obj.get("key_id", 0))
            if not select_min_key:
                return obj
            if best is None or key_id < int(best["key_id"]):
                best = obj
    if best is None:
        raise RuntimeError(f"No valid QuickDraw sample found for {category}")
    return best


def render_quickdraw(drawing: list[list[list[int]]], size: int = CANVAS_SIZE) -> Image.Image:
    image = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    draw = ImageDraw.Draw(image)
    scale = size / 256.0
    for stroke in drawing:
        if len(stroke) < 2:
            continue
        points = [(int(round(x * scale)), int(round(y * scale))) for x, y in zip(stroke[0], stroke[1])]
        if len(points) >= 2:
            try:
                draw.line(points, fill=(20, 35, 60, 255), width=7, joint="curve")
            except TypeError:
                draw.line(points, fill=(20, 35, 60, 255), width=7)
    return image


def download_quickdraw(output_dir: Path, limit: int, max_lines: int = 30000, select_min_key: bool = False) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    for sample_id, category in QUICKDRAW_ITEMS[:limit]:
        try:
            obj = select_quickdraw(category, max_lines=max_lines, select_min_key=select_min_key)
        except Exception as exc:
            print(f"warning: failed to select QuickDraw {category}: {exc}")
            continue
        image = render_quickdraw(obj["drawing"])
        raw = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        paths = write_sample_artifacts(output_dir, sample_id, "quickdraw_10", image, raw, ".json")
        rows.append(
            make_row(
                sample_id=sample_id,
                phase="core" if len(rows) < 5 else "full",
                source_name="QuickDraw",
                source_url=quickdraw_url(category),
                selector_type="deterministic_rule",
                source_identifier=f"{category}|key_id={obj.get('key_id')}",
                category=category,
                subtype="line_drawing",
                paths=paths,
                notes=(
                    "recognized=true, 2<=stroke_count<=8, bbox area ratio 0.30-0.85; "
                    + ("smallest key_id among scanned rows" if select_min_key else "first valid row for fast deterministic build")
                ),
            )
        )
    return rows


def load_font(font_path: str, size: int) -> ImageFont.ImageFont:
    if font_path:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            pass
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf",
        r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\calibrib.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def render_text_samples(output_dir: Path, font_path: str, limit: int) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    font = load_font(font_path, 88)
    for sample_id, text in TEXT_ITEMS[:limit]:
        image = Image.new("RGBA", (CANVAS_SIZE, CANVAS_SIZE), (255, 255, 255, 0))
        draw = ImageDraw.Draw(image)
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=2)
        x = (CANVAS_SIZE - (bbox[2] - bbox[0])) // 2 - bbox[0]
        y = (CANVAS_SIZE - (bbox[3] - bbox[1])) // 2 - bbox[1]
        draw.text((x, y), text, font=font, fill=(20, 30, 45, 255), stroke_width=2, stroke_fill=(255, 255, 255, 255))
        raw = json.dumps({"text": text, "font_path": font_path or "system_fallback"}, ensure_ascii=False).encode("utf-8")
        paths = write_sample_artifacts(output_dir, sample_id, "text_render_5", image, raw, ".json")
        rows.append(
            make_row(
                sample_id=sample_id,
                phase="core" if len(rows) < 2 else "full",
                source_name="Rendered text",
                source_url=SOURCE_NOTES["Rendered text"]["url"],
                selector_type="rendered_text",
                source_identifier=text,
                category=text,
                subtype="text",
                paths=paths,
                notes="locally rendered text; record a specific open font before formal public release",
            )
        )
    return rows


def copy_public_images(
    source_dir: Path,
    output_dir: Path,
    source_name: str,
    subset_dir: str,
    prefix: str,
    limit: int,
    subtype: str,
) -> list[ManifestRow]:
    rows: list[ManifestRow] = []
    if not source_dir.exists():
        return rows
    files = sorted(
        path
        for path in source_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    )
    for index, path in enumerate(files[:limit], start=1):
        sample_id = f"{prefix}_{index:03d}"
        payload = path.read_bytes()
        image = Image.open(path)
        paths = write_sample_artifacts(output_dir, sample_id, subset_dir, image, payload, path.suffix.lower())
        rows.append(
            make_row(
                sample_id=sample_id,
                phase="local",
                source_name=source_name,
                source_url=SOURCE_NOTES[source_name]["url"],
                selector_type="local_curated_file",
                source_identifier=path.name,
                category=path.stem,
                subtype=subtype,
                paths=paths,
                notes=f"copied from local file: {path}",
            )
        )
    return rows


def write_manifest(rows: Iterable[ManifestRow], output_dir: Path) -> None:
    rows = list(rows)
    manifest_path = output_dir / "manifest.csv"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(ManifestRow.__dataclass_fields__.keys())
    with manifest_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def write_readme(output_dir: Path, rows: list[ManifestRow]) -> None:
    counts: dict[str, int] = {}
    for row in rows:
        counts[row.source_name] = counts.get(row.source_name, 0) + 1
    lines = [
        "# Public Benchmark v1",
        "",
        "This directory is an evaluation benchmark for image-to-embroidery DST/PES generation.",
        "It is intended for executability and visual-risk evaluation, not professional DST ground-truth matching.",
        "",
        "## Composition",
        "",
        "| Source | Samples | License / Terms |",
        "| --- | ---: | --- |",
    ]
    for source, info in SOURCE_NOTES.items():
        lines.append(f"| {source} | {counts.get(source, 0)} | {info['license']} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "Each sample has a normalized PNG, foreground mask, Sobel edge image, Canny edge image, distance transform, and skeleton image.",
            "",
            "## Manifest",
            "",
            "`manifest.csv` records source identifiers, license notes, generated artifact paths, and attribution requirements.",
            "",
            "## Boundary",
            "",
            "This benchmark is for external executability and visual-risk evaluation. It does not provide professional DST ground truth.",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare public_benchmark_v1 for image-to-embroidery evaluation.")
    parser.add_argument("--output-dir", default="datasets/public_benchmark_v1")
    parser.add_argument("--openclipart-dir", default="", help="Optional local directory with curated Openclipart PNG/JPG files.")
    parser.add_argument("--openmoji-dir", default="", help="Optional local directory with curated OpenMoji PNG files.")
    parser.add_argument("--quickdraw-dir", default="", help="Optional local directory with rendered QuickDraw PNG files.")
    parser.add_argument("--oxford-pet-dir", default="", help="Optional local directory with selected Oxford Pet images.")
    parser.add_argument("--download-openclipart", action="store_true", help="Download fixed Openclipart PNG renders by ID.")
    parser.add_argument("--download-openmoji", action="store_true", help="Download fixed OpenMoji PNGs from official GitHub raw URLs.")
    parser.add_argument("--download-quickdraw", action="store_true", help="Download and render deterministic QuickDraw samples.")
    parser.add_argument("--quickdraw-max-lines", type=int, default=30000, help="Maximum ndjson rows scanned per QuickDraw category.")
    parser.add_argument("--quickdraw-select-min-key", action="store_true", help="Scan rows and select the smallest valid key_id instead of the first valid row.")
    parser.add_argument("--font-path", default="", help="Optional TTF/OTF font path for rendered text samples.")
    parser.add_argument("--clean", action="store_true", help="Delete and recreate output dir before building.")
    parser.add_argument("--openclipart-limit", type=int, default=10)
    parser.add_argument("--openmoji-limit", type=int, default=10)
    parser.add_argument("--quickdraw-limit", type=int, default=8)
    parser.add_argument("--oxford-pet-limit", type=int, default=8)
    parser.add_argument("--text-limit", type=int, default=4)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ManifestRow] = []
    if args.download_openclipart:
        rows.extend(download_openclipart(output_dir, args.openclipart_limit))
    if args.openclipart_dir:
        rows.extend(copy_public_images(Path(args.openclipart_dir), output_dir, "Openclipart", "openclipart_local", "ocp_local", args.openclipart_limit, "vector_or_logo"))

    if args.download_openmoji:
        rows.extend(download_openmoji(output_dir, args.openmoji_limit))
    elif args.openmoji_dir:
        rows.extend(copy_public_images(Path(args.openmoji_dir), output_dir, "OpenMoji", "openmoji_local", "omj_local", args.openmoji_limit, "multicolor_icon"))

    if args.download_quickdraw:
        rows.extend(download_quickdraw(output_dir, args.quickdraw_limit, args.quickdraw_max_lines, args.quickdraw_select_min_key))
    if args.quickdraw_dir:
        rows.extend(copy_public_images(Path(args.quickdraw_dir), output_dir, "QuickDraw", "quickdraw_local", "qd_local", args.quickdraw_limit, "line_drawing"))

    if args.oxford_pet_dir:
        rows.extend(copy_public_images(Path(args.oxford_pet_dir), output_dir, "Oxford-IIIT Pet", "oxford_pet_local", "pet_local", args.oxford_pet_limit, "real_photo"))

    rows.extend(render_text_samples(output_dir, args.font_path, args.text_limit))

    write_manifest(rows, output_dir)
    write_readme(output_dir, rows)
    (output_dir / "source_licenses.json").write_text(json.dumps(SOURCE_NOTES, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "samples": len(rows), "manifest": str(output_dir / "manifest.csv")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
