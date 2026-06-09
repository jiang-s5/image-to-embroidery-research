from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PALETTES = [
    [(20, 88, 150), (241, 181, 51), (236, 83, 75), (22, 24, 29)],
    [(28, 130, 105), (246, 214, 96), (56, 62, 82), (238, 242, 240)],
    [(38, 64, 139), (86, 188, 171), (245, 91, 67), (245, 245, 245)],
    [(34, 38, 42), (93, 170, 220), (239, 197, 75), (231, 83, 95)],
    [(108, 67, 152), (49, 151, 149), (244, 162, 97), (250, 250, 250)],
    [(17, 90, 64), (225, 111, 68), (247, 211, 97), (40, 42, 54)],
    [(42, 49, 120), (219, 75, 75), (245, 196, 81), (237, 238, 242)],
    [(0, 119, 182), (0, 180, 216), (255, 183, 3), (33, 37, 41)],
]

LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"
CATEGORIES = [
    "badge",
    "monogram",
    "shield",
    "orbit",
    "leaf",
    "geometric",
    "flower",
    "stripe",
]


@dataclass
class LogoRecord:
    logo_id: str
    filename: str
    category: str
    colors_hex: str
    complexity: str
    recommended_max_colors: int
    target_width_mm: int
    notes: str


def rgb_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def get_font(size: int, bold: bool = True) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf",
        r"C:\Windows\Fonts\segoeuib.ttf" if bold else r"C:\Windows\Fonts\segoeui.ttf",
        r"C:\Windows\Fonts\calibrib.ttf" if bold else r"C:\Windows\Fonts\calibri.ttf",
    ]
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def centered_text(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    fill: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = box
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = left + (right - left - w) / 2 - bbox[0]
    y = top + (bottom - top - h) / 2 - bbox[1]
    draw.text((x, y), text, fill=fill, font=font)


def regular_polygon(
    cx: float,
    cy: float,
    radius: float,
    sides: int,
    rotation: float = 0.0,
) -> list[tuple[float, float]]:
    return [
        (
            cx + math.cos(rotation + i * 2 * math.pi / sides) * radius,
            cy + math.sin(rotation + i * 2 * math.pi / sides) * radius,
        )
        for i in range(sides)
    ]


def draw_badge(draw: ImageDraw.ImageDraw, rng: random.Random, colors: list[tuple[int, int, int]]) -> str:
    bg, accent, warm, dark = colors
    draw.rounded_rectangle((105, 90, 535, 330), radius=46, fill=bg)
    draw.rounded_rectangle((128, 112, 512, 308), radius=32, outline=accent, width=14)
    draw.ellipse((242, 135, 398, 291), fill=warm)
    draw.rectangle((110, 238, 530, 286), fill=dark)
    text = "".join(rng.sample(LETTERS, 2))
    centered_text(draw, (110, 231, 530, 290), text, (255, 255, 255), get_font(52))
    return f"rounded badge with initials {text}"


def draw_monogram(draw: ImageDraw.ImageDraw, rng: random.Random, colors: list[tuple[int, int, int]]) -> str:
    bg, accent, warm, dark = colors
    draw.ellipse((160, 70, 480, 390), fill=bg)
    draw.ellipse((188, 98, 452, 362), outline=accent, width=18)
    draw.polygon(regular_polygon(320, 230, 86, 3, -math.pi / 2), fill=warm)
    text = "".join(rng.sample(LETTERS, 2))
    centered_text(draw, (185, 105, 455, 355), text, dark, get_font(105))
    return f"circular monogram {text}"


def draw_shield(draw: ImageDraw.ImageDraw, rng: random.Random, colors: list[tuple[int, int, int]]) -> str:
    bg, accent, warm, dark = colors
    shield = [(320, 58), (488, 120), (458, 312), (320, 392), (182, 312), (152, 120)]
    draw.polygon(shield, fill=bg)
    inner = [(320, 94), (445, 140), (424, 292), (320, 354), (216, 292), (195, 140)]
    draw.polygon(inner, outline=accent, width=12)
    draw.polygon([(320, 132), (378, 246), (320, 221), (262, 246)], fill=warm)
    draw.rectangle((267, 263, 373, 294), fill=dark)
    return "shield mark with center symbol"


def draw_orbit(draw: ImageDraw.ImageDraw, rng: random.Random, colors: list[tuple[int, int, int]]) -> str:
    bg, accent, warm, dark = colors
    draw.ellipse((210, 120, 430, 340), fill=bg)
    for angle, color in [(0, accent), (28, warm), (-28, dark)]:
        draw.ellipse((126, 168, 514, 292), outline=color, width=10)
        # Rotate by redrawing on transparent layer would add complexity; varied ovals still test curves.
    draw.ellipse((292, 202, 348, 258), fill=(255, 255, 255))
    draw.ellipse((305, 215, 335, 245), fill=warm)
    return "orbit mark with nested ellipses"


def draw_leaf(draw: ImageDraw.ImageDraw, rng: random.Random, colors: list[tuple[int, int, int]]) -> str:
    bg, accent, warm, dark = colors
    draw.rounded_rectangle((112, 92, 528, 328), radius=36, fill=(255, 255, 255))
    draw.pieslice((185, 85, 455, 355), start=35, end=215, fill=bg)
    draw.pieslice((185, 85, 455, 355), start=215, end=35, fill=accent)
    draw.line((235, 282, 405, 128), fill=dark, width=12)
    draw.arc((168, 80, 470, 365), start=35, end=215, fill=warm, width=10)
    return "two-tone leaf mark"


def draw_geometric(draw: ImageDraw.ImageDraw, rng: random.Random, colors: list[tuple[int, int, int]]) -> str:
    bg, accent, warm, dark = colors
    sides = rng.choice([5, 6, 8])
    draw.polygon(regular_polygon(320, 220, 154, sides, math.pi / sides), fill=bg)
    draw.polygon(regular_polygon(320, 220, 104, sides, 0), fill=accent)
    draw.polygon(regular_polygon(320, 220, 58, sides, math.pi / sides), fill=warm)
    draw.line((210, 340, 430, 100), fill=dark, width=15)
    return f"{sides}-sided geometric mark"


def draw_flower(draw: ImageDraw.ImageDraw, rng: random.Random, colors: list[tuple[int, int, int]]) -> str:
    bg, accent, warm, dark = colors
    for i in range(8):
        angle = i * math.pi / 4
        cx = 320 + math.cos(angle) * 84
        cy = 220 + math.sin(angle) * 84
        draw.ellipse((cx - 58, cy - 34, cx + 58, cy + 34), fill=bg if i % 2 == 0 else accent)
    draw.ellipse((256, 156, 384, 284), fill=warm)
    draw.ellipse((287, 187, 353, 253), fill=dark)
    return "radial flower mark"


def draw_stripe(draw: ImageDraw.ImageDraw, rng: random.Random, colors: list[tuple[int, int, int]]) -> str:
    bg, accent, warm, dark = colors
    draw.rounded_rectangle((105, 100, 535, 320), radius=30, fill=bg)
    for i, color in enumerate([accent, warm, dark, accent]):
        x0 = 145 + i * 82
        draw.polygon([(x0, 100), (x0 + 74, 100), (x0 + 22, 320), (x0 - 52, 320)], fill=color)
    draw.ellipse((246, 146, 394, 294), fill=(255, 255, 255))
    centered_text(draw, (246, 146, 394, 294), rng.choice(LETTERS), dark, get_font(82))
    return "slanted stripe mark"


DRAWERS = {
    "badge": draw_badge,
    "monogram": draw_monogram,
    "shield": draw_shield,
    "orbit": draw_orbit,
    "leaf": draw_leaf,
    "geometric": draw_geometric,
    "flower": draw_flower,
    "stripe": draw_stripe,
}


def generate_logo(index: int, out_dir: Path, rng: random.Random) -> LogoRecord:
    category = CATEGORIES[index % len(CATEGORIES)]
    palette = PALETTES[(index + rng.randrange(len(PALETTES))) % len(PALETTES)].copy()
    rng.shuffle(palette)

    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)

    # A faint construction-safe background ring makes some masks non-trivial while staying removable.
    if rng.random() < 0.35:
        draw.ellipse((110, 30, 530, 450), outline=(235, 238, 240), width=10)

    notes = DRAWERS[category](draw, rng, palette)
    logo_id = f"logo_{index + 1:04d}_{category}"
    filename = f"{logo_id}.png"
    path = out_dir / filename
    image.save(path)
    complexity = "low" if category in {"badge", "stripe", "shield"} else "medium"
    return LogoRecord(
        logo_id=logo_id,
        filename=filename,
        category=category,
        colors_hex=";".join(rgb_hex(color) for color in palette),
        complexity=complexity,
        recommended_max_colors=5,
        target_width_mm=80,
        notes=notes,
    )


def write_manifest(records: list[LogoRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(LogoRecord.__annotations__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(record.__dict__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect a reproducible logo-like dataset for Stage 1.")
    parser.add_argument("--count", type=int, default=100, help="Number of logo samples to create.")
    parser.add_argument("--seed", type=int, default=3830, help="Deterministic random seed.")
    parser.add_argument(
        "--output",
        default="datasets/logos",
        help="Dataset folder containing images/ and manifest.csv.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    root = Path(args.output)
    image_dir = root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    records = [generate_logo(i, image_dir, rng) for i in range(args.count)]
    write_manifest(records, root / "manifest.csv")
    print(f"created {len(records)} logos in {image_dir}")
    print(f"manifest: {root / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
