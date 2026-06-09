from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (640, 640), "white")
    draw = ImageDraw.Draw(image)

    black = (35, 32, 28)
    white = (252, 249, 239)
    orange = (224, 115, 32)
    tan = (244, 191, 91)
    red = (230, 55, 48)
    blue = (41, 116, 166)
    light_blue = (91, 169, 205)
    yellow = (247, 205, 70)

    # Main rounded cat head.
    draw.rounded_rectangle((135, 116, 520, 440), radius=96, fill=white, outline=black, width=12)

    # Left fox-like orange face patch and ears.
    draw.polygon([(168, 116), (248, 46), (282, 146)], fill=white, outline=black)
    draw.polygon([(399, 119), (458, 58), (472, 154)], fill=white, outline=black)
    draw.polygon([(184, 128), (255, 72), (264, 164)], fill=orange)
    draw.polygon([(409, 129), (454, 83), (459, 150)], fill=orange)

    draw.pieslice((112, 122, 356, 388), start=110, end=300, fill=orange, outline=black, width=10)
    draw.polygon([(202, 179), (232, 224), (282, 221), (243, 252), (257, 305), (213, 271), (171, 304), (186, 252), (146, 221), (196, 224)], fill=tan)

    # White face overlay on the right keeps the cat face clear.
    draw.pieslice((230, 116, 526, 438), start=260, end=95, fill=white, outline=black, width=10)

    # Bow.
    draw.ellipse((379, 93, 454, 168), fill=red, outline=black, width=7)
    draw.ellipse((445, 94, 520, 170), fill=red, outline=black, width=7)
    draw.ellipse((434, 116, 471, 153), fill=red, outline=black, width=6)

    # Eyes, nose, whiskers.
    draw.ellipse((224, 235, 252, 266), fill=black)
    draw.ellipse((402, 237, 428, 266), fill=black)
    draw.ellipse((315, 272, 342, 292), fill=yellow, outline=black, width=5)
    for y in [250, 278, 306]:
        draw.line((350, y, 450, y - 16), fill=black, width=5)
        draw.line((167, y - 10, 84, y - 30), fill=black, width=5)

    # Lower cup/body.
    draw.rounded_rectangle((248, 406, 410, 500), radius=32, fill=blue, outline=black, width=8)
    draw.ellipse((263, 382, 395, 435), fill=light_blue, outline=black, width=7)
    draw.arc((395, 420, 470, 490), start=265, end=95, fill=black, width=8)
    draw.arc((405, 432, 455, 478), start=265, end=95, fill=light_blue, width=14)

    # Small cup decorations.
    for x, y, r, color in [
        (282, 440, 8, tan),
        (316, 464, 7, red),
        (358, 442, 7, yellow),
        (382, 466, 6, tan),
    ]:
        draw.ellipse((x - r, y - r, x + r, y + r), fill=color)

    path = out_dir / "user_cat_fox_reference.png"
    image.save(path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
