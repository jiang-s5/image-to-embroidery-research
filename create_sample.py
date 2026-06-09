from pathlib import Path

from PIL import Image, ImageDraw


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "examples"
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(image)

    draw.rounded_rectangle((95, 95, 545, 325), radius=42, fill=(22, 96, 168))
    draw.ellipse((165, 135, 315, 285), fill=(243, 180, 42))
    draw.polygon([(350, 150), (490, 210), (350, 270)], fill=(235, 70, 70))
    draw.line((135, 345, 505, 345), fill=(20, 20, 20), width=16)
    draw.line((180, 365, 460, 365), fill=(20, 20, 20), width=9)

    path = out_dir / "sample_logo.png"
    image.save(path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
