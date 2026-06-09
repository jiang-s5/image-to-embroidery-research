from pathlib import Path

from PIL import Image, ImageDraw


S = 3


def sc(points):
    return [(int(x * S), int(y * S)) for x, y in points]


def box(x0, y0, x1, y1):
    return (int(x0 * S), int(y0 * S), int(x1 * S), int(y1 * S))


def main() -> int:
    out_dir = Path(__file__).resolve().parent / "inputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (900 * S, 900 * S), (248, 225, 154))
    draw = ImageDraw.Draw(image)

    brown = (139, 82, 52)
    dark_brown = (91, 58, 38)
    cream = (250, 250, 248)
    red = (238, 22, 18)
    orange = (236, 139, 35)
    orange2 = (249, 176, 55)
    yellow = (255, 218, 45)
    blue = (29, 146, 198)
    pale_shadow = (239, 212, 166)
    doodle = (206, 143, 37)

    # Background doodles. These are deliberately sparse and simple so the
    # embroidery conversion does not waste all stitches on the background.
    for pts in [
        [(30, 88), (58, 88), (79, 100)],
        [(165, 175), (190, 175), (208, 186)],
        [(400, 90), (430, 76), (465, 92), (515, 100)],
        [(710, 72), (740, 88), (775, 72)],
        [(786, 182), (826, 176), (858, 195)],
        [(92, 472), (125, 462), (160, 473)],
        [(765, 468), (810, 470), (858, 478)],
        [(42, 610), (92, 618), (145, 606)],
        [(234, 780), (260, 780), (278, 790)],
        [(40, 820), (118, 812), (175, 824)],
        [(400, 816), (468, 810), (540, 823)],
        [(640, 818), (720, 807), (810, 826)],
    ]:
        draw.line(sc(pts), fill=doodle, width=14 * S, joint="curve")
    for cx, cy in [(96, 676), (238, 705), (742, 690), (804, 652)]:
        draw.ellipse(box(cx - 18, cy - 9, cx + 18, cy + 9), outline=(224, 126, 19), width=7 * S)
    for pts in [
        [(80, 650), (102, 628), (125, 640), (114, 668), (86, 668)],
        [(720, 654), (742, 632), (765, 646), (750, 674), (724, 671)],
    ]:
        draw.polygon(sc(pts), fill=(244, 174, 54), outline=(225, 125, 12))

    # Soft ground shadow.
    draw.ellipse(box(265, 580, 660, 790), fill=pale_shadow)

    # Legs behind body.
    draw.rounded_rectangle(box(356, 704, 463, 782), radius=22 * S, fill=brown)
    draw.rounded_rectangle(box(470, 704, 577, 782), radius=22 * S, fill=brown)
    draw.rounded_rectangle(box(371, 719, 458, 749), radius=13 * S, fill=cream)
    draw.rounded_rectangle(box(482, 719, 565, 749), radius=13 * S, fill=cream)

    # Body and arms.
    draw.rounded_rectangle(box(305, 558, 610, 745), radius=55 * S, fill=brown)
    draw.rounded_rectangle(box(349, 563, 557, 676), radius=35 * S, fill=blue)
    draw.ellipse(box(278, 550, 362, 638), fill=cream, outline=brown, width=15 * S)
    draw.ellipse(box(568, 560, 646, 650), fill=cream, outline=brown, width=15 * S)
    draw.pieslice(box(423, 556, 500, 625), start=0, end=180, fill=yellow)
    draw.ellipse(box(396, 635, 428, 668), fill=yellow, outline=brown, width=8 * S)
    draw.ellipse(box(499, 635, 531, 668), fill=yellow, outline=brown, width=8 * S)

    # Main head: outline then face.
    head = [
        (224, 236), (278, 180), (358, 208), (458, 203), (552, 180), (665, 201),
        (708, 290), (705, 420), (687, 514), (638, 590), (542, 640), (430, 649),
        (315, 621), (242, 560), (205, 475), (193, 360),
    ]
    draw.polygon(sc(head), fill=brown)
    face = [
        (244, 248), (287, 203), (365, 232), (459, 226), (550, 205), (639, 225),
        (671, 298), (671, 430), (650, 513), (603, 569), (526, 604), (429, 612),
        (330, 589), (264, 532), (234, 456), (224, 360),
    ]
    draw.polygon(sc(face), fill=cream)

    # Ears.
    draw.polygon(sc([(231, 239), (285, 126), (335, 229)]), fill=brown)
    draw.polygon(sc([(255, 228), (288, 156), (318, 226)]), fill=cream)
    draw.polygon(sc([(554, 205), (655, 148), (679, 270)]), fill=brown)
    draw.polygon(sc([(575, 213), (641, 174), (657, 252)]), fill=cream)

    # Autumn leaf over the left side.
    leaf_outline = [
        (132, 350), (204, 357), (229, 289), (282, 351), (345, 302), (338, 388),
        (420, 406), (343, 448), (362, 527), (285, 489), (237, 560), (214, 479),
        (136, 498), (179, 430),
    ]
    draw.polygon(sc(leaf_outline), fill=brown)
    leaf = [
        (158, 365), (211, 373), (232, 315), (280, 370), (327, 335), (321, 400),
        (383, 414), (323, 440), (337, 498), (284, 468), (244, 526), (226, 462),
        (168, 476), (203, 427),
    ]
    draw.polygon(sc(leaf), fill=orange)
    draw.polygon(sc([(202, 378), (257, 354), (245, 442)]), fill=orange2)
    draw.polygon(sc([(248, 360), (322, 399), (250, 424)]), fill=(255, 187, 66))
    draw.line(sc([(238, 505), (257, 439), (268, 352)]), fill=yellow, width=9 * S)
    for pts in [
        [(258, 420), (205, 395)],
        [(261, 413), (316, 375)],
        [(257, 442), (203, 462)],
        [(264, 443), (321, 472)],
    ]:
        draw.line(sc(pts), fill=yellow, width=8 * S)

    # Leaf eye.
    draw.ellipse(box(276, 398, 363, 470), fill=cream, outline=(194, 72, 15), width=10 * S)
    draw.ellipse(box(316, 407, 354, 461), fill=(0, 0, 0))

    # Bow.
    draw.ellipse(box(468, 180, 592, 312), fill=brown)
    draw.ellipse(box(493, 200, 580, 296), fill=red)
    draw.ellipse(box(632, 248, 759, 377), fill=brown)
    draw.ellipse(box(648, 268, 735, 354), fill=red)
    draw.ellipse(box(562, 248, 657, 338), fill=brown)
    draw.ellipse(box(588, 272, 633, 318), fill=red)
    draw.ellipse(box(635, 264, 708, 337), fill=red)

    # Face details.
    draw.ellipse(box(571, 436, 615, 504), fill=(0, 0, 0))
    draw.ellipse(box(435, 464, 486, 504), fill=brown)
    draw.ellipse(box(449, 471, 480, 498), fill=yellow)
    for pts in [
        [(665, 412), (751, 394)],
        [(666, 446), (748, 450)],
        [(661, 480), (735, 505)],
    ]:
        draw.line(sc(pts), fill=brown, width=10 * S)

    # Small body details drawn last.
    draw.ellipse(box(397, 643, 417, 662), fill=orange)
    draw.ellipse(box(505, 643, 525, 662), fill=orange)

    image = image.resize((900, 900), Image.Resampling.LANCZOS)
    path = out_dir / "user_cat_fox_reference_v2.png"
    image.save(path)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
