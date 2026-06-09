from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass, asdict
from html import unescape
from html.parser import HTMLParser
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen


USER_AGENT = (
    "Mozilla/5.0 (compatible; EmbroideryResearchDatasetBot/0.1; "
    "+local-research; respectful-rate-limit)"
)


class LinkImageParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.links: list[tuple[str, str]] = []
        self.images: list[str] = []
        self.meta: dict[str, str] = {}
        self._current_href: str | None = None
        self._current_text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr = {k.lower(): v or "" for k, v in attrs}
        if tag.lower() == "a":
            self._current_href = attr.get("href")
            self._current_text = []
        elif tag.lower() == "img":
            src = attr.get("src")
            if src:
                self.images.append(src)
        elif tag.lower() == "meta":
            key = attr.get("property") or attr.get("name")
            value = attr.get("content")
            if key and value:
                self.meta[key.lower()] = value

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() == "a" and self._current_href:
            text = " ".join("".join(self._current_text).split())
            self.links.append((self._current_href, text))
            self._current_href = None
            self._current_text = []

    def handle_data(self, data: str) -> None:
        if self._current_href:
            self._current_text.append(data)


@dataclass
class DesignRecord:
    id: str
    url: str
    title: str
    size_text: str
    colors: str
    stitches: str
    preview_url: str
    preview_local: str
    dst_url: str
    pes_url: str
    color_chart_url: str
    file_links_json: str
    status: str


def safe_name(text: str, max_len: int = 80) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text.strip()).strip("_")
    return text[:max_len] or "item"


def request_url(url: str, cookie_header: str = "", timeout: int = 25) -> tuple[int, bytes, str]:
    headers = {"User-Agent": USER_AGENT, "Accept": "text/html,application/xhtml+xml,*/*"}
    if cookie_header:
        headers["Cookie"] = cookie_header
    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read(), resp.headers.get_content_type()
    except HTTPError as exc:
        return exc.code, exc.read(), exc.headers.get_content_type()
    except URLError as exc:
        return 0, str(exc).encode("utf-8", "ignore"), "error/urllib"


def fetch_text(url: str, cookie_header: str = "") -> tuple[int, str]:
    status, body, _ = request_url(url, cookie_header=cookie_header)
    for enc in ("utf-8", "latin-1"):
        try:
            return status, body.decode(enc)
        except UnicodeDecodeError:
            continue
    return status, body.decode("utf-8", "ignore")


def parse_design_page(url: str, html: str) -> tuple[DesignRecord, list[str]]:
    parser = LinkImageParser()
    parser.feed(html)
    text = unescape(re.sub(r"<[^>]+>", " ", html))
    text = " ".join(text.split())

    title = parser.meta.get("og:title", "")
    if not title:
        m = re.search(r"<h1[^>]*>(.*?)</h1>", html, flags=re.I | re.S)
        title = unescape(re.sub(r"<[^>]+>", " ", m.group(1))).strip() if m else ""
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.I | re.S)
        title = unescape(m.group(1)).strip() if m else Path(urlparse(url).path).stem

    size_match = re.search(r"Size:\s*([^C]+?)\s+Colors[;:]", text, flags=re.I)
    size_text = size_match.group(1).strip() if size_match else ""
    colors_match = re.search(r"Colors[;:]\s*(\d+)", text, flags=re.I)
    stitches_match = re.search(r"Stitches:\s*(\d+)", text, flags=re.I)

    file_links: dict[str, list[dict[str, str]]] = {}
    design_links: list[str] = []
    for href, label in parser.links:
        full = urljoin(url, href)
        clean_label = " ".join(label.split())
        lower = clean_label.lower()
        if "/free-embroidery-designs/individual/" in urlparse(full).path:
            design_links.append(full)
        fmt_match = re.search(r":\s*([a-z0-9+]+)\b", lower)
        if fmt_match:
            fmt = fmt_match.group(1)
            file_links.setdefault(fmt, []).append({"label": clean_label, "url": full})

    image_candidates: list[str] = []
    gallery_match = re.search(
        r'<a[^>]+href=["\']([^"\']+\.(?:jpg|jpeg|png|webp))["\'][^>]*(?:class=["\'][^"\']*gal_img|data-gallery=)',
        html,
        flags=re.I,
    )
    if gallery_match:
        image_candidates.append(gallery_match.group(1))
    if parser.meta.get("og:image"):
        image_candidates.append(parser.meta["og:image"])
    image_candidates.extend(parser.images)
    preview_url = ""
    for img in image_candidates:
        full = urljoin(url, img)
        low = full.lower()
        if any(ext in low for ext in (".jpg", ".jpeg", ".png", ".webp")):
            preview_url = full
            if "embroidery_design" in low or "files/" in low:
                break

    uid = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
    record = DesignRecord(
        id=uid,
        url=url,
        title=title,
        size_text=size_text,
        colors=colors_match.group(1) if colors_match else "",
        stitches=stitches_match.group(1) if stitches_match else "",
        preview_url=preview_url,
        preview_local="",
        dst_url=(file_links.get("dst") or [{}])[0].get("url", ""),
        pes_url=(file_links.get("pes") or [{}])[0].get("url", ""),
        color_chart_url=(file_links.get("jpg") or [{}])[0].get("url", ""),
        file_links_json=json.dumps(file_links, ensure_ascii=False),
        status="parsed",
    )
    return record, sorted(set(design_links))


def download_public_asset(url: str, output_path: Path, cookie_header: str = "") -> bool:
    if not url:
        return False
    status, body, content_type = request_url(url, cookie_header=cookie_header)
    if status != 200 or not body:
        return False
    if "html" in content_type.lower() and len(body) < 200_000:
        return False
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(body)
    return True


def iter_seed_urls(seed_urls: Iterable[str], seed_file: str | None) -> list[str]:
    urls = [u.strip() for u in seed_urls if u.strip()]
    if seed_file:
        urls.extend(
            line.strip()
            for line in Path(seed_file).read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return list(dict.fromkeys(urls))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Respectful Embroideres dataset collector: pages, preview images, and download manifests."
    )
    ap.add_argument("--seed-url", action="append", default=[], help="Design or category URL. Repeatable.")
    ap.add_argument("--seed-file", default="", help="Text file containing one URL per line.")
    ap.add_argument("--output-dir", default="datasets/embroideres_free_raw_20260513")
    ap.add_argument("--max-pages", type=int, default=20)
    ap.add_argument("--sleep", type=float, default=2.5, help="Seconds between requests.")
    ap.add_argument("--crawl-same-domain", action="store_true", help="Also follow design links found on pages.")
    ap.add_argument("--cookie-header", default="", help="Optional normal logged-in browser Cookie header.")
    ap.add_argument("--download-public-images", action="store_true", default=True)
    args = ap.parse_args()

    out = Path(args.output_dir)
    pages_dir = out / "pages"
    images_dir = out / "images"
    pending_dir = out / "download_links_pending"
    pages_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    pending_dir.mkdir(parents=True, exist_ok=True)

    queue = iter_seed_urls(args.seed_url, args.seed_file or None)
    seen: set[str] = set()
    records: list[DesignRecord] = []

    while queue and len(seen) < args.max_pages:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        status, html = fetch_text(url, cookie_header=args.cookie_header)
        uid = hashlib.sha1(url.encode("utf-8")).hexdigest()[:12]
        (pages_dir / f"{uid}.html").write_text(html, encoding="utf-8")
        if status != 200:
            records.append(
                DesignRecord(uid, url, "", "", "", "", "", "", "", "", "", "{}", f"http_{status}")
            )
            time.sleep(args.sleep)
            continue

        parser_for_links = LinkImageParser()
        parser_for_links.feed(html)
        found_links = sorted(
            {
                urljoin(url, href)
                for href, _ in parser_for_links.links
                if "/free-embroidery-designs/individual/" in urlparse(urljoin(url, href)).path
            }
        )
        is_design_page = "/free-embroidery-designs/individual/" in urlparse(url).path
        if not is_design_page:
            if args.crawl_same_domain:
                domain = urlparse(url).netloc
                for link in found_links:
                    if urlparse(link).netloc == domain and link not in seen and link not in queue:
                        queue.append(link)
            time.sleep(args.sleep)
            continue

        record, found_links = parse_design_page(url, html)
        if record.preview_url and args.download_public_images:
            ext = Path(urlparse(record.preview_url).path).suffix or ".jpg"
            image_path = images_dir / f"{safe_name(record.title)}_{record.id}{ext}"
            if download_public_asset(record.preview_url, image_path, cookie_header=args.cookie_header):
                record.preview_local = str(image_path)

        pending = {
            "id": record.id,
            "url": record.url,
            "title": record.title,
            "note": "Use normal login/manual download if the site requires authentication. Do not bypass access controls.",
            "files": json.loads(record.file_links_json or "{}"),
        }
        (pending_dir / f"{record.id}.json").write_text(
            json.dumps(pending, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        records.append(record)

        if args.crawl_same_domain:
            domain = urlparse(url).netloc
            for link in found_links:
                if urlparse(link).netloc == domain and link not in seen and link not in queue:
                    queue.append(link)
        time.sleep(args.sleep)

    manifest = out / "manifest.csv"
    with manifest.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()) if records else list(DesignRecord.__annotations__.keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))

    summary = {
        "output_dir": str(out),
        "pages_seen": len(seen),
        "records": len(records),
        "images_downloaded": sum(1 for r in records if r.preview_local),
        "dst_links_found": sum(1 for r in records if r.dst_url),
        "pes_links_found": sum(1 for r in records if r.pes_url),
        "manifest": str(manifest),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
