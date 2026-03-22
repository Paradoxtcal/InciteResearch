#!/usr/bin/env python3
"""
Batch PDF compression utility.

Use case: compress many PDFs and pack them into a ZIP for downstream analysis.

Examples:
    python compress_pdfs.py
    python compress_pdfs.py --input ./papers --output papers_compressed.zip
    python compress_pdfs.py --level 2 --max-size 400

Dependencies:
    pip install pikepdf Pillow tqdm
"""

import argparse
import os
import sys
import zipfile
from pathlib import Path


def check_dependencies():
    missing = []
    for pkg in ["pikepdf", "PIL", "tqdm"]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        pkgs = ["Pillow" if p == "PIL" else p for p in missing]
        print(f"Missing dependencies. Run: pip install {' '.join(pkgs)}")
        sys.exit(1)


def compress_single_pdf(
    input_path: Path,
    output_path: Path,
    level: int = 2,
    image_dpi: int = 150,
) -> tuple[float, float]:
    """
    Compress a single PDF.

    Returns (original size MB, compressed size MB).
    """
    import pikepdf
    from PIL import Image
    import io

    orig_size = input_path.stat().st_size / 1024 / 1024

    try:
        pdf = pikepdf.open(input_path)
        
        if level <= 2:
            for page in pdf.pages:
                for key, obj in page.get("/Resources", {}).get("/XObject", {}).items():
                    try:
                        if obj.get("/Subtype") == "/Image":
                            raw = obj.read_raw_bytes()
                            img = Image.open(io.BytesIO(raw))
                            
                            target_dpi = 72 if level == 1 else image_dpi
                            w, h = img.size
                            scale = min(1.0, target_dpi / max(img.info.get("dpi", (150, 150))))
                            if scale < 1.0:
                                new_w = int(w * scale)
                                new_h = int(h * scale)
                                img = img.resize((new_w, new_h), Image.LANCZOS)
                            
                            if level == 1 and img.mode not in ("L", "LA"):
                                img = img.convert("L")
                            
                            buf = io.BytesIO()
                            img.save(buf, format="JPEG", quality=70 if level == 1 else 85, optimize=True)
                            obj.write(buf.getvalue(), filter=pikepdf.Name("/DCTDecode"))
                    except Exception:
                        pass

        with pdf.open_metadata() as meta:
            meta.clear()

        pdf.save(output_path, compress_streams=True, object_stream_mode=pikepdf.ObjectStreamMode.generate)
        compressed_size = output_path.stat().st_size / 1024 / 1024

        return orig_size, compressed_size

    except Exception as e:
        import shutil
        shutil.copy2(input_path, output_path)
        return orig_size, orig_size


def compress_and_zip(
    input_dir: str = ".",
    output_zip: str = "papers_compressed.zip",
    level: int = 2,
    max_size_mb: int = 400,
    recursive: bool = False,
) -> dict:
    """
    Compress all PDFs in a directory and pack into one or more ZIP files.
    """
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = lambda x, **kw: x

    input_path = Path(input_dir)
    pattern = "**/*.pdf" if recursive else "*.pdf"
    pdf_files = sorted(input_path.glob(pattern))

    if not pdf_files:
        print(f"No PDF files found in {input_dir}")
        return {}

    print(f"Found {len(pdf_files)} PDFs. Compressing (level={level})...")

    import tempfile
    tmpdir = Path(tempfile.mkdtemp())
    stats = {"total": len(pdf_files), "success": 0, "orig_mb": 0, "compressed_mb": 0}

    compressed_files = []
    for pdf in tqdm(pdf_files, desc="Compressing"):
        out = tmpdir / pdf.name
        orig, comp = compress_single_pdf(pdf, out, level=level)
        stats["orig_mb"] += orig
        stats["compressed_mb"] += comp
        stats["success"] += 1
        compressed_files.append(out)

    zip_paths = []
    current_zip_size = 0
    zip_index = 1
    base_name = Path(output_zip).stem
    ext = ".zip"

    def new_zip_path():
        if zip_index == 1 and len(compressed_files) <= 50:
            return output_zip
        return f"{base_name}_part{zip_index}{ext}"

    current_zip_path = new_zip_path()
    current_zip = zipfile.ZipFile(current_zip_path, "w", zipfile.ZIP_DEFLATED)
    zip_paths.append(current_zip_path)

    for f in compressed_files:
        file_size_mb = f.stat().st_size / 1024 / 1024
        if current_zip_size + file_size_mb > max_size_mb:
            current_zip.close()
            zip_index += 1
            current_zip_path = new_zip_path()
            current_zip = zipfile.ZipFile(current_zip_path, "w", zipfile.ZIP_DEFLATED)
            zip_paths.append(current_zip_path)
            current_zip_size = 0
        current_zip.write(f, f.name)
        current_zip_size += file_size_mb

    current_zip.close()

    import shutil
    shutil.rmtree(tmpdir)

    ratio = (1 - stats["compressed_mb"] / max(stats["orig_mb"], 0.001)) * 100
    print("\n✅ Compression complete!")
    print(f"   Original:   {stats['orig_mb']:.1f} MB")
    print(f"   Compressed: {stats['compressed_mb']:.1f} MB (saved {ratio:.0f}%)")
    print(f"   Output:     {', '.join(zip_paths)}")
    print("\nUpload to ChatGPT with the prompt in prompts/.")

    stats["zip_paths"] = zip_paths
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compress PDFs and pack into ZIP files")
    parser.add_argument("--input", default=".", help="Input directory containing PDFs")
    parser.add_argument("--output", default="papers_compressed.zip", help="Output ZIP filename")
    parser.add_argument("--level", type=int, default=2, choices=[1, 2, 3], help="Compression level: 1=max 2=balanced 3=fast")
    parser.add_argument("--max-size", type=int, default=400, help="Max ZIP size in MB per file")
    parser.add_argument("--recursive", action="store_true", help="Search subdirectories recursively")
    args = parser.parse_args()

    check_dependencies()
    compress_and_zip(
        input_dir=args.input,
        output_zip=args.output,
        level=args.level,
        max_size_mb=args.max_size,
        recursive=args.recursive,
    )
