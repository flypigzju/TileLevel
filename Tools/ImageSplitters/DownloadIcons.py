import io
import os
import zipfile
import requests

# 输出目录（会自动创建）
OUT_DIR = r"D:\pixelarticons_svg"

# GitHub 仓库 zip（master 分支）
ZIP_URL = "https://github.com/halfmage/pixelarticons/archive/refs/heads/master.zip"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("[1/3] downloading zip...")
    resp = requests.get(ZIP_URL, timeout=60)
    resp.raise_for_status()

    print("[2/3] extracting svg/ ...")
    zf = zipfile.ZipFile(io.BytesIO(resp.content))

    # zip 内部路径类似：pixelarticons-master/svg/xxx.svg
    extracted = 0
    for name in zf.namelist():
        if not name.endswith(".svg"):
            continue
        # 只要 svg 目录
        if "/svg/" not in name.replace("\\", "/"):
            continue

        base = os.path.basename(name)
        if not base:
            continue

        out_path = os.path.join(OUT_DIR, base)
        with zf.open(name) as src, open(out_path, "wb") as dst:
            dst.write(src.read())
        extracted += 1

    print(f"[3/3] done! extracted {extracted} svgs -> {OUT_DIR}")

if __name__ == "__main__":
    main()
