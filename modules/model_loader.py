import os
import urllib.request
from urllib.parse import urlparse
from typing import Optional


def get_hf_token() -> Optional[str]:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token.strip()
    for token_path in [
        os.path.expanduser("~/.cache/huggingface/token"),
        os.path.expanduser("~/.huggingface/token")
    ]:
        if os.path.exists(token_path):
            try:
                with open(token_path, "r", encoding="utf-8") as f:
                    t = f.read().strip()
                    if t:
                        return t
            except Exception:
                pass
    return None


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """Download a file from `url` into `model_dir`, using the file present if possible.

    Returns the path to the downloaded file.
    """
    domain = os.environ.get("HF_MIRROR", "https://huggingface.co").rstrip('/')
    url = str.replace(url, "https://huggingface.co", domain, 1)
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        token = get_hf_token()
        if "huggingface.co" in url and token:
            req = urllib.request.Request(
                url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "Fooocus/1.0"
                }
            )
            try:
                from tqdm import tqdm
                with urllib.request.urlopen(req) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                    block_size = 1024 * 1024  # 1 MB chunks
                    with open(cached_file, "wb") as out_file, tqdm(
                        total=total_size,
                        unit='B',
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=file_name,
                        disable=not progress
                    ) as pbar:
                        while True:
                            buffer = response.read(block_size)
                            if not buffer:
                                break
                            out_file.write(buffer)
                            pbar.update(len(buffer))
            except Exception as e:
                if os.path.exists(cached_file):
                    os.remove(cached_file)
                raise e
        else:
            from torch.hub import download_url_to_file
            download_url_to_file(url, cached_file, progress=progress)
    return cached_file
