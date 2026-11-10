import os
import modal

app = modal.App("bodym-download")
vol = modal.Volume.from_name("bodym", create_if_missing=True)
image = modal.Image.debian_slim().pip_install("boto3")


@app.function(image=image, volumes={"/data": vol}, timeout=3600)
def download():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from concurrent.futures import ThreadPoolExecutor

    bucket = "amazon-bodym"
    s3 = boto3.client("s3", region_name="us-west-2",
                      config=Config(signature_version=UNSIGNED, max_pool_connections=64))

    objs = [o for p in s3.get_paginator("list_objects_v2").paginate(Bucket=bucket)
            for o in p.get("Contents", []) if not o["Key"].endswith("/")]

    def fetch(o):
        dst = os.path.join("/data", o["Key"])
        if os.path.exists(dst) and os.path.getsize(dst) == o["Size"]:
            return 0
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        s3.download_file(bucket, o["Key"], dst)
        return 1

    with ThreadPoolExecutor(32) as ex:
        n = sum(ex.map(fetch, objs))
    vol.commit()
    print(f"{len(objs)} files in bucket, downloaded {n}")


@app.local_entrypoint()
def main():
    download.remote()