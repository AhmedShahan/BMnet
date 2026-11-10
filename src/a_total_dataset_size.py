import boto3
from botocore import UNSIGNED
from botocore.config import Config
from collections import defaultdict

B = "amazon-bodym"
s3 = boto3.client("s3", region_name="us-west-2", config=Config(signature_version=UNSIGNED))

size, count = defaultdict(int), defaultdict(int)
for p in s3.get_paginator("list_objects_v2").paginate(Bucket=B):
    for o in p.get("Contents", []):
        parts = o["Key"].split("/")
        k = "/".join(parts[:2]) if len(parts) > 2 else parts[0]   # যেমন train/mask
        size[k] += o["Size"]; count[k] += 1

for k in sorted(size):
    print(f"{k:30s} {count[k]:7d} files {size[k]/1e6:10.1f} MB")
print("TOTAL", round(sum(size.values())/1e9, 2), "GB")



'''
testA                                3 files        0.2 MB
testA/mask                        1684 files        9.5 MB
testA/mask_left                   1684 files        7.5 MB
testB                                3 files        0.2 MB
testB/mask                        1160 files        6.7 MB
testB/mask_left                   1160 files        5.2 MB
train                                3 files        1.2 MB
train/mask                        6134 files       34.5 MB
train/mask_left                   6134 files       26.5 MB
TOTAL 0.09 GB


'''