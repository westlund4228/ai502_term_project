import urllib.request
from pathlib import Path

dropbox_url = "https://www.dropbox.com/scl/fi/ihkv3g6p24cvxprg9t3t1/mobilenetv1_cifar10_seed2025_acc86.tar?rlkey=ddjj1zpxdo8pd5hibhm6tt1wh&st=df1efq4j&dl=1"
local_path = Path("models/state_dict/mobilenetv1_cifar.pth.tar")
local_path.parent.mkdir(parents=True, exist_ok=True)

urllib.request.urlretrieve(dropbox_url, local_path)

print(f"Model saved to {local_path}")