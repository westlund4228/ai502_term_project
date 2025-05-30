import urllib.request
from pathlib import Path

dropbox_url = "https://www.dropbox.com/scl/fi/q3dm2eqcprqxyxlbxis1n/resnet18_cifar10_seed2025.pth.tar?rlkey=69hxy7wo1g6jfnlub22a58xoi&st=t0hlajrx&dl=1"
local_path = Path("models/state_dict/resnet_cifar.pth.tar")
local_path.parent.mkdir(parents=True, exist_ok=True)

urllib.request.urlretrieve(dropbox_url, local_path)

print(f"Model saved to {local_path}")