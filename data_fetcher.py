import os
import time
import requests
import pandas as pd
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")
if API_KEY is None:
    raise ValueError("GOOGLE_MAPS_API_KEY not set")

BASE_URL = "https://maps.googleapis.com/maps/api/staticmap"

IMAGE_SIZE = "224x224"
ZOOM = 18
MAP_TYPE = "satellite"
SLEEP_TIME = 0.1  # rate limiting

# -----------------------------
# IMAGE DOWNLOAD FUNCTION
# -----------------------------
def download_image(lat, lon, save_path):
    params = {
        "center": f"{lat},{lon}",
        "zoom": ZOOM,
        "size": IMAGE_SIZE,
        "maptype": MAP_TYPE,
        "key": API_KEY
    }

    response = requests.get(BASE_URL, params=params, timeout=10)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
        return True
    else:
        return False

# -----------------------------
# MAIN PIPELINE
# -----------------------------
def fetch_images(csv_path, image_dir):
    df = pd.read_csv(csv_path)

    os.makedirs(image_dir, exist_ok=True)

    failed = []

    for _, row in tqdm(df.iterrows(), total=len(df)):
        img_id = row["id"]
        lat = row["lat"]
        lon = row["long"]

        img_path = os.path.join(image_dir, f"{img_id}.png")

        # Skip if already downloaded
        if os.path.exists(img_path):
            continue

        success = download_image(lat, lon, img_path)

        if not success:
            failed.append(img_id)

        time.sleep(SLEEP_TIME)

    print(f"Completed. Failed downloads: {len(failed)}")
    return failed


if __name__ == "__main__":
    print("Downloading TRAIN images...")
    fetch_images("data/train.csv", "data/images/train")

    print("Downloading TEST images...")
    fetch_images("data/test.csv", "data/images/test")
