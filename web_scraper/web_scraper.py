import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import time
import sys

MAX_ATTEMPTS = 5
EACH_PAGE_IMAGE_CNT=20

def download_images(search_term: str, num_images: int = 5, sleep_time: float = 1.0, target_path: str = "downloaded_images"):
    base_url = "https://www.google.com/search?hl=en&tbm=isch&q={}&ijn={}"

    headers = {"User-Agent": "Mozilla/5.0"}
    img_urls = []

    for page in range((num_images // EACH_PAGE_IMAGE_CNT) + 1):
        url = base_url.format(search_term, page)
        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.text, "html.parser")
        img_tags = soup.find_all('img')

        img_urls += [img['src'] for img in img_tags if 'src' in img.attrs]

        if len(img_urls) >= num_images:
            break

        time.sleep(sleep_time)

    os.makedirs(target_path, exist_ok=True)


    for i in range(min(num_images, len(img_urls))):
        img_url = img_urls[i]
        success = False
        attempt = 0

        while not success and (attempt < MAX_ATTEMPTS):
            try:
                img_response = requests.get(img_url)
                img_response.raise_for_status()
                img = Image.open(BytesIO(img_response.content))
                img.save(f'{target_path}/image_{i + 1}.png')
                print(f'Downloaded image_{i + 1}.png')
                success = True

            except Exception as e:
                attempt += 1
                print(f'Attempt {attempt}: Could not download image {i + 1} from {img_url}: {e}')
                img_urls.remove(img_url)
                if i < len(img_urls):
                    img_url = img_urls[i]
                else:
                    print("No more images to attempt.")
                    break

            time.sleep(sleep_time)

def main():
    search_term = sys.argv[1] if len(sys.argv) > 1 \
        else input("Enter the search term: ")
    num_images = int(sys.argv[2]) if len(sys.argv) > 2 \
        else int(input("Enter the number of images to download (default is 5): ") or 5)
    sleep_time = float(sys.argv[3]) if len(sys.argv) > 3 \
        else float(input("Enter the sleep time between downloads (default is 1.0 second): ") or 1.0)
    target_path = sys.argv[4] if len(sys.argv) > 4 \
        else input("Enter the target directory for downloaded images \
                   (default is 'downloaded_images'): ") or "downloaded_images"

    download_images(search_term, num_images, sleep_time, target_path)

if __name__ == "__main__":
    main()
