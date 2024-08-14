import os
import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import time

MAX_ATTEMPTS=5

def download_images(search_term: str, num_images: int = 5, sleep_time: float = 1.0, target_path: str = "downloaded_images"):
    url = f"https://www.google.com/search?hl=en&tbm=isch&q={search_term}"

    headers = {"User-Agent": "Mozilla/5.0"}

    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    img_tags = soup.find_all('img')
    img_urls = [img['src'] for img in img_tags if 'src' in img.attrs]

    os.makedirs(target_path, exist_ok=True)

    for i in range(num_images):
        img_url = img_urls[i] if i < len(img_urls) else None
        if img_url is None:
            print("Not enough images found.")
            break

        success = False
        attempt = 0

        while not success and (attempt < MAX_ATTEMPTS):
            try:
                img_response = requests.get(img_url)
                img_response.raise_for_status()  # Raise an error for bad responses
                img = Image.open(BytesIO(img_response.content))
                img.save(f'{target_path}/image_{i+1}.png')
                print(f'Downloaded image_{i+1}.png')
                success = True

            except Exception as e:
                attempt += 1
                print(f'Attempt {attempt}: Could not download image {i+1} from {img_url}: {e}')
                img_urls.remove(img_url)  # Remove the invalid URL from the list
                if i < len(img_urls):  # If there's another image, pick the next one
                    img_url = img_urls[i]
                else:
                    print("No more images to attempt.")
                    break

            time.sleep(sleep_time)

def main():
    search_term = input("Enter the search term: ")
    num_images = int(input("Enter the number of images to download (default is 5 number): ") or 5)
    sleep_time = float(input("Enter the sleep time between downloads (default is 1.0 second): ") or 1.0)
    target_path = input("Enter the target directory for downloaded images (default is 'downloaded_images'): ") or "downloaded_images"

    download_images(search_term, num_images, sleep_time, target_path)

if __name__ == "__main__":
    main()
