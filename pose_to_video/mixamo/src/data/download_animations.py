import requests
import time
import os
from typing import Optional



class Mixamo:
    def __init__(self, character_id: str, access_token: str):
        self.character_id = character_id
        self.access_token = access_token
        self.headers = {
            'Accept': 'application/json',
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_token}',
            'X-Api-Key': 'mixamo2'
        }
        self.download_path = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'animations-fbx')
        os.makedirs(self.download_path, exist_ok=True)

    def get_animation_list(self, page: int):
        print(f"Downloading page {page}")
        list_url = f'https://www.mixamo.com/api/v1/products?page={page}&limit=96&order=&type=Motion%2CMotionPack&query='
        response = requests.get(list_url, headers=self.headers)
        if response.status_code != 200:
            raise Exception('Failed to download animation list')
        return response.json()['results']

    def get_product_hash(self, anim_id: str, character: str):
        print(f"Reading product {anim_id}")
        product_url = f'https://www.mixamo.com/api/v1/products/{anim_id}?similar=0&character_id={character}'
        response = requests.get(product_url, headers=self.headers)
        if response.status_code != 200:
            raise Exception('Failed to download product details')
        return response.json()['details']['gms_hash']

    def download_animation(self, anim_id: str, product_name: str) -> Optional[str]:
        if ',' in product_name:
            # print(f'Skipping pack {product_name}')
            return None

        for i in range(5):
            try:
                gms_hash = self.get_product_hash(anim_id, self.character_id)
                pvals = ','.join(str(param[1]) for param in gms_hash['params'])
                gms_hash['params'] = pvals
                return self.export_animation([gms_hash], product_name)
            except Exception as e:
                print(e)
                print(f"Attempt {i + 1}: Failed to download animation '{product_name}'. Retrying...")
                time.sleep(1)
        raise Exception(f"Unable to download animation {anim_id}")

    def download_anims_in_page(self, page, tot_pages, character):
        if page >= tot_pages:
            print('All pages have been downloaded')
            return

        animation_list = self.get_animation_list(page)
        obj = {
            'anims': animation_list['results'],
            'currentPage': animation_list['pagination']['page'],
            'totPages': animation_list['pagination']['num_pages'],
            'character': character
        }
        self.download_anim_loop(**obj)

    def export_animation(self, gms_hash_array, product_name):
        print(f"Exporting animation {product_name}")
        export_url = 'https://www.mixamo.com/api/v1/animations/export'
        export_body = {
            'character_id': self.character_id,
            'gms_hash': gms_hash_array,
            'preferences': {'format': 'fbx7', 'skin': 'false', 'fps': '30', 'reducekf': '0'},
            'product_name': product_name,
            'type': 'Motion'
        }
        response = requests.post(export_url, headers=self.headers, json=export_body)
        if response.status_code // 100 != 2:
            raise Exception('Failed to export animation')
        self.monitor_animation(product_name)

    def monitor_animation(self, product_name):
        print("Monitoring animation")
        monitor_url = f'https://www.mixamo.com/api/v1/characters/{self.character_id}/monitor'
        while True:
            response = requests.get(monitor_url, headers=self.headers)
            if response.status_code == 404:
                raise Exception(f"ERROR: Monitor got 404 error: {response.error} message={response.message}")
            if response.status_code in [202, 200]:
                msg = response.json()
                print('...', msg['status'])
                if msg['status'] == 'completed':
                    download_url = msg['job_result']
                    self.download_file(download_url, product_name)
                    break
                elif msg['status'] == 'processing':
                    time.sleep(1)
                else:
                    if "UTF-8" in str(msg['job_result']): # ERROR occured on export: "\\xE2" from ASCII-8BIT to UTF-8
                        # print("Skipping download", msg['job_result'])
                        break
                    raise Exception(
                        f"ERROR: Monitor status: {msg['status']} message: {msg['message']} result: {msg['job_result']}")
            else:
                raise Exception('Response not handled')

    def download_file(self, url, filename):
        print(f"Downloading file {filename}")
        local_filename = self.valid_file_name(filename)
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_filename, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        print(f"Downloaded file: {local_filename}")
        print("----\n\n")


    def valid_file_name(self, value) -> str:
        value = str(value)
        value = value.replace('/', '_')
        return os.path.join(self.download_path, f"{value}.fbx")

    def start(self):
        i = 1
        while True:
            animations_list = self.get_animation_list(i)
            if len(animations_list) == 0:
                break

            for animation in animations_list:
                if os.path.exists(self.valid_file_name(animation['description'])):
                    print(f"Skipping animation {animation['id']}")
                else:
                    print(f"Downloading animation {animation['id']}")
                    self.download_animation(animation['id'], animation['description'])

            i += 1


if __name__ == '__main__':
    character_id = '037852b5-74da-44aa-878b-eccda13e5139'
    # Read access token from file
    with open(os.path.join(os.path.dirname(__file__), '..', '..', 'access_token.txt'), 'r') as f:
        access_token = f.read()

    mixamo = Mixamo(character_id, access_token)
    mixamo.start()
