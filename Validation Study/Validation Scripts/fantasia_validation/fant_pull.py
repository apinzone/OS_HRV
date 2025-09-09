import requests
import os

def download_fantasia_validation_set():
    """Download just 5 Fantasia records for validation"""
    
    base_url = "https://physionet.org/files/fantasia/1.0.0/"
    download_dir = "fantasia_validation"
    
    if not os.path.exists(download_dir):
        os.makedirs(download_dir)
    
    # Just 5 records - sufficient for validation
    records = ['f1y01', 'f1y02', 'f1y03', 'f1y04', 'f1y05']
    
    for record in records:
        for ext in ['dat', 'hea', 'ecg']:
            filename = f"{record}.{ext}"
            url = base_url + filename
            local_path = os.path.join(download_dir, filename)
            
            print(f"Downloading {filename}...")
            response = requests.get(url)
            
            if response.status_code == 200:
                with open(local_path, 'wb') as f:
                    f.write(response.content)
                print(f"Downloaded {filename}")
            else:
                print(f"Failed to download {filename}")

download_fantasia_validation_set()