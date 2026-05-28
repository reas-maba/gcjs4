import os
import urllib.request
import zipfile
import tarfile
import shutil

def download_file(url, dest_path):
    print(f'Downloading {url}...')
    urllib.request.urlretrieve(url, dest_path)
    print(f'Downloaded to {dest_path}')

def extract_zip(file_path, dest_dir):
    print(f'Extracting {file_path}...')
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)
    print(f'Extracted to {dest_dir}')

def extract_tar(file_path, dest_dir):
    print(f'Extracting {file_path}...')
    with tarfile.open(file_path, 'r:gz') as tar_ref:
        tar_ref.extractall(dest_dir)
    print(f'Extracted to {dest_dir}')

def organize_dataset(base_dir):
    images_dir = os.path.join(base_dir, 'Images')
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    breed_folders = [f for f in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, f))]
    
    for breed_folder in breed_folders:
        breed_path = os.path.join(images_dir, breed_folder)
        images = [f for f in os.listdir(breed_path) if f.endswith('.jpg')]
        
        train_count = int(len(images) * 0.8)
        train_images = images[:train_count]
        val_images = images[train_count:]
        
        train_breed_dir = os.path.join(train_dir, breed_folder)
        val_breed_dir = os.path.join(val_dir, breed_folder)
        
        os.makedirs(train_breed_dir, exist_ok=True)
        os.makedirs(val_breed_dir, exist_ok=True)
        
        for img in train_images:
            src = os.path.join(breed_path, img)
            dst = os.path.join(train_breed_dir, img)
            shutil.copy(src, dst)
        
        for img in val_images:
            src = os.path.join(breed_path, img)
            dst = os.path.join(val_breed_dir, img)
            shutil.copy(src, dst)
        
        print(f'Organized {breed_folder}: {len(train_images)} train, {len(val_images)} val')

def main():
    data_dir = '../data'
    os.makedirs(data_dir, exist_ok=True)
    
    try:
        images_url = 'http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar'
        images_file = os.path.join(data_dir, 'images.tar')
        
        download_file(images_url, images_file)
        extract_tar(images_file, data_dir)
        
        organize_dataset(data_dir)
        
        print('Dataset download and organization completed successfully!')
        
    except Exception as e:
        print(f'Error: {e}')
        print('Please download the dataset manually from:')
        print('http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar')
        print('And extract it to the data directory')

if __name__ == '__main__':
    main()