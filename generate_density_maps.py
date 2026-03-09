import os, scipy.io as io, numpy as np, cv2
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

def generate_density_map(img, points, sigma=15):
    density = np.zeros(img.shape[:2], dtype=np.float32)
    for x, y in points:
        if int(y) < img.shape[0] and int(x) < img.shape[1]:
            density[int(y), int(x)] = 1
    return gaussian_filter(density, sigma=sigma)

def process_dataset(img_dir, gt_dir, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    for img_file in tqdm(os.listdir(img_dir)):
        if not img_file.endswith('.jpg'): continue
        img_path = os.path.join(img_dir, img_file)
        gt_path = os.path.join(gt_dir, 'GT_' + img_file[:-4] + '.mat')
        mat = io.loadmat(gt_path)
        points = mat["image_info"][0,0][0,0][0]
        img = cv2.imread(img_path)
        density_map = generate_density_map(img, points)
        np.save(os.path.join(save_dir, img_file.replace('.jpg', '.npy')), density_map)

# Example usage:
# process_dataset("dataset/part_A_final/train_data/images",
#                 "dataset/part_A_final/train_data/ground_truth",
#                 "dataset/part_A_final/train_data/density_maps")
