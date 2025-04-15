import cv2
import os
import random
import numpy as np
import pyvista as pv


if __name__ == "__main__":
    angle_prune = True
    # folder = "./data/railroad/train_relationformer" if angle_prune else "./data/railroad/train_relationformer_noangle" 
    # img_size = [128, 128, 3]
    # folder = "./data/train_data_g256_comb_less_neg_topo"
    # img_size = [256, 256, 3]
     
    folder = "./data/test_aug"
    os.makedirs(os.path.join(folder, "viz"), exist_ok=True)
    img_size = [256, 256, 3]
    random.seed(42)
    
    raw_images = os.listdir(os.path.join(folder, "raw"))
    raw_images = sorted(raw_images)
    img_ind = [random.randint(1, len(raw_images) - 1) for _ in range(42)]
    selected_images = [raw_images[i] for i in img_ind]
    
    for img_name in selected_images:
        original_image = cv2.imread(os.path.join(folder, "raw", img_name))
        image = original_image.copy()
        vtp_name = img_name[:-8] + "graph.vtp"
        mesh = pv.read(os.path.join(folder, "vtp", vtp_name))

        points = []
        for p in mesh.points:
            x, y, _ = p
            x *= img_size[0]
            y *= img_size[1]
            
            points.append((int(y), int(x)))
        
        connect_num = int(mesh.lines.shape[0] / 3)
        
        drawed_lines = []
        for conn_i in range(connect_num): 
            start_idx = mesh.lines[conn_i * 3 + 1]
            end_idx = mesh.lines[conn_i * 3 + 2]

            if (end_idx, start_idx) in drawed_lines:
                continue
                
            start_point = points[start_idx]
            end_point = points[end_idx]

            cv2.line(image, start_point, end_point, color=(0, 255, 0))
            cv2.circle(image, start_point, radius=2, color=(0, 0, 255))
            cv2.circle(image, end_point, radius=2, color=(0, 0, 255))
            
            drawed_lines.append((start_idx, end_idx))

        separator = np.ones((image.shape[0], 20, 3), dtype=np.uint8) * 255
        combined_image = np.hstack((original_image, separator, image))

        type_folder = "prune" if angle_prune else "noprune"
        image_path = os.path.join(folder, "viz", img_name)
        cv2.imwrite(image_path, combined_image)
        print(f"Draw {img_name}")


