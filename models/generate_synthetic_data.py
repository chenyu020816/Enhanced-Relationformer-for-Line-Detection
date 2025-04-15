import argparse
import cv2
import os
import pyvista as pv
from tqdm import tqdm 

from augmentations import *

def aug_pipeline(data):
    data = hori_flip(data, p=0.5)
    data = vert_flip(data, p=0.5)
    data = random_hide(data, max_hide_size=(30, 30), fix_hide_size=False, p=0.3)
    data = random_add_point(data, p=1, max_add_point_num=10, min_points_dist=10)
    data = jpeg_compress(data, p=0.3)
    data = gaussian_blur(data, p=0.3)
    return data


def transform_img(line_data, transform):
    new_line_data = transform(line_data)
    image_data = new_line_data.image
    poly_data = new_line_data.graph.to_polydata()

    return image_data, poly_data


def get_linedata(data):
    image_data = cv2.imread(data['img'])
    poly_data = pv.read(data['vtp'])
    h, w, _ = image_data.shape
    graph = Graph(poly_data, h, w)
    line_data = LineData(image_data, graph)
    return line_data


def main(args):
    data_folder = args.data_folder
    save_folder = args.save_folder
    aug_size = args.aug_size
    mixup = args.mixup

    img_folder = os.path.join(data_folder, 'raw')
    seg_folder = os.path.join(data_folder, 'seg')
    vtk_folder = os.path.join(data_folder, 'vtp')
    file_names = []
    img_files = []
    vtk_files = []
    seg_files = []

    for file_ in os.listdir(img_folder):
        file_ = file_[:-8]
        file_names.append(file_)
        img_files.append(os.path.join(img_folder, file_+'data.png'))
        vtk_files.append(os.path.join(vtk_folder, file_+'graph.vtp'))
        seg_files.append(os.path.join(seg_folder, file_+'seg.png'))

    data_dicts = [
        {"file_name": file_name, "img": img_file, "vtp": vtk_file, "seg": seg_file} 
        for file_name, img_file, vtk_file, seg_file in zip(file_names, img_files, vtk_files, seg_files)
    ]

    transform = ComposeLineData([
        lambda x: aug_pipeline(x)
    ])

    output_img_folder = os.path.join(save_folder, 'raw')
    output_vtk_folder = os.path.join(save_folder, 'vtp')
    output_seg_folder = os.path.join(save_folder, 'seg')
    os.makedirs(output_img_folder, exist_ok=True)
    os.makedirs(output_vtk_folder, exist_ok=True)
    os.makedirs(output_seg_folder, exist_ok=True)

    for idx in tqdm(range(len(data_dicts[:50]))):
        for i in range(aug_size - mixup):
            data = data_dicts[idx]
            file_name = data['file_name']
            line_data = get_linedata(data)
            seg_data = cv2.imread(data['seg'])

            if mixup > 0: # and random.random() > 0.7:
                indices = random.sample(range(len(data_dicts)), 3)
                datas = []
                datas.append(line_data)
                for idx in indices: 
                    data = data_dicts[idx]
                    line_data = get_linedata(data)
                    datas.append(line_data)
                new_line_data = mosaic(datas)
                image_data = new_line_data.image
                poly_data = new_line_data.graph.to_polydata()
            else:
                image_data, poly_data = transform_img(line_data, transform)

            image_path = os.path.join(output_img_folder, f'{file_name}{i}_data.png')
            vtk_path = os.path.join(output_vtk_folder, f'{file_name}{i}_graph.vtp')
            seg_path = os.path.join(output_seg_folder, f'{file_name}{i}_seg.png')

            cv2.imwrite(image_path, image_data)
            poly_data.save(vtk_path)
            cv2.imwrite(seg_path, seg_data)

    folder = save_folder
    os.makedirs(os.path.join(folder, "viz"), exist_ok=True)
    img_size = image_data.shape
    random.seed(42)
    
    raw_images = os.listdir(os.path.join(folder, "raw"))
    raw_images = sorted(raw_images)
    selected_images = raw_images
    
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

        image_path = os.path.join(folder, "viz", img_name)
        cv2.imwrite(image_path, combined_image)
    return 


if __name__ ==  "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder')
    parser.add_argument('--save_folder')
    parser.add_argument('--aug_size', type=int, required=True)
    parser.add_argument('--mixup', type=int, default=0)
    args = parser.parse_args()
    assert args.aug_size >= args.mixup, "Error"
    main(args)
