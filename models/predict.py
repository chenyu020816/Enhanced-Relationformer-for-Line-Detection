import argparse 
import cv2
import json
import torch
import os
import yaml 
import numpy as np

from models import build_model
from models.inference import relation_infer

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)

def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def main(args):
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.multiprocessing.set_sharing_strategy('file_system')
    device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")
    
    net = build_model(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    cnt = 0
    i = 0
    for image_path in os.listdir("./data/test_data/raw")[:500]:
        if not (image_path.endswith("png") or image_path.endswith("jpg")): continue
        image = cv2.imread(os.path.join("./data/test_data/raw/", image_path))
        image_c = image.copy()
        img = image.copy()
        image = torch.from_numpy(image)
        image = image.unsqueeze(0).float().to(device)
        image = image.permute(0, 3, 1, 2)
        with torch.no_grad():
            h, out, _, _, _, _ = net(image, seg=False)
        pred_nodes, pred_edges, _, pred_nodes_box, pred_nodes_box_score,\
        pred_nodes_box_class, pred_edges_box_score, pred_edges_box_class = relation_infer(
            h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN,
            nms=False, map_=True
        )

        if pred_edges[0].shape[0] == 0:
            print(f"{image_path} is empty")
            continue
        img_size = config.DATA.IMG_SIZE[0]
        for val in zip(pred_edges, pred_nodes):
            edges_, nodes_ = val
            nodes_ = nodes_.cpu().numpy()

            for i_idx, j_idx in edges_:                 
                n1, n2 = (nodes_[i_idx]*img_size).astype('int32'), (nodes_[j_idx]*img_size).astype('int32')
                
                cv2.line(img, (n1[0], n1[1]), (n2[0], n2[1]), (255,255,255), 1)
                cv2.circle(img, (n1[0], n1[1]), 2, (0,255,0), -1)
                cv2.circle(img, (n2[0], n2[1]), 2, (0,255,0), -1)
        
        save_path = './pred_map/pred/' + image_path
        print(save_path)
        cv2.imwrite(save_path, img)
        cv2.imwrite(f'./pred_map/gt/{image_path}', image_c)
        print('*** save the predicted map in {} ***'.format(save_path))
        i += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config',
                        default=None,
                        help='config file (.yml) containing the hyper-parameters for training. '
                            'If None, use the nnU-Net config. See /config for examples.')
    parser.add_argument('--checkpoint', default=None, help='checkpoint of the model to test.')
    parser.add_argument('--device', default='cuda',
                            help='device to use for training')
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0,1],
                            help='list of index where skip conn will be made.')
    parser.add_argument('--buffer', type=int, default=10,
                            help='the buffer size for nodes conflation')


    args = parser.parse_args()
    main(args)
