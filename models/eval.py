from argparse import ArgumentParser
from monai.data import DataLoader
import os
import pdb
import torch
from tqdm import tqdm
import yaml


from dataset_road_network import build_road_network_data
from inference import relation_infer
from train import dict2obj
from trainer import build_trainer
from models import build_model
from models.matcher import build_matcher
from losses import SetCriterion
from utils import image_graph_collate_road_network
from topo_metrics import *


def main(args):
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config['log']['message'])
    config = dict2obj(config)
    device = torch.device("cuda") if args.device=='cuda' else torch.device("cpu")
    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))
    net = build_model(config).to(device)
    seg_net = build_model(config).to(device)
    matcher = build_matcher(config)
    loss = SetCriterion(config, matcher, net)
    
    
    test_ds = build_road_network_data(
        config, mode='train'
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=True,
        num_workers=config.DATA.NUM_WORKERS,
        collate_fn=image_graph_collate_road_network,
        pin_memory=True
    )

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    net.load_state_dict(checkpoint['net'])
    net.eval()
    correctness_result = []
    completeness_result = []
    apls_result = []

    with torch.no_grad():
        print('Started processing test set.')
        for batchdata in test_loader: 

            # extract data and put to device
            images, segs, nodes, edges = batchdata[0], batchdata[1], batchdata[2], batchdata[3]
            images = images.to(args.device,  non_blocking=False)
            segs = segs.to(args.device,  non_blocking=False)
            nodes = [node.to(args.device,  non_blocking=False) for node in nodes]
            edges = [edge.to(args.device,  non_blocking=False) for edge in edges]
            
            h, out, _ = net(images, seg=False)
            pred_nodes, pred_edges, pred_nodes_box, pred_nodes_box_score, pred_nodes_box_class, pred_edges_box_score, pred_edges_box_class = relation_infer(
                h.detach(), out, net, config.MODEL.DECODER.OBJ_TOKEN, config.MODEL.DECODER.RLN_TOKEN,
                nms=False, map_=True
            )

            gt_nodes_np = nodes[0].cpu().numpy()
            gt_edges_np = edges[0].cpu().numpy()
            pred_nodes_np = pred_nodes[0].cpu().numpy()
            pred_edges_np = pred_edges[0]

            match_dict = match_nodes(pred_nodes_np, gt_nodes_np, tolerance=0.05)

            # 2. remap pred_edges
            mapped_pred_edges = remap_edges(pred_edges_np, match_dict)

            # 3. build graphs
            G_gt = build_graph(gt_nodes_np, gt_edges_np)
            G_pred = build_graph(gt_nodes_np, mapped_pred_edges)

            # 4. evaluate
            correctness_result.append(correctness(gt_edges_np, mapped_pred_edges))
            completeness_result.append(completeness(gt_edges_np, mapped_pred_edges))
            apls_result.append(compute_apls(G_gt, G_pred))
            # pdb.set_trace()
    """
    print(args.checkpoint)
    print(f"Correctness: {(sum(correctness_result) / len(correctness_result)):.5f}")
    print(f"Completeness: {(sum(completeness_result) / len(completeness_result)):.4f}")
    print(f"APLS: {(sum(apls_result) / len(apls_result)):.5f}")
    """
    def log_and_print(message):
        print(message)
        with open("eval.txt", "a") as f:
            f.write(message + "\n")

    log_and_print(f"Checkpoint: {args.checkpoint}")
    log_and_print(f"Correctness: {(sum(correctness_result) / len(correctness_result)):.5f}")
    log_and_print(f"Completeness: {(sum(completeness_result) / len(completeness_result)):.5f}")
    log_and_print(f"APLS: {(sum(apls_result) / len(apls_result)):.5f}")
    log_and_print("-" * 40)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config',
                        default=None,
                        help='config file (.yml) containing the hyper-parameters for training. '
                            'If None, use the nnU-Net config. See /config for examples.')
    parser.add_argument('--checkpoint', default=None, help='checkpoint of the last epoch of the model')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training')
    parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0,1],
                        help='list of index where skip conn will be made')
    args = parser.parse_args()

    main(args)
