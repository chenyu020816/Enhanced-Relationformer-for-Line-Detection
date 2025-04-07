import albumentations as A
import numpy as np
import torchvision.transforms.functional as tvf
import random


class Graph:
    def __init__(self, polydata, h, w):
        self.polydata = polydata
        self.h = h
        self.w = w
        self.points = self._get_points_list()
        self.lines = self._get_lines_list()
        self.conns = self._get_conn_dict()

    def update_points_list(self, new_points_list):
        self.points = new_points_list
        self.conns = self._get_conn_dict()
        return 
        
    def update_lines_list(self, new_lines_list):
        self.lines = new_lines_list
        self.conns = self._get_conn_dict()
        return 
        
    def update_graph(self, new_points_list: None, new_lines_list: None):
        if new_points_list is not None:
            self.update_points_list(new_points_list)
        if new_lines_list is not None:
            self.update_lines_list(new_lines_list)
        return 
        
    def _get_points_list(self):
        points = self.polydata.points.copy()
        points[:, 0] *= self.h
        points[:, 1] *= self.w
        points_list = [(int(point[1]), int(point[0])) for point in points]
        return points_list

    def _get_lines_list(self):
        lines_list = [[self.polydata.lines[i], self.polydata.lines[i+1]]  for i in range(1, len(self.polydata.lines), 3)]
        return lines_list

    def _get_conn_dict(self):
        conns = dict()
        for line in self.lines:
            p1, p2 = line
            if p1 not in conns.keys():
                conns[p1] = []
            if p2 not in conns.keys():
                conns[p2] = []
            if p2 in conns[p1]: 
                continue
            else: 
                conns[p1].append(p2)
            if p1 in conns[p2]: 
                continue
            else: 
                conns[p2].append(p1)
            
        return conns

    def to_polydata(self, new_h = None, new_w = None):
        if new_h is not None:
            self.h = new_h
        
        if new_w is not None:
            self.w = new_w
        points = []
        for coord in self.points: 
            y = coord[1] / self.h
            x = coord[0] / self.w
            point = [y, x, 0]
            points.append(point)
        vtp_points = np.array(points)
        conns = []
        for conn in self.lines:
            conns.append(2)
            for p in conn:
                conns.append(p)
        vtp_lines = np.array(conns)
        if len(vtp_points) > 0:
            self.polydata.points = vtp_points
            self.polydata.lines = vtp_lines
        else:
            self.polydata.points = np.empty((0, 3))
            self.polydata.lines = None
        
        return self.polydata


    def copy(self):
        new_polydata = self.to_polydata()
        return Graph(self.to_polydata(), self.h, self.w)        
    

class LineData:
    def __init__(self, image, graph):
        self.image = image
        self.graph = graph

    def update_image(self, new_image):
        self.image = new_image

    def update_graph(self, new_points_list: None, new_lines_list: None):
        self.graph.update_graph(new_points_list, new_lines_list)
        return 
    

def random_horizontal_flip(image, seg, prob=0.5):
    if random.random() < prob:
        image = tvf.hflip(image)
        seg = tvf.hflip(seg)
    return image, seg


def random_vertical_flip(image, seg, prob=0.5):
    if random.random() < prob:
        image = tvf.vflip(image)
        seg = tvf.vflip(seg)
    return image, seg


class ComposeLineData:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


def remove_padding(data: LineData, padding_size=(5, 5)):
    if padding_size is None:
        return data
    image = data.image
    graph = data.graph
    h, w, _ = image.shape
    new_image = image[padding_size[0]:(h-padding_size[0]),padding_size[1]:(w-padding_size[1]),:]
    trans_keypoints = [(p[0] - padding_size[1], p[1] - padding_size[0]) for p in graph.points]
    trans_graph = graph.copy()
    trans_graph.update_points_list(trans_keypoints)
    newLineData = LineData(new_image, trans_graph)
    return newLineData


def add_padding(data: LineData, padding_size=(5, 5), bw=False):
    if padding_size is None:
        return data
    image = data.image
    graph = data.graph
    lines = graph.lines
    if not bw:
        h, w, _ = image.shape
    else:
        h, w = image.shape
    new_h = h + padding_size[0] * 2
    new_w = w + padding_size[1] * 2
    if not bw:
        new_image = np.zeros((new_h, new_w, 3), dtype=np.uint8)
        new_image[padding_size[0]:(new_h-padding_size[0]),padding_size[1]:(new_w-padding_size[1]), :] = image
    else:
        new_image = np.zeros((new_h, new_w), dtype=np.uint8)
        new_image[padding_size[0]:(new_h-padding_size[0]), padding_size[1]:(new_w-padding_size[1])] = image
    trans_keypoints = [(p[0] + padding_size[1], p[1] + padding_size[0]) for p in graph.points]
    trans_graph = graph.copy()
    trans_graph.update_graph(trans_keypoints, lines)
    newLineData = LineData(new_image, trans_graph)
    return newLineData


def hori_flip(data: LineData, rm_padding=None, padding = None):
    data = remove_padding(data, rm_padding)
    image = data.image
    graph = data.graph
    keypoints = np.array([(x, y, 0, 1) for x, y in graph.points])
    transform = A.HorizontalFlip(p=1)
    result = transform(image=image, keypoints=keypoints)
    trans_keypoints = [(point[0], point[1]) for point in result["keypoints"]]
    trans_graph = graph.copy()
    trans_graph.update_points_list(trans_keypoints)
    newLineData = LineData(result["image"], trans_graph)
    newLineData = add_padding(newLineData, padding)
    return newLineData


def vert_flip(data: LineData, rm_padding=None, padding = None):
    data = remove_padding(data, rm_padding)
    image = data.image
    graph = data.graph
    keypoints =  np.array([(x, y, 0, 1) for x, y in graph.points])
    transform = A.VerticalFlip(p=1)
    result = transform(image=image, keypoints=keypoints)
    trans_keypoints = [(point[0], point[1]) for point in result["keypoints"]]
    trans_graph = graph.copy()
    trans_graph.update_points_list(trans_keypoints)
    newLineData = LineData(result["image"], trans_graph)
    newLineData = add_padding(newLineData, padding)
    return newLineData