import requests as http
import base64
from typing import Union
import django
# from absl import app
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', "ORM.settings")
django.setup()
from nasbench import api
from nasbench101.models import NASBench101Result


_CREATE_IMAGE = 'create-image'
_DELETE_IMAGE = 'delete-image'
_DELETE_CLUSTER = 'delete-cluster'
_GET_IMAGE = 'get-image'

CREATE_APPEND = 'append'
CREATE_REPLACE = 'replace'

GRAPH_NB101 = 'nas-bench-101'
GRAPH_NB201 = 'nas-bench-201'
GRAPH_DARTS = 'darts-nasnet'

cfg_host = '124.221.166.124'
cfg_port = 4096


class Visualizer():
    def __init__(self, graph_type: str, host: str = cfg_host, port: int = cfg_port) -> None:
        self.port = port
        self.host = host
        self.graph_type = graph_type

    def create_image(self, cluster: str, image: str, payload: dict, type: str = CREATE_REPLACE) -> Union[str, None]:
        obj = {'type': self.graph_type}
        obj.update(payload)
        res = http.post(f'http://{self.host}:{self.port}/{_CREATE_IMAGE}',
                        json={'cluster': cluster, 'image': image, 'payload': obj, 'type': type})
        if res.status_code != 200:
            return res.text

    def delete_image(self, cluster: str, image: str, index: int = -1) -> Union[str, None]:
        res = http.post(f'http://{self.host}:{self.port}/{_DELETE_IMAGE}',
                        json={'cluster': cluster, 'image': image, 'index': index})
        if res.status_code != 200:
            return res.text

    def delete_cluster(self, cluster: str) -> Union[str, None]:
        res = http.post(f'http://{self.host}:{self.port}/{_DELETE_CLUSTER}',
                        json={'cluster': cluster})
        if res.status_code != 200:
            return res.text

    def get_image(self, cluster: str, image: str, page: int = 0, save_path: str = '.', series: str = None) -> Union[str, None]:
        type_with_series = f'{self.graph_type}:{series}' if series is not None else self.graph_type
        res = http.post(f'http://{self.host}:{self.port}/{_GET_IMAGE}',
                        json={'cluster': cluster, 'image': image, 'page': page, 'type': type_with_series})
        if res.status_code != 200:
            return res.text

        file_name = f'{image}_{page}_{series}' if series is not None else f'{image}_{page}'
        with open(f'{save_path}/{file_name}.png', 'wb') as img:
            img_content = res.content.decode('ascii').split(';base64,')[1]
            img.write(base64.b64decode(img_content))


if __name__ == '__main__':
    data_NB101 = [
        {
            'matrix': [
                [0, 1, 1, 1, 0],
                [0, 0, 1, 0, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0]
            ],
            'ops': ['input', 'conv1x1-bn-relu',
                    'conv3x3-bn-relu', 'maxpool3x3', 'output']
        },
        {
            'matrix': [import django
# from absl import app
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', "ORM.settings")
django.setup()
from nasbench import api
from nasbench101.models import NASBench101Result
                [0, 1, 1, 1, 0],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0]
            ],
            'ops': ['input', 'conv1x1-bn-relu',
                    'maxpool3x3', 'conv3x3-bn-relu', 'output']
        },
        {
            'matrix': [
                [0, 1, 1, 0, 1],
                [0, 0, 0, 1, 0],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0]
            ],
            'ops': ['I', 'A', 'A', 'B', 'O']
        },
        {
            'matrix': [
                [0, 1, 1, 1, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 1],
                [0, 0, 0, 0, 0, 0, 0]
            ],
            'ops': ['input', 'conv1x1-bn-relu', 'conv1x1-bn-relu',
                    'conv3x3-bn-relu', 'conv3x3-bn-relu', 'maxpool3x3', 'output']
        }
    ]
    data_NB201 = [
        {
            'arch': '|nor_conv_3x3~0|+|nor_conv_3x3~0|avg_pool_3x3~1|+|skip_connect~0|nor_conv_3x3~1|skip_connect~2|'
        },
        {
            'arch': '|nor_conv_1x1~0|+|avg_pool_3x3~0|nor_conv_1x1~1|+|skip_connect~0|none~1|nor_conv_3x3~2|'
        }
    ]
    data_DARTS = [
        {
            'normal': [
                ['sep_conv_5x5', 1],
                ['sep_conv_3x3', 0],
                ['sep_conv_5x5', 0],
                ['sep_conv_3x3', 0],
                ['avg_pool_3x3', 1],
                ['skip_connect', 0],
                ['avg_pool_3x3', 0],
                ['avg_pool_3x3', 0],
                ['sep_conv_3x3', 1],
                ['skip_connect', 1],
            ],
            'normal_concat': [2, 3, 4, 5, 6],
            'reduce': [
                ['sep_conv_3x3', 1],
                ['sep_conv_5x5', 0],
                ['max_pool_3x3', 1],
                ['sep_conv_5x5', 0],
                ['avg_pool_3x3', 1],
                ['sep_conv_5x5', 0],
                ['skip_connect', 3],
                ['avg_pool_3x3', 2],
                ['sep_conv_3x3', 2],
                ['max_pool_3x3', 1],
            ],
            'reduce_concat': [4, 5, 6]
        }
    ]

    vis_NB101 = Visualizer(GRAPH_NB101)
    data = list(NASBench101Result.objects.all())
    # for i in data:


    for data in data_NB101:
        vis_NB101.create_image('Page A', 'Graph 1', data, CREATE_APPEND)
    vis_NB101.get_image('Page A', 'Graph 1', 3)

    # vis_NB201 = Visualizer(GRAPH_NB201)
    # for data in data_NB201:
    #     vis_NB201.create_image('Page A', 'Graph 2', data, CREATE_APPEND)
    # # vis_NB201.get_image('Page A', 'Graph 2', 0)
    #
    # vis_DARTS = Visualizer(GRAPH_DARTS)
    # for data in data_DARTS:
    #     vis_DARTS.create_image('Page B', 'Graph 3', data, CREATE_APPEND)
    # # vis_DARTS.get_image('Page B', 'Graph 3', series='reduce')
