import django
import numpy as np
import os
import json
os.environ.setdefault('DJANGO_SETTINGS_MODULE', "ORM.settings")
django.setup()
from mobilenetv3.models import MobileNetV3Result
from resnet50.models import ResNet50Result

if __name__ == '__main__':
    v3 = json.load(open("r50_data.json", "r"))
    ans = []
    for i in v3:
        ans.append(
            ResNet50Result(
                phenotype = i['phenotype'],
                params = i['params'],
                flops = i['flops'],
                # latency = i['latency'],
                valid_err = i['valid_err'],
                test_err = i['test_err']
            )
        )
    ResNet50Result.objects.bulk_create(ans)