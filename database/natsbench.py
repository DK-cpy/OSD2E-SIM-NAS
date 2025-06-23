import django
import numpy as np
import os
from nats_bench import create
os.environ.setdefault('DJANGO_SETTINGS_MODULE', "ORM.settings")
django.setup()
from natsbenchsss.models import NATSBenchResult
from nats_bench import create
api = create('/home/satan/NATS-sss-v1_0-50262-simple', 'sss', fast_mode=True, verbose=True)
l = 32768
if __name__ == '__main__':
    results = []
    NATSBenchResult.objects.all().delete()
    for i in range(32768):
        # print(i)
        tmp = {
                "cifar10-valid":dict(),
                "cifar10":dict(),
                "cifar100":dict(),
                "ImageNet16-120":dict()
            }
        for w in ["01", '12', '90', '200']:
            try:
                x = api.query_by_index(i, hp=w).state_dict()
            except Exception:
                break
            for j, k in x['all_results'].items():
                if w not in tmp[j[0]]:
                    tmp[j[0]][w] = dict()
                tmp[j[0]][w][j[1]] = k
        results.append(
            NATSBenchResult(
                id=i,
                index=i,
                phenotype=x['arch_str'],
                cifar10=tmp["cifar10"],
                cifar100=tmp["cifar100"],
                cifar10_valid=tmp["cifar10-valid"],
                ImageNet16_120=tmp["ImageNet16-120"],
                # genotype=x['all_results']["arch_config"]["genotype"]
            )
        )
        if i and i % 1024 == 0:
            NATSBenchResult.objects.bulk_create(results)
            results.clear()
    NATSBenchResult.objects.bulk_create(results)