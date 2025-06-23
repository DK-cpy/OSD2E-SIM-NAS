from tabnanny import verbose
import django
import numpy as np
import os
# from nats_bench import create
os.environ.setdefault('DJANGO_SETTINGS_MODULE', "ORM.settings")
django.setup()
from nas_201_api import NASBench201API as API
from nas_201_api import ArchResults
from nasbench201.models import NASBench201Result

# api2 = create("/home/satan/桌面/NATS-tss-v1_0-3ffb9-simple", 'tss', fast_mode=True, verbose=False)

# cifar10-valid : training the model on the CIFAR-10 training set.
#         -- cifar10 : training the model on the CIFAR-10 training + validation set.
#         -- cifar100 : training the model on the CIFAR-100 training set.
#         -- ImageNet16-120
dataset = ["cifar10-valid", "cifar10", "cifar100", "ImageNet16-120"]
res_hash = {
    "cifar10-valid":["train", "x-valid", "ori-test"],
"cifar10":["train",  "ori-test"],
"cifar100":["train", "x-valid", "ori-test", "x-test"],
"ImageNet16-120":["train", "x-valid", "ori-test"],
}
if __name__ == '__main__':
    print("in")
    # NASBench201Result.objects.all().delete()
    api = API('/home/satan/Downloads/NAS-Bench-201-v1_1-096897.pth')
    print(1)
    # x = list(NASBench201Result.objects.all()[9010:])
    # bug = []
    # ans = []
    # l = len(x)
    # for i in range(9010,l):
    #     # info12 = api.query_meta_info_by_index(i, "12")
    #     # info200 = api.query_meta_info_by_index(i, "200")
    #     # res12, res200, cost12, cost200 = dict(), dict(), dict(), dict()
    #     # for j in dataset:
    #     #     res12[j] = dict()
    #     #     res200[j] = dict()
    #     #     cost12[j] = info12.get_compute_costs(j)
    #     #     cost200[j] = info200.get_compute_costs(j)
    #     #     for k in res_hash[j]:
    #     #         # print(i, j, k)
    #     #         res12[j][k] = info12.get_metrics(j, k)
    #     #         res200[j][k] = info200.get_metrics(j, k)
    #
    #     #         if np.isnan(res12[j][k]["loss"]):
    #     #             res12[j][k]["loss"] = 0
    #     #         if np.isnan(res200[j][k]["loss"]):
    #     #             res200[j][k]["loss"] = 0
    #     result = dict()
    #
    #     for j in dataset:
    #         result[j] = dict()
    #         for k in ["12", "200"]:
    #             result[j][k] = api2.get_cost_info(i, j, hp=k)
    #     x[i].result = result
    #     x[i].save()
    #     # ans.append(
    #     #     NASBench201Result(
    #     #         id=i,
    #     #         index=i,
    #     #         phenotype=info12.arch_str,
    #     #         res12=res12,
    #     #         res200=res200,
    #     #         cost12=cost12,
    #     #         cost200=cost200,
    #     #         result=result
    #     #     )
    #     # )
    #     # if i and i % 128 == 0:
    #     print(i)
    # # NASBench201Result.objects.bulk_update(x, update_fields = ['result'])
    #         # try:
    #         #     NASBench201Result.objects.bulk_create(ans)
    #         # except Exception:
    #         #     print(ans[0].res200)
    #         #     break
    #         # ans.clear()
    #         # break
    # # res_metrics = info.get_metrics('cifar10', 'train')  # This is a dict with metric names as keys
    # # cost_metrics = info.get_compute_costs('cifar100')  # This is a dict with metric names as keys, e.g., flops, params, latency
    # # print(res_metrics)
    # # print(cost_metrics)
    # # print(info.arch_str)
