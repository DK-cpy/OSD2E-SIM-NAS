import django
# from absl import app
import os
os.environ.setdefault('DJANGO_SETTINGS_MODULE', "ORM.settings")
django.setup()
from nasbench import api
from nasbench101.models import NASBench101Result
# Replace this string with the path to the downloaded nasbench.tfrecord before
# executing.
NASBENCH_TFRECORD = 'nasbench_full.tfrecord'

INPUT = 'input'
OUTPUT = 'output'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
NASBENCH_TFRECORD = '/home/satan/Downloads/nasbench_full.tfrecord'
if __name__ == '__main__':
    results = []
    nasbench = api.NASBench(NASBENCH_TFRECORD)
    # NASBench101Result.objects.all().delete()
    print(len(nasbench.fixed_statistics), len(nasbench.computed_statistics))
    for index, i in enumerate(nasbench.fixed_statistics):
        print(i)
        db = NASBench101Result.objects.get(index=i)
        fix = nasbench.fixed_statistics[i]
        fix["module_adjacency"] = fix["module_adjacency"].tolist()
        for j in fix:
            assert fix[j] == db.phenotype[j]
        print(db.result)
        print(nasbench.computed_statistics[i])
        assert db.result == nasbench.computed_statistics[i]
        assert db.epoch4 == nasbench.computed_statistics[i][4]
        assert db.epoch12==nasbench.computed_statistics[i][12]
        assert db.epoch36==nasbench.computed_statistics[i][36]
        assert db.epoch108==nasbench.computed_statistics[i][108]
        continue

        # results.append(
        #     NASBench101Result(
        #         index=i,
        #         phenotype=fix,
        #         result=nasbench.computed_statistics[i],
        #         epoch4=nasbench.computed_statistics[i][4],
        #         epoch12=nasbench.computed_statistics[i][12],
        #         epoch36=nasbench.computed_statistics[i][36],
        #         epoch108=nasbench.computed_statistics[i][108]
        #     )
        # )
        # if index % 256 == 0:
        #     print(index)
        #     NASBench101Result.objects.bulk_create(results)
        #     results.clear()
    print(results)