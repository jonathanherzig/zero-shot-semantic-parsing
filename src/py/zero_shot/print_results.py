import os
from collections import defaultdict
import json

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
RES_PATH = os.path.join(ROOT_DIR, '../../../res/')


def print_result(domains, result):
    header = ['version'] + domains + ['average']
    data = []
    for version in result:
        average = 0
        res = []
        for d in domains:
            score = result[version].get(d)
            res.append(score)
            if score is not None:
                average += score
            else:
                average += 0

        ave = (average*100) / len(domains)
        data.append(([version] + [("%.3f" % (r*100)) if isinstance(r, float) else r for r in res] + ["%.3f" % (ave)], ave))
        data.sort(key=lambda tup: tup[1], reverse=True)

    fmt = '{:<12}' + '{:<14}' * (len(header) - 1)

    print(fmt.format(*header))
    for datum in data:
        datum_to_write = datum[0]
        print(fmt.format(*datum_to_write))


if __name__ == "__main__": # prints results nicely
    results = defaultdict(dict)
    struct_acc = defaultdict(dict)
    infer_acc = defaultdict(dict)
    infer_error_acc = defaultdict(dict)
    infer_ave_rank_acc = defaultdict(dict)
    domains = set()
    for filename in os.listdir(RES_PATH):
        if not filename.endswith('.json'):
            continue
        items = filename.replace('.json', '').split('_')
        domain = items[-1]
        domains.add(domain)
        version = '_'.join(items[1:-1])
        with open(os.path.join(RES_PATH, filename), 'r') as f:
            for line in f:
                #print line
                rec = json.loads(line)
                for key in rec:
                    if key.startswith('dev'):
                        # dev, domain = key.split('_')
                        den_acc = rec[key]['denotation']['accuracy']
                        results[version][domain] = den_acc

                        if rec[key].get('structure_correct') is not None:
                            structure_accuracy = rec[key]['structure_correct']['accuracy']
                            struct_acc[version][domain] = structure_accuracy

                        if rec[key].get('inference_correct') is not None:
                            inference_correct = rec[key]['inference_correct']['accuracy']
                            infer_acc[version][domain] = inference_correct

                        if rec[key].get('inference_error') is not None:
                            inference_error = rec[key]['inference_error']['accuracy']
                            infer_error_acc[version][domain] = inference_error

                        if rec[key].get('inf_ave_rank') is not None:
                            inf_ave_rank = rec[key]['inf_ave_rank']['accuracy']
                            infer_ave_rank_acc[version][domain] = inf_ave_rank


    domains = sorted(list(domains))

    print 'denotation accuracy'
    print_result(domains, results)
    print 'structure accuracy'
    print_result(domains, struct_acc)
    print 'inference accuracy'
    print_result(domains, infer_acc)
    print 'inference error'
    print_result(domains, infer_error_acc)
    print 'inference average rank'
    print_result(domains, infer_ave_rank_acc)




