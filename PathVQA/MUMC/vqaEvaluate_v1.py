import os
import argparse
import json
from utils import Logger
from vqaTools.vqa import *
from vqaTools.vqaEval import *
from dataset.utils import pre_answer

def compute_vqa_acc(answer_list_path, epoch=40, res_file_path=30):
    quesFile = answer_list_path
    all_result_list = []
    vqa = VQA(quesFile, quesFile)
    for i in range(epoch):
        resFile = res_file_path.replace('<epoch>', str(i))
        print(resFile)

        # create vqa object and vqaRes object
        vqaRes = vqa.loadRes(resFile, quesFile)

        # create vqaEval object by taking vqa and vqaRes
        vqaEval = VQAEval(vqa, vqaRes, n=2)  # n is precision of accuracy (number of places after decimal), default is 2
        # evaluate results
        vqaEval.evaluate()

        # print accuracies
        acc_dict = {}
        print("\n")
        print("Overall Accuracy is: %.02f\n" % (vqaEval.accuracy['overall']))
        acc_dict['Epoch'] = i + 1
        acc_dict['Overall'] = vqaEval.accuracy['overall']
        print("Per Answer Type Accuracy is the following:")
        for ansType in vqaEval.accuracy['perAnswerType']:
            acc_dict[ansType] = vqaEval.accuracy['perAnswerType'][ansType]
            print("%s : %.02f" % (ansType, vqaEval.accuracy['perAnswerType'][ansType]))
        print("\n")

        # save evaluation results to ./results folder
        accuracyFile = resFile.replace('.json', '_acc.json')
        json.dump(vqaEval.accuracy, open(accuracyFile, 'w'))
        compareFile = resFile.replace('.json', '_compare.json')
        json.dump(vqaEval.ansComp, open(compareFile, 'w'))
        all_result_list.append(acc_dict)
    index = res_file_path.rfind('/')
    compareFile = res_file_path[0:index]
    compareFile = os.path.join(compareFile, 'all_acc.json')
    all_result_list.sort(key=lambda x: x['Overall'])
    for result in all_result_list:
        print(result)
    json.dump(all_result_list, open(compareFile, 'w'))
    print('All accurary file saved to: ', compareFile)

def compute_vqa_acc_topk(answer_list_path, epoch=40, res_file_path=30):
    quesFile = answer_list_path
    vqa = VQA(quesFile, quesFile)
    all_rows = []
    for i in range(epoch):
        resFile_topk = res_file_path.replace('<epoch>', str(i)).replace('.json', '_topk.json')
        # 加载top-k结果
        with open(resFile_topk, 'r') as f:
            topk_data = json.load(f)
        # 将topk转换为qid->list[dict] 支持包含概率的信息
                # 将topk转换为qid->list[str]
        qid_to_topk = { int(item['qid']): [pre_answer(a) for a in item['answers']] for item in topk_data }

        def ans_type(ans: str):
            a = ans.strip().lower()
            if a in {'yes', 'no'}:
                return 'yes/no'
            try:
                float(a)
                return 'number'
            except Exception:
                return 'other'

        def pct(n, d):
            return round(100.0 * n / max(d, 1), 2)

        def compute_at_k(k: int):
            totals = {'overall': 0, 'yes/no': 0, 'number': 0, 'other': 0}
            hits   = {'overall': 0, 'yes/no': 0, 'number': 0, 'other': 0}
            for qid in vqa.getQuesIds():
                gt = pre_answer(vqa.qa[qid]['answer'])
                preds = qid_to_topk.get(int(qid), [])
                t = ans_type(gt)
                totals['overall'] += 1
                totals[t] += 1
                if gt in preds[:k]:
                    hits['overall'] += 1
                    hits[t] += 1
            return {
                'Epoch': i + 1,
                'Overall': pct(hits['overall'], totals['overall']),
                'other': pct(hits['other'], totals['other']),
                'yes/no': pct(hits['yes/no'], totals['yes/no']),
                'number': pct(hits['number'], totals['number']),
            }

        acc1 = compute_at_k(1)
        acc3 = compute_at_k(3)
        acc5 = compute_at_k(5)
        acc1['TopK'] = 1
        acc3['TopK'] = 3
        acc5['TopK'] = 5
        all_rows.extend([acc1, acc3, acc5])

    index = res_file_path.rfind('/')
    out_dir = res_file_path[0:index]
    out_path = os.path.join(out_dir, 'all_acc_topk.json')
    # 为可读性，可以按Epoch与TopK排序（先Epoch，再TopK）
    all_rows.sort(key=lambda x: (x['Epoch'], x['TopK']))
    json.dump(all_rows, open(out_path, 'w'))
    print('All top-k accurary file saved to: ', out_path)