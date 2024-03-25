import json
import os
import uuid
from collections import defaultdict, OrderedDict
from pprint import pprint
import GPUtil
import pandas as pd
from io import StringIO
import ast
import torch
from flask import Flask, request, jsonify
import subprocess
import random
import string
import time
import datetime
import pytz
import shutil


from fastchat.llm_judge.gen_model_answer import run_eval
from fastchat.serve.flask.utils import calculate_model_scores_dimension, calculate_model_scores_category
from fastchat.utils import str_to_torch_dtype
from flask_utils import (get_free_gpus, append_dict_to_jsonl, get_end_time, get_start_time, parse_params,
                         safe_literal_eval, generate_random_model_id, is_non_empty_file, gen_eval_report,
                         calculate_score, get_total_scores, get_report_by_names, get_report_all, random_uuid,
                         set_gpu, copy_file)
from fastchat.llm_judge.report.assist1 import generate_report, get_system_prompt, get_cache
from fastchat.serve.flask.functions.evalInterfaceV3 import gen_eval_report

app_dir = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(app_dir, 'resources', 'data_config.json')
with open(DATA_PATH, 'r', encoding='utf-8') as file:
    DATA_JSON = json.load(file)
DATA_DICT = {}
for DATA_CATEGORY in DATA_JSON:
    for DATA in DATA_CATEGORY['datasets']:
        DATA_DICT[DATA['data_id']] = DATA
DATA_IDS = [dataset["data_id"] for dataset in DATA_JSON[0]["datasets"]]
MODEL_PATH = os.path.join(app_dir, 'resources', 'model_config.json')
with open(MODEL_PATH) as file:
    MODEL_JSON = json.load(file)
MODEL_DICT = {model["name"].split('/')[-1]: model for model in MODEL_JSON["models"]}
MODEL_NAMES = [model['name'] for model in MODEL_JSON["models"]]
MODEL_IDS = [model['model_id'] for model in MODEL_JSON["models"]]
BASE_PATH = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print("BASE_PATH:", BASE_PATH)
print("DATA_PATH:", DATA_PATH)
print("MODEL_PATH:", MODEL_PATH)
RENAME_DATA = {
    'political_ethics_dataset': '政治伦理',
    'economic_ethics_dataset': '经济伦理',
    'social_ethics_dataset': '社会伦理',
    'cultural_ethics_dataset': '文化伦理',
    'technology_ethics_dataset': '科技伦理',
    'environmental_ethics_dataset': '环境伦理',
    'medical_ethics_dataset': '医疗健康伦理',
    'education_ethics_dataset': '教育伦理',
    'professional_ethics_dataset': '职业道德伦理',
    'cyber_information_ethics_dataset': '网络伦理',
    'international_relations_ethics_dataset': '国际关系与全球伦理',
    'psychology_ethics_dataset': '心理伦理',
    'bioethics_dataset': '生物伦理学',
    'sports_ethics_dataset': '运动伦理学',
    'military_ethics_dataset': '军事伦理'
}

app = Flask(__name__)


@app.route('/get_modelpage_list', methods=['POST'])
def get_modelpage_list():
    request_id = random_uuid()
    result = MODEL_JSON.copy()
    result.update({"request_id": request_id})
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_modelpage_detail', methods=['POST'])
def get_modelpage_detail():
    request_id = random_uuid()
    data = request.json
    if not all(key in data for key in ['model_name']):
        return jsonify({"error": "Missing required fields in the request"}), 400

    MODEL_NAME = data.get('model_name')
    DATA_IDS = list(DATA_DICT.keys())
    print("model_name:", MODEL_NAME, "data_ids:", DATA_IDS)
    # overall_report = calculate_model_scores(DATA_IDS)
    report_per_model, report_per_data = calculate_model_scores_category("moral_bench_test5")
    print("report_per_model:", report_per_model)
    print("report_per_data:", report_per_data)
    # sys_prompt = get_system_prompt()
    # report = generate_report(sys_prompt, overall_report[MODEL_ID]["error_examples"])
    report = get_cache()
    try:
        MODEL_NAME = MODEL_NAME.split('/')[-1] if MODEL_NAME not in report_per_model else MODEL_NAME
    except AttributeError as e:
        print(e)
        return jsonify({"error": f"Model NAME '{MODEL_NAME}' not found in the report", "code": "ModelNotFound"}), 404
    if MODEL_NAME not in report_per_model:
        return jsonify({"error": f"Model NAME '{MODEL_NAME}' not found in the report", "code": "ModelNotFound"}), 404
    else:
        ability_scores = report_per_model[MODEL_NAME]["score_per_category"]
        ability_scores_array = []
        for ability, scores in ability_scores.items():
            ability_scores_array.append({"ability": ability, **scores})

        scores_per_data_id = report_per_model[MODEL_NAME]["scores_per_data_id"]
        data_id_scores = []
        for data_id, scores in scores_per_data_id.items():
            data_id_scores.append(
                {"data_id": data_id, "score": scores["correct"], "total": scores["total"],
                 "accuracy": scores["accuracy"]})
        result = {
            "request_id": str(request_id),
            "model_name": MODEL_NAME,
            "score": report_per_model[MODEL_NAME]["score_total"],
            "correct": report_per_model[MODEL_NAME]["total_correct"],
            "total": report_per_model[MODEL_NAME]["total_questions"],
            "ability_scores": ability_scores_array,
            "data_id_scores": data_id_scores,
            "model_description": MODEL_DICT.get(MODEL_NAME, {}),
            "report": report
        }
        return json.dumps(result, ensure_ascii=False)


@app.route('/get_datapage_list', methods=['POST'])
def get_datapage_list():
    request_id = random_uuid()
    result = DATA_JSON.copy()
    result.append({"request_id": request_id})
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_datapage_detail', methods=['POST'])
def get_datapage_detail():
    request_id = random_uuid()
    data = request.json
    if not all(key in data for key in ['data_id']):
        return jsonify({"error": "Missing required fields in the request"}), 400
    DATA_ID = data.get('data_id')
    DATA_RENAME = RENAME_DATA.get(DATA_ID, None)
    report_per_model, report_per_data = calculate_model_scores_category("moral_bench_test5")

    result = {
        "request_id": request_id,
        "data_id": DATA_ID,
        "data_description": DATA_DICT.get(DATA_ID, {}),
        "score": report_per_data.get(DATA_RENAME, 0),
        "model_ids": list(report_per_model.keys()),
    }
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_leaderboard_detail', methods=['POST'])
def get_leaderboard_detail():
    CATEGORY = ["合规性", "公平性", "知识产权", "隐私保护", "可信度"]
    filter_params = request.json
    categories = filter_params.get('categories', None)
    if categories is None:
        categories = CATEGORY.copy()
    model_sizes = filter_params.get('model_sizes', None)
    datasets = filter_params.get('datasets', None)
    print("categories:", categories, "model_sizes:", model_sizes, "datasets:", datasets)
    filtered_cates = CATEGORY.copy()
    if categories is not None:
        filtered_cates = [cate for cate in CATEGORY if cate in categories]
    filtered_models = [model["name"].split('/')[-1] for model in MODEL_JSON["models"]]
    if model_sizes is not None:
        filtered_models = [model for model in filtered_models if
                           any(size.lower() in model.lower() for size in model_sizes)]
    filtered_data = ["moral_bench_test5"]
    print("filtered_cates:", filtered_cates, "filtered_models:", filtered_models, "filtered_data:", filtered_data)

    report_per_model, report_per_data = calculate_model_scores_category("moral_bench_test5")
    aggregated_scores = {}
    for model_name in filtered_models:
        if model_name not in report_per_model:
            print("model_name not in report_per_model:", model_name)
            continue
        else:
            model_data = report_per_model[model_name]
            aggregated_scores[model_name] = {category: 0 for category in categories}
            aggregated_scores[model_name]['count'] = 0

            for category in categories:
                category_score = model_data['score_per_category'].get(category, {})
                aggregated_scores[model_name][category] = category_score.get('accuracy', 0)

            aggregated_scores[model_name]['count'] = model_data['total_questions']

    print("aggregated_scores:", aggregated_scores)

    final_data = []
    for model_name, scores in aggregated_scores.items():
        if model_name in filtered_models:
            avg_scores = {cat: scores[cat] for cat in categories}
            final_data.append({
                "模型": model_name,
                "发布日期": MODEL_DICT.get(model_name, {}).get('date', ''),
                "发布者": MODEL_DICT.get(model_name, {}).get('promulgator', ''),
                "国内/国外模型": MODEL_DICT.get(model_name, {}).get('country', ''),
                "参数量": MODEL_DICT.get(model_name, {}).get('parameters_size', ''),
                "综合": sum(avg_scores.values()) / len(categories),
                **avg_scores
            })
    print("final_data:", final_data)
    result = {
        "request_id": str(uuid.uuid4()),
        "header": [
                      "模型", "发布者", "发布日期", "国内/国外模型", "参数量", "综合"
                  ] + categories,
        "data": final_data
    }
    return json.dumps(result, ensure_ascii=False)


@app.route('/get_report', methods=['POST'])
def get_report():
    def get_evaluation_results(request_id):
        log_folder = os.path.join(BASE_PATH, "llm_judge", "log")
        os.makedirs(log_folder, exist_ok=True)
        log_path = os.path.join(log_folder, "eval_log.jsonl")
        with open(log_path, 'r') as f:
            for line in f:
                js0 = json.loads(line)
                if request_id in js0:
                    return js0[request_id]
        return None

    data = request.json
    request_id = data.get('request_id')
    if not request_id:
        return jsonify({"error": "Missing request_id in the request"}), 400

    evaluation_results = get_evaluation_results(request_id)
    print("evaluation_results:", evaluation_results)
    if evaluation_results is not None:
        data_ids = evaluation_results["data_ids"]
        model_names = [model_name.split('/')[-1] for model_name in evaluation_results["model_names"]]
        print(__name__, "data_ids:", data_ids, "model_names:", model_names)
        return get_report_by_names(request_id, data_ids, model_names)
    else:
        return jsonify({"error": f"No evaluation results found by request_id {request_id}"}), 400


@app.route('/run_evaluate', methods=['POST'])
def run_evaluate():
    set_gpu()
    global ray
    data = request.json
    params_config = {
        'task_id': (None, str),
        'model_names': ('[]', safe_literal_eval),
        'model_ids': ('[]', safe_literal_eval),
        'data_ids': ('[]', safe_literal_eval)
    }
    params = parse_params(data, params_config)
    request_id = params.get('task_id') if params.get('task_id') else random_uuid()
    model_names = params.get('model_names') if len(params.get('model_names')) > 0 else MODEL_NAMES
    model_ids = params.get('model_ids') if len(params.get('model_ids')) > 0 else MODEL_IDS
    data_ids = params.get('data_ids') if len(params.get('data_ids')) > 0 else DATA_IDS

    if len(model_names) != len(model_ids):
        print(model_names, model_ids)
        return jsonify({"error": "model_names and model_ids should have the same length"}), 400

    revision = data.get('revision', None)
    question_begin = data.get('question_begin', None)
    question_end = data.get('question_end', None)
    max_new_token = data.get('max_new_token', 1024)
    num_choices = data.get('num_choices', 1)
    num_gpus_per_model = data.get('num_gpus_per_model', 1)
    num_gpus_total = data.get('num_gpus_total', 1)
    max_gpu_memory = data.get('max_gpu_memory', 70)
    dtype = str_to_torch_dtype(data.get('dtype', None))
    cache_dir = os.environ.get('CACHE_DIR', "/home/Userlist/madehua/model/")
    print("model_names:", model_names, "model_ids:", model_ids, "data_ids:", data_ids, "cache_dir:", cache_dir)
    failed = []
    if num_gpus_total // num_gpus_per_model > 1:
        import ray
        ray.init()
    else:
        ray = None
    print("ray:", ray)

    try:
        start_time = get_start_time()
        outputs = []
        for data_id in data_ids:
            if data_id not in DATA_IDS:
                if not is_non_empty_file(data_id):
                    return json.dumps({"error": f"data_id {data_id} not found"}), 400
                new_data_dir = os.path.join(BASE_PATH, "llm_judge", "data", str(data_id.split("/")[-1].split(".")[0]))
                new_answer_dir = os.path.join(new_data_dir, "model_answer")
                if not os.path.exists(new_data_dir) or not os.path.isdir(new_data_dir):
                    os.makedirs(new_data_dir, exist_ok=True)
                    os.makedirs(new_answer_dir, exist_ok=True)
                    copy_file(data_id, new_data_dir)
                    os.rename(os.path.join(new_data_dir, data_id.split("/")[-1]), os.path.join(new_data_dir, "question.jsonl"))
                data_id = str(data_id.split("/")[-1].split(".")[0])
                question_file = os.path.join(BASE_PATH, "llm_judge", "data", str(data_id), "question.jsonl")
            for model_name, model_id in zip(model_names, model_ids):
                model_name_saved = model_name.split('/')[-1]
                output_file = os.path.join(BASE_PATH, "llm_judge", "data", str(data_id), "model_answer",
                                           f"{model_name_saved}_{start_time}.jsonl")
                if is_non_empty_file(output_file):
                    print(
                        f"Skipping model_id {model_id} for data_id {data_id} as output file already exists and is non-empty.")
                else:
                    print("eval model:", model_name, model_id)
                    try:
                        run_eval(
                            ray=ray,
                            model_path=model_name, model_id=model_id, question_file=question_file,
                            question_begin=question_begin, question_end=question_end,
                            answer_file=output_file, max_new_token=max_new_token,
                            num_choices=num_choices, num_gpus_per_model=num_gpus_per_model,
                            num_gpus_total=num_gpus_total, max_gpu_memory=max_gpu_memory,
                            dtype=dtype, revision=revision, cache_dir=cache_dir
                        )
                    except AttributeError as e:
                        print("eval model error:", model_name, model_id)
                        print(e)
                        failed.append({"model_id": model_id, "reason": str(e)})
                        continue
                    except torch.cuda.OutOfMemoryError as e1:
                        print("eval model error:", model_name, model_id)
                        print(e1)
                        failed.append({"model_id": model_id, "reason": str(e1)})
                        continue
                temp = {"data_id": data_id,
                        "model_id": model_id, "model_name": model_name,
                        "output": output_file}
                outputs.append(temp)

        end_time = get_end_time()
        result = {
            "outputs": outputs,
            "model_names": model_names,
            "model_ids": model_ids,
            "data_ids": data_ids,
            "time_start": start_time,
            "time_end": end_time,
            "failed": failed
        }
        log_folder = os.path.join(BASE_PATH, "llm_judge", "log")
        os.makedirs(log_folder, exist_ok=True)
        log_path = os.path.join(log_folder, "eval_log.jsonl")
        print("log_path:", log_path)
        append_dict_to_jsonl(log_path, {request_id: result})

        scores = []
        scores_out = []
        for data_id in data_ids:
            scores.append(calculate_model_scores_dimension(data_id.split("/")[-1].split(".")[0]))
            print(scores)
            print(data_id)
        for score in scores:
            # score:tuple
            for item_dict in score:
                for key in item_dict.keys():
                    if start_time in key:
                        scores_out.append({key: {"total_correct": item_dict[key]["total_correct"],
                                                 "total_questions": item_dict[key]["total_questions"]},
                                           "score_total": item_dict[key]["score_total"],
                                           "score_per_category": dict(item_dict[key]["score_per_category"])
                                           })
        result["scores"] = scores_out
        return jsonify(result)
    except subprocess.CalledProcessError:
        return jsonify({"error": "Script execution failed"}), 500


@app.route('/get_eval_report', methods=['POST'])
def get_eval_report():
    data = request.json
    log_folder = os.path.join(BASE_PATH, "llm_judge", "log")
    log_json = None
    question_file = []
    model_name = []
    params_comfig = {
        'task_id': (None, str)
    }

    params = parse_params(data, params_comfig)
    task_id = params.get('task_id')
    print(task_id,str(task_id))
    with open(os.path.join(log_folder, "eval_log.jsonl"), 'r', encoding="utf-8") as f:
        log_lines = list(f)
    for line in reversed(log_lines):
        log = json.loads(line)
        if task_id in log.keys():
            log_json = log
            break
    try:
        if is_non_empty_file(f"./report/report_{task_id}.md"):
            pass
        else:
            for data_id in log_json[task_id]["data_ids"]:
                question_file.append(os.path.join(BASE_PATH, "llm_judge", "data", str(data_id), "question.jsonl"))
            for model in log_json[task_id]["model_names"]:
                model_name.append(model.split("/")[-1])
            time_suffix = log_json[task_id]["outputs"][0]["output"].split("/")[-1].split("_")[-1].split(".")[0]
            gen_eval_report(task_id, question_file, model_name, time_suffix)

            with open(f"./report/report_{task_id}.md", 'r', encoding="utf-8") as f:
                report = f.read()
            return report
    except:
        with open(f"./report/report.md", 'r', encoding="utf-8") as f:
            report = f.read()
        return report


@app.route('/run_generate_eval', methods=['POST'])
def run_generate_eval():

    return None


@app.route('/cal_scores', methods=['POST'])
def cal_scores():
    data = request.json


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5004)
