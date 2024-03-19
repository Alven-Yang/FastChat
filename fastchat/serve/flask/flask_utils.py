import os
import subprocess
import json
import time
import datetime
import pytz
import ast
import string, random
import uuid
from collections import defaultdict
from fastchat.serve.flask.utils import calculate_model_scores, read_jsonl_files, calculate_model_scores2
from fastchat.llm_judge.report.assist1 import get_cache


def append_dict_to_jsonl(file_path, data_dict):
    with open(file_path, 'a', encoding='utf-8') as f:
        print("save the file_path to", file_path)
        try:
            json_str = json.dumps(data_dict, ensure_ascii=False)
            f.write(json_str + '\n')
        except TypeError as e:
            print("TypeError: ", e, data_dict)
        except UnboundLocalError as e2:
            print("UnboundLocalError: ", e2, data_dict)


def get_free_gpus():
    try:
        # 执行 nvidia-smi 命令
        cmd = "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8")

        # 分析输出结果
        free_gpus = []
        lines = output.strip().split("\n")
        for line in lines:
            index, memory_used = line.split(", ")
            if int(memory_used) <= 100:
                free_gpus.append(int(index))

        return free_gpus
    except Exception as e:
        print(f"Error: {e}")
        return []


def get_start_time():
    start_time = time.time()
    dt_utc = datetime.datetime.fromtimestamp(start_time, tz=pytz.utc)
    dt_beijing = dt_utc.astimezone(pytz.timezone("Asia/Shanghai"))
    formatted_start_time = dt_beijing.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_start_time


def get_end_time():
    end_time = time.time()
    dt_utc = datetime.datetime.fromtimestamp(end_time, tz=pytz.utc)
    dt_beijing = dt_utc.astimezone(pytz.timezone("Asia/Shanghai"))
    formatted_end_time = dt_beijing.strftime('%Y-%m-%d %H:%M:%S')
    return formatted_end_time


def parse_params(data, params_config):
    """
    根据提供的参数配置解析请求数据。

    :param data: 包含参数的字典。
    :param params_config: 一个字典，键为参数名，值为包含默认值和解析函数的元组。
    :return: 解析后的参数字典。
    """
    parsed_params = {}
    for param, (default, parse_func) in params_config.items():
        value_str = data.get(param, default)
        value = parse_func(value_str) if value_str else default
        parsed_params[param] = value

    return parsed_params


def safe_literal_eval(value_str):
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        return value_str


def generate_random_model_id():
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(16))


def calculate_score(result_dict):
    score_result = {}
    for model, model_result in result_dict.items():
        category_status = defaultdict(list)
        for answer in model_result:
            category = answer["category"].split('|||')[0]
            pred = answer["choices"][0]["turns"][0].split('')[0]
            pred_counts = {option: pred.count(option) for option in ['A', 'B', 'C', 'D']}
            refer_counts = {option: answer["reference_answer"].count(option) for option in ['A', 'B', 'C', 'D']}
            status = all(pred_counts[option] == refer_counts[option] for option in ['A', 'B', 'C', 'D'])
            category_status[category].append(status)

        category_score = {k: (sum(v) / len(v), sum(v), len(v)) for k, v in category_status.items()}
        total_correct = sum(v[1] for v in category_score.values())
        total_questions = sum(v[2] for v in category_score.values())
        score_result[model] = (
            total_correct, total_questions, total_correct / total_questions if total_questions else 0)

    return score_result


def get_total_scores(model_scores):
    total_scores = {}
    for model, scores in model_scores.items():
        total_scores[model] = sum(scores.values())
    return total_scores


def get_report_by_names(request_id, data_ids, model_names):
    report_per_model, report_per_data = calculate_model_scores2("moral_bench_test5")
    categories = ['合规性', '公平性', '知识产权', '隐私保护', '可信度']
    header = ['Model ID', 'Total Score'] + categories + ["Evaluate Time", "Report"]
    leaderboard = [header]
    for model, model_data in report_per_model.items():
        if model not in model_names:
            print("model not in model_names:", model, model_names)
            continue
        else:
            row = [model]
            total_correct = model_data['total_correct']
            total_questions = model_data['total_questions']
            total_score = total_correct / total_questions if total_questions > 0 else 0
            row.append(total_score)
            for category in categories:
                score_per_category_id = model_data['score_per_category'].get(category, {"correct": 0, "total": 0})
                category_score = score_per_category_id['correct'] / score_per_category_id['total'] \
                    if score_per_category_id['total'] > 0 else 0
                row.append(category_score)
            # report = get_cache()
            report = ""
            row.append(get_end_time())
            row.append(report)
            leaderboard.append(row)
    return json.dumps({"request_id": request_id, "leaderboard": leaderboard}, ensure_ascii=False)


def get_report_all():
    report_per_model, report_per_data = calculate_model_scores2("moral_bench_test5")
    result = {}
    for model, model_data in report_per_model.items():
        total_correct = model_data['total_correct']
        total_questions = model_data['total_questions']
        total_score = total_correct / total_questions if total_questions > 0 else 0
        report = get_cache()
        model_data.update({"Total Score": total_score, "Report": report, "Evaluate Time": get_end_time()})
        result.update({model: model_data})
    return result


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


def is_non_empty_file(file_path):
    return os.path.isfile(file_path) and os.path.getsize(file_path) > 0


def gen_eval_report(task_id, question_file_path, model_name, time_suffix):
    return None

