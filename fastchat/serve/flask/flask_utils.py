import os, sys
import subprocess
import json
import time
import datetime
import pytz
import ast
import string, random
import uuid
import torch
import shutil
from minio import Minio
from minio.error import S3Error
from collections import defaultdict
# 从对象存储上生成下载文件链接
import boto3
from dotenv import load_dotenv

sys.path.append("./fastchat")
from collections import defaultdict
from utils import calculate_model_scores, read_jsonl_files, calculate_model_scores_category, calculate_model_scores_dimension
from llm_judge.report.assist1 import get_cache

load_dotenv("./.env", verbose=True, override=True)


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
            if int(memory_used) <= 300:
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
    """
    参数说明：
    """
    score_result = {}
    for model, model_result in result_dict.items():
        category_status = defaultdict(list)
        for answer in model_result:
            category = answer["category"].split('|||')[0]
            pred = answer["choices"][0]["turns"][0][0]
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
    report_per_model, report_per_data = calculate_model_scores_category("moral_bench_test5")
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
    report_per_model, report_per_data = calculate_model_scores_category("moral_bench_test5")
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


def set_all_gpus():
    free_gpus = get_free_gpus()
    if free_gpus:
        for i, selected_gpu in enumerate(free_gpus):
            # 在代码中设置要使用的GPU
            torch.cuda.set_device(selected_gpu)
            print(f"已经指定GPU {selected_gpu} 进行运算。")
        print(f"已指定所有共计{len(free_gpus)}个空闲GPU进行运算。")
    else:
        print("没有检测到空闲的GPU。")

def set_gpu():
    free_gpus = get_free_gpus()
    if free_gpus:
        # 如果存在空闲的GPU，选择第一个空闲的GPU
        selected_gpu = free_gpus[0]
        # 直接在代码中设置要使用的GPU
        torch.cuda.set_device(selected_gpu)
        print(f"已经指定GPU {selected_gpu} 进行运算。")
    else:
        print("没有检测到空闲的GPU。")



def copy_file(source_file, destination_folder):
    try:
        # 使用 shutil 的 copy2 函数来复制文件，保留元数据（如修改时间）
        shutil.copy2(source_file, destination_folder)
        print("文件复制成功！")
    except FileNotFoundError:
        print("找不到源文件或目标文件夹。")
    except PermissionError:
        print("权限错误，无法复制文件。")
    except Exception as e:
        print("发生了未知错误:", e)


# def post_request(route, data):
#     url = f"http://10.110.147.178:5004/{route}"
#     headers = {
#         "Content-Type": "application/json"
#     }
#     params_config = {
#         "task_id": ()
#     }


# 上传文件到对象存储
def upload_file(source_file,filename):
    client = Minio(os.getenv("OSS_IP"),
        access_key=os.getenv("OSS_ACCESS_KEY_ID"),
        secret_key=os.getenv("OSS_ACCESS_KEY_SECRET"),
        secure=False,
    )
    bucket_name = "generation"
    try:
        # Make the bucket if it doesn't exist.
        found = client.bucket_exists(bucket_name)
        if not found:
            client.make_bucket(bucket_name)
            print("Created bucket", bucket_name)
        else:
            print("Bucket", bucket_name, "already exists")

        # 获取文件名
        source_filename = filename

        # 上传文件
        client.fput_object(
            bucket_name, source_filename, source_file,
        )
        print(
            source_file, "successfully uploaded as object",
            source_filename, "to bucket", bucket_name,
        )
        return {"status": "success"}
    except S3Error as err:
        print("Error:", err)
        return {"status": "error", "message": str(err)}


# 获取下载链接
def generate_presigned_url(destination_file):
    # Configure the S3 client
    s3 = boto3.client("s3",
        aws_access_key_id="dmeinbi",
        aws_secret_access_key="Denb-emd98-semb",
        endpoint_url="http://192.144.141.249:52300",
    )
    bucket_name = "generation"

    # 生成下载链接
    presigned_url = s3.generate_presigned_url(
        ClientMethod="get_object",
        Params={
            "Bucket": bucket_name,
            "Key": destination_file,
        },
        ExpiresIn=100*360*24*3600,  # 设置链接有效时间
    )
    print("Presigned URL for downloading the file:", presigned_url)

    return presigned_url



if __name__ == '__main__':
    scores = []
    scores_out = []
    for data_id in ["all_questions3"]:
        scores.append(calculate_model_scores_dimension(data_id.split("/")[-1].split(".")[0]))
        for score in scores:
            # score:tuple
            for item_dict in score:
                for key in item_dict.keys():
                    if "2024-03-22" in key:                        
                        with open(f"/home/Userlist/yanganwen/temp/oss/{key}.json", "w", encoding="utf-8") as f:
                            json.dump(item_dict[key]["error_examples"], f, ensure_ascii=False, indent=4)
                            upload_file(f"/home/Userlist/yanganwen/temp/oss/{key}.json", f"{key}.json")
                        with open(f"/home/Userlist/yanganwen/temp/oss/{key}_result.json", "w", encoding="utf-8") as f:
                            json.dump(item_dict[key]["result"], f, ensure_ascii=False, indent=4)
                        upload_file(f"/home/Userlist/yanganwen/temp/oss/{key}_result.json", f"{key}_result.json")
                        scores_out.append({key: {"total_correct": item_dict[key]["total_correct"],
                                                 "total_questions": item_dict[key]["total_questions"]},
                                           "score_total": item_dict[key]["score_total"],
                                           "score_per_dimension": dict(item_dict[key]["score_per_category"]),
                                           "error_file_path": generate_presigned_url(f"{key}.json"),
                                           "result": generate_presigned_url(f"{key}_result.json")
                                           })

    # print(scores[0][0], len(scores))
    print(scores_out[0], len(scores_out))
