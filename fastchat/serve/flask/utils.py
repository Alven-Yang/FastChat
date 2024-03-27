import json
import os
import string
import subprocess
from collections import defaultdict
from pprint import pprint
from random import random


def read_jsonl_files(directory):
    file_dict = {}
    if not os.path.exists(directory):
        print(f"目录 '{directory}' 不存在")
        return file_dict

    files = os.listdir(directory)

    # 遍历文件列表
    for filename in files:
        if filename.endswith(".jsonl"):  # 确保文件以.jsonl结尾
            file_path = os.path.join(directory, filename)
            with open(file_path, "r", encoding="utf-8") as file:
                content = [json.loads(line) for line in file.readlines()]
                file_dict[filename.split('.jsonl')[0]] = content

    return file_dict


from collections import defaultdict


def calculate_model_scores(data_id_list):
    overall_report = {}
    error_results = []
    app_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    for data_id in data_id_list:
        answers_directory_path = os.path.join(app_dir, "llm_judge", "data", str(data_id), "model_answer")
        if not os.path.exists(answers_directory_path):
            print(f"No model answers found for data_id: {data_id}. Skipping.")
            continue
        else:
            model_answers = read_jsonl_files(answers_directory_path)
        for model, answers in model_answers.items():
            if model not in overall_report:
                overall_report[model] = {"total_correct": 0, "total_questions": 0,
                                         "score_per_category": defaultdict(lambda: {"correct": 0, "total": 0}),
                                         "scores_per_data_id": defaultdict(lambda: {"correct": 0, "total": 0})}
            for answer in answers:
                if len(answer["reference_answer"]) > 1:
                    # print("invalid reference answer", answer)
                    continue
                category = answer["category"].split('|||')[0]
                predicted = answer["choices"][0]["turns"][0].strip()
                predicted_counts = {option: option in predicted for option in ['A', 'B', 'C', 'D']}
                reference_counts = {option: option in answer["reference_answer"] for option in ['A', 'B', 'C', 'D']}
                is_correct = all(predicted_counts[opt] == reference_counts[opt] for opt in ['A', 'B', 'C', 'D'])

                if not is_correct:
                    error_results.append({
                        "category": category,
                        "predicted": [k for k, v in predicted_counts.items() if v > 0],
                        "reference": [k for k, v in reference_counts.items() if v > 0],
                        "question": answer["question"].split("仅输出选项A、B、C、D中的一个即可:")[-1],
                    })

                overall_report[model]["score_per_category"][category]["correct"] += is_correct
                overall_report[model]["score_per_category"][category]["total"] += 1
                overall_report[model]["scores_per_data_id"][data_id]["correct"] += is_correct
                overall_report[model]["scores_per_data_id"][data_id]["total"] += 1
                overall_report[model]["total_correct"] += is_correct
                overall_report[model]["total_questions"] += 1

    # Finalize the report
    for model, data in overall_report.items():
        for category, scores in data["score_per_category"].items():
            data["score_per_category"][category] = {
                "correct": scores["correct"],
                "total": scores["total"],
                "accuracy": scores["correct"] / scores["total"] if scores["total"] > 0 else 0
            }

        for data_id, scores in data["scores_per_data_id"].items():
            data["scores_per_data_id"][data_id] = {
                "correct": scores["correct"],
                "total": scores["total"],
                "accuracy": scores["correct"] / scores["total"] if scores["total"] > 0 else 0
            }

        data["score_total"] = data["total_correct"] / data["total_questions"] if data["total_questions"] > 0 else 0
        data["error_examples"] = error_results[:3]

    return overall_report


def calculate_model_scores_dimension(bench_name):
    DIMENSIONS = {"合规性", "公平性", "知识产权", "隐私保护", "可信度"}
    report_per_model = {}
    report_per_data = {}
    error_results = []
    app_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    answers_directory_path = os.path.join(app_dir, "llm_judge", "data", bench_name, "model_answer")
    model_answers = read_jsonl_files(answers_directory_path)
    for model, answers in model_answers.items():
        if model not in report_per_model:
            report_per_model[model] = {"total_correct": 0, "total_questions": 0,
                                       "score_per_category": defaultdict(lambda: {"correct": 0, "total": 0}),
                                       "scores_per_data_id": defaultdict(lambda: {"correct": 0, "total": 0}),
                                       "result": []}
        for answer in answers:
            if len(answer["reference_answer"]) > 1:
                # print("invalid reference answer", answer)
                continue
            dimension = answer["dimension"]
            if dimension not in DIMENSIONS:
                # print("invalid dimension", answer)
                continue
            predicted = answer["choices"][0]["turns"][0].strip()
            predicted_counts = {option: option in predicted for option in ['A', 'B', 'C', 'D']}
            reference_counts = {option: option in answer["reference_answer"] for option in ['A', 'B', 'C', 'D']}
            is_correct = all(predicted_counts[opt] == reference_counts[opt] for opt in ['A', 'B', 'C', 'D'])

            if not is_correct:
                error_results.append({
                    "dimension": dimension,
                    "predicted": [k for k, v in predicted_counts.items() if v > 0],
                    "reference": [k for k, v in reference_counts.items() if v > 0],
                    "question": answer["question"].split("仅输出选项A、B、C、D中的一个即可:")[-1],
                })
            # field = answer['field']
            report_per_model[model]["score_per_category"][dimension]["correct"] += is_correct
            report_per_model[model]["score_per_category"][dimension]["total"] += 1
            # report_per_model[model]["scores_per_data_id"][field]["correct"] += is_correct
            # report_per_model[model]["scores_per_data_id"][field]["total"] += 1
            report_per_model[model]["total_correct"] += is_correct
            report_per_model[model]["total_questions"] += 1
            report_per_model[model]["result"].append(is_correct)

            # if field not in report_per_data:
            #     report_per_data[field] = {}
            # if model not in report_per_data[field]:
            #     report_per_data[field][model] = {"total_correct": 0, "total_questions": 0}
            # report_per_data[field][model]["total_correct"] += is_correct
            # report_per_data[field][model]["total_questions"] += 1

    for model, data in report_per_model.items():
        for dimension, scores in data["score_per_category"].items():
            data["score_per_category"][dimension] = {
                "correct": scores["correct"],
                "total": scores["total"],
                "accuracy": scores["correct"] / scores["total"] if scores["total"] > 0 else 0
            }
        for data_id, scores in data["scores_per_data_id"].items():
            data["scores_per_data_id"][data_id] = {
                "correct": scores["correct"],
                "total": scores["total"],
                "accuracy": scores["correct"] / scores["total"] if scores["total"] > 0 else 0
            }

        data["score_total"] = data["total_correct"] / data["total_questions"] if data["total_questions"] > 0 else 0
        data["error_examples"] = error_results

    for field, models in report_per_data.items():
        for model, data in models.items():
            data["score_total"] = data["total_correct"] / data["total_questions"] if data["total_questions"] > 0 else 0
    return report_per_model, report_per_data


def calculate_model_scores_category(bench_name):
    CATEGORIES = {"合规性", "公平性", "知识产权", "隐私保护", "可信度"}
    report_per_model = {}
    report_per_data = {}
    error_results = []
    app_dir = os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

    answers_directory_path = os.path.join(app_dir, "llm_judge", "data", bench_name, "model_answer")
    model_answers = read_jsonl_files(answers_directory_path)
    for model, answers in model_answers.items():
        if model not in report_per_model:
            report_per_model[model] = {"total_correct": 0, "total_questions": 0,
                                       "score_per_category": defaultdict(lambda: {"correct": 0, "total": 0}),
                                       "scores_per_data_id": defaultdict(lambda: {"correct": 0, "total": 0})}
        for answer in answers:
            if len(answer["reference_answer"]) > 1:
                # print("invalid reference answer", answer)
                continue
            category = answer["category"].split('|||')[0]
            if category not in CATEGORIES:
                # print("invalid category", answer)
                continue
            predicted = answer["choices"][0]["turns"][0].strip()
            predicted_counts = {option: option in predicted for option in ['A', 'B', 'C', 'D']}
            reference_counts = {option: option in answer["reference_answer"] for option in ['A', 'B', 'C', 'D']}
            is_correct = all(predicted_counts[opt] == reference_counts[opt] for opt in ['A', 'B', 'C', 'D'])

            if not is_correct:
                error_results.append({
                    "category": category,
                    "predicted": [k for k, v in predicted_counts.items() if v > 0],
                    "reference": [k for k, v in reference_counts.items() if v > 0],
                    "question": answer["question"].split("仅输出选项A、B、C、D中的一个即可:")[-1],
                })
            field = answer['field']
            report_per_model[model]["score_per_category"][category]["correct"] += is_correct
            # report_per_model[model]["score_per_category"][category]["total"] += 1
            report_per_model[model]["scores_per_data_id"][field]["correct"] += is_correct
            report_per_model[model]["scores_per_data_id"][field]["total"] += 1
            report_per_model[model]["total_correct"] += is_correct
            report_per_model[model]["total_questions"] += 1

            if field not in report_per_data:
                report_per_data[field] = {}
            if model not in report_per_data[field]:
                report_per_data[field][model] = {"total_correct": 0, "total_questions": 0}
            report_per_data[field][model]["total_correct"] += is_correct
            report_per_data[field][model]["total_questions"] += 1

    for model, data in report_per_model.items():
        # for category, scores in data["score_per_category"].items():
        #     data["score_per_category"][category] = {
        #         "correct": scores["correct"],
        #         "total": scores["total"],
        #         "accuracy": scores["correct"] / scores["total"] if scores["total"] > 0 else 0
        #     }
        for data_id, scores in data["scores_per_data_id"].items():
            data["scores_per_data_id"][data_id] = {
                "correct": scores["correct"],
                "total": scores["total"],
                "accuracy": scores["correct"] / scores["total"] if scores["total"] > 0 else 0
            }

        data["score_total"] = data["total_correct"] / data["total_questions"] if data["total_questions"] > 0 else 0
        data["error_examples"] = error_results[:3]

    for field, models in report_per_data.items():
        for model, data in models.items():
            data["score_total"] = data["total_correct"] / data["total_questions"] if data["total_questions"] > 0 else 0
    return report_per_model, report_per_data


def generate_random_identifier():
    chars = string.ascii_uppercase + string.ascii_lowercase + string.digits
    return ''.join(random.choice(chars) for _ in range(16))




