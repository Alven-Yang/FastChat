import subprocess

param1 = "hf_qwen_7b_chat"
param2 = "mmlu_ppl"


command = ["zsh", "/home/yanganwen/code/opencompass/run.sh", param1, param2]


result = subprocess.run(command)


