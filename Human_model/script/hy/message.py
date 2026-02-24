import urllib.parse
import os
import requests
import json
from socket import gethostname
def message_to_feishu(message):
    # 获取主机名并提取第一个点前面的部分
    full_hostname = gethostname()
    short_hostname = full_hostname.split('.')[0]

    # 优先获取 SLURM ARRAY JOB ID 格式
    slurm_array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID")
    slurm_array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = None
    if slurm_array_job_id and slurm_array_task_id:
        # 使用 ARRAYJOBID_ARRAYINDEX 格式
        job_id = f"{slurm_array_job_id}_{slurm_array_task_id}"
    else:
        # 回退到普通 SLURM_JOB_ID
        job_id = os.environ.get("SLURM_JOB_ID")

    # 构建格式化消息
    if job_id is None:
        formatted_message = f"[{short_hostname}] {message}"
    else:
        formatted_message = f"[{job_id}@{short_hostname}] {message}"

    print(formatted_message)

    # 这里写你需要通知的渠道或者机器人的URL