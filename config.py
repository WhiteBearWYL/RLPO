
"""
RLPO 配置文件
所有配置项都以常量形式定义
"""

from datetime import datetime
from enum import Enum, unique

"""
数据库配置
"""

IPS = ["127.0.0.1"]

LATENCY = 5 # 单位：秒

# Database port number
PORT = 5432

# Host username
USER = "root"

# Host password
PASSWD = "12345678"

# Database username
DB_USER = "gpadmin"

# Database password
DB_PASSWD = "123456"

# Database name
DBNAME = "tpch20"

"""
训练/推理配置
"""

# 负载配置
WORKLOAD_NAME = "tpch"
DATA_FOLDER = "text_material/tpch"

# 训练代理配置
LAST_AGENT_PATH = "results/dqn_agent_tpch.pth"

# 运行模式: "offline_train", "online_train", "inference"
MODE = "offline_train"

# 离线训练配置
OFFLINE_TRAIN = {
    "num_episodes": 1800,
    "batch_size": 64,
    "hidden_dim": 256,
    "lr": 0.0005,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.01,
    "epsilon_decay": 0.99,
    "target_update": 100,
    "buffer_capacity": 20000,
    "tmax": 30,
}

# 在线训练配置（预留功能）
ONLINE_TRAIN = {
    "num_steps": 1000,
    "batch_size": 64,
}

# 推理配置
INFERENCE = {
    "max_steps": 50,
}

current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')