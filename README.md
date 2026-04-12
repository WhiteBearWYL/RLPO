
# RLPO - Reinforcement Learning Partitioning Advisor

复现 SIGMOD 2020 论文《Learning a Partitioning Advisor for Cloud Databases》的工作。

## 项目结构

```
RLPO/
├── encoding.py         # 状态编码模块
├── action.py           # 动作定义和编码模块
├── env.py              # 强化学习环境
├── agent.py            # DQN 智能体
├── data_loader.py      # 数据加载工具
├── db.py               # 数据库操作工具
├── online_train.py     # 在线训练模块
├── config.py           # 配置文件（常量定义）
├── main.py             # 主程序入口
├── requirements.txt    # 依赖文件
├── text_material/      # 训练数据目录
│   └── tpch/           # TPCH 数据集
│       ├── schema.txt          # 数据库 schema
│       ├── row_count.txt       # 表行数
│       ├── dist_key.txt        # 初始分布键配置
│       └── train_queries/      # 训练查询
└── results/            # 结果目录
    └── tpch/
        ├── dqn_agent.pth              # 训练好的模型
        ├── dqn_agent_online.pth       # 在线训练模型
        ├── training_curves.png         # 训练曲线
        ├── training_stats.pkl          # 训练统计
        └── z_{timestamp}.txt           # 推理结果
```

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置项目

编辑 `config.py` 文件，配置以下内容：

#### 数据库配置（在线训练需要）
```python
IPS = ["127.0.0.1"]
PORT = 5432
DB_USER = "gpadmin"
DB_PASSWD = "123456"
DBNAME = "tpch20"
```

#### 训练配置
```python
WORKLOAD_NAME = "tpch"
DATA_FOLDER = "text_material/tpch"
LAST_AGENT_PATH = ""  # 上次训练的模型路径，为空则重新训练
MODE = "offline_train"  # 可选: offline_train, online_train, inference
```

### 3. 运行

#### 离线训练
```bash
python3 main.py
```

#### 在线训练
```bash
# 先设置 MODE = "online_train"，然后运行
python3 main.py
```

#### 推理
```bash
# 先设置 MODE = "inference" 和 LAST_AGENT_PATH，然后运行
python3 main.py
```

## 数据准备

数据应放置在 `text_material/{workload_name}/` 目录下，包含以下文件：

### 1. schema.txt - 数据库 schema
JSON 格式，定义表结构
```json
{
  "CUSTOMER": {"C_CUSTKEY": "INTEGER", ...},
  "ORDERS": {"O_ORDERKEY": "INTEGER", ...},
  ...
}
```

### 2. row_count.txt - 表行数
格式：`"表名" : "行数", "表名":"行数", ...`

### 3. dist_key.txt - 初始分布键配置
JSON 格式，定义表的初始分区状态
```json
{
  "CUSTOMER": {
    "KEY": "C_CUSTKEY",
    "REPLICATED": "FALSE",
    "REPLICATE_ABLE": "TRUE"
  },
  ...
}
```

### 4. train_queries/ - 训练查询
包含多个 `.sql` 文件，每个文件是一个查询

## 动作类型

项目支持四种动作：

1. **PARTITION_TABLE** - 按某个属性分区表
2. **REPLICATE_TABLE** - 复制表
3. **ACTIVATE_EDGE** - 激活连接边
4. **DEACTIVATE_EDGE** - 停用连接边

## 运行模式

### 1. 离线训练 (offline_train)
- 使用基于成本模型的奖励函数
- 不连接实际数据库
- 速度快，适合初步训练

### 2. 在线训练 (online_train)
- 使用实际执行查询的延迟计算奖励
- 需要连接实际数据库
- 包含查询延迟缓存机制，提高训练效率
- 奖励更准确，但速度较慢

### 3. 推理 (inference)
- 使用训练好的模型输出分区推荐
- 结果保存到 `results/{workload}/z_{timestamp}.txt`

## 超参数调整

在 `config.py` 中可以调整以下超参数：

### 离线训练超参数
```python
OFFLINE_TRAIN = {
    "num_episodes": 1800,      # 训练回合数
    "batch_size": 64,           # 批量大小
    "hidden_dim": 256,          # 隐藏层维度
    "lr": 0.0005,               # 学习率
    "gamma": 0.99,              # 折扣因子
    "epsilon_start": 1.0,       # 初始探索率
    "epsilon_end": 0.01,        # 最终探索率
    "epsilon_decay": 0.99,      # 探索率衰减
    "target_update": 100,       # 目标网络更新频率
    "buffer_capacity": 20000,   # 经验回放缓冲区大小
    "tmax": 30,                 # 每回合最大步数
}
```

### 在线训练超参数
```python
ONLINE_TRAIN = {
    "num_steps": 1000,          # 训练步数
    "batch_size": 64,
}
```

## 在线训练特点

1. **延迟缓存机制**：缓存查询在每个状态下的延迟，相同状态下复用历史数据，避免重复执行
2. **实际执行**：使用 `EXPLAIN ANALYZE` 实际执行查询，获取真实延迟
3. **分区状态应用**：自动将环境状态应用到数据库
4. **数据库连接管理**：自动处理数据库连接的建立和关闭

