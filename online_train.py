
"""
在线训练模块
包含查询延迟缓存机制、数据库操作和实际执行奖励计算
"""
import os
import time
import hashlib
import psycopg2
import json
import copy

from config import (
    DBNAME,
    DB_USER,
    DB_PASSWD,
    IPS,
    PORT,
)

from db import (
    get_database_connection_and_cursor,
    get_explain_analyze_sql,
    get_all_tables_dist_keys,
)


class QueryLatencyCache:
    """
    查询延迟缓存类
    用于缓存不同数据库状态下查询的执行延迟，避免重复执行
    """
    def __init__(self, cache_file="query_latency_cache.json"):
        self.cache_file = cache_file
        self.cache = {}
        self.load_cache()
    
    def load_cache(self):
        """从文件加载缓存"""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception as e:
                print(f"加载缓存失败: {e}")
                self.cache = {}
    
    def save_cache(self):
        """保存缓存到文件"""
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存缓存失败: {e}")
    
    def _state_to_key(self, partition_state, replication_state):
        """
        将数据库状态转换为缓存键
        对每个状态生成唯一的哈希值
        """
        state_dict = {
            "partition": copy.deepcopy(partition_state),
            "replication": copy.deepcopy(replication_state),
        }
        state_str = json.dumps(state_dict, sort_keys=True)
        return hashlib.md5(state_str.encode("utf-8")).hexdigest()
    
    def get_query_latency(self, query_name, partition_state, replication_state):
        """
        获取查询在特定状态下的延迟
        
        返回:
            - 如果缓存命中，返回缓存的延迟
            - 如果缓存未命中，返回 None
        """
        state_key = self._state_to_key(partition_state, replication_state)
        if state_key in self.cache:
            if query_name in self.cache[state_key]:
                return self.cache[state_key][query_name]
        return None
    
    def set_query_latency(self, query_name, partition_state, replication_state, latency):
        """设置查询在特定状态下的延迟"""
        state_key = self._state_to_key(partition_state, replication_state)
        if state_key not in self.cache:
            self.cache[state_key] = {}
        self.cache[state_key][query_name] = latency
        self.save_cache()
    
    def clear_cache(self):
        """清空缓存"""
        self.cache = {}
        self.save_cache()


class DatabaseManager:
    """
    数据库管理器
    负责处理数据库连接、分区键设置和查询执行
    """
    def __init__(self, host_idx=0):
        self.host = IPS[host_idx]
        self.db_config = {
            "dbname": DBNAME,
            "user": DB_USER,
            "password": DB_PASSWD,
            "host": self.host,
            "port": PORT,
        }
        self.connection = None
        self.cursor = None
        self._connect()
    
    def _connect(self):
        """建立数据库连接"""
        conn, cur, err = get_database_connection_and_cursor(self.db_config)
        if err:
            raise Exception(f"数据库连接失败: {err}")
        self.connection = conn
        self.cursor = cur
        print(f"成功连接到数据库: {self.host}:{PORT}/{DBNAME}")
    
    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
        print("数据库连接已关闭")
    
    def get_current_dist_keys(self):
        """获取当前所有表的分布键"""
        return get_all_tables_dist_keys(self.cursor)
    
    def _get_alter_distributed_sql(self, table_name, dist_key, is_replicated=False):
        """
        生成 ALTER TABLE ... SET DISTRIBUTED BY 语句
        
        如果 dist_key 为 None 或 is_replicated 为 True，则生成复制表语句
        """
        if is_replicated or dist_key is None:
            return f"ALTER TABLE {table_name} SET DISTRIBUTED REPLICATED;"
        else:
            return f"ALTER TABLE {table_name} SET DISTRIBUTED BY ({dist_key});"
    
    def apply_partition_state(self, partition_state, replication_state):
        """
        将指定的分区状态应用到数据库
        
        参数:
            partition_state: 字典 {表名: 分区键}
            replication_state: 字典 {表名: 是否复制}
        """
        print("\n正在应用分区状态到数据库...")
        for table_name in partition_state.keys():
            dist_key = partition_state.get(table_name)
            is_replicated = replication_state.get(table_name, False)
            
            try:
                sql = self._get_alter_distributed_sql(table_name, dist_key, is_replicated)
                print(f"  执行: {sql}")
                self.cursor.execute(sql)
                self.connection.commit()
            except Exception as e:
                print(f"  警告: 无法设置表 {table_name} 的分布键: {e}")
                self.connection.rollback()
        print("分区状态应用完成!\n")
    
    def execute_query_with_analyze(self, query_sql):
        """
        使用 EXPLAIN ANALYZE 执行查询并获取实际执行时间
        
        返回:
            - 执行时间（秒）
        """
        try:
            explain_sql = get_explain_analyze_sql(query_sql)
            
            start_time = time.time()
            self.cursor.execute(explain_sql)
            self.cursor.fetchall()
            end_time = time.time()
            
            return end_time - start_time
        except Exception as e:
            print(f"  查询执行失败: {e}")
            self.connection.rollback()
            return None


class OnlineTrainRewardCalculator:
    """
    在线训练奖励计算器
    通过实际执行查询并测量延迟来计算奖励
    """
    def __init__(self, db_manager, latency_cache, queries):
        self.db_manager = db_manager
        self.latency_cache = latency_cache
        self.queries = queries
    
    def get_partition_and_replication_from_state(self, state):
        """从环境状态中提取分区和复制状态"""
        partition_state = {}
        for table, attrs in state.partition_state.items():
            chosen_attr = None
            for attr, is_set in attrs.items():
                if is_set:
                    chosen_attr = attr
                    break
            partition_state[table] = chosen_attr
        
        replication_state = copy.deepcopy(state.replication_state)
        return partition_state, replication_state
    
    def calculate_total_latency(self, state):
        """
        计算所有查询在当前状态下的总延迟
        
        参数:
            state: 环境状态对象
        
        返回:
            总延迟（秒）
        """
        partition_state, replication_state = self.get_partition_and_replication_from_state(state)
        
        total_latency = 0.0
        queries_executed = 0
        queries_cached = 0
        
        print("\n正在计算查询延迟...")
        
        for query_name, query_sql in self.queries.items():
            cached_latency = self.latency_cache.get_query_latency(
                query_name, partition_state, replication_state
            )
            
            if cached_latency is not None:
                queries_cached += 1
                total_latency += cached_latency
            else:
                queries_executed += 1
                print(f"  执行查询: {query_name}")
                
                self.db_manager.apply_partition_state(partition_state, replication_state)
                
                latency = self.db_manager.execute_query_with_analyze(query_sql)
                
                if latency is not None:
                    self.latency_cache.set_query_latency(
                        query_name, partition_state, replication_state, latency
                    )
                    total_latency += latency
        
        print(f"查询延迟计算完成! 缓存命中: {queries_cached}, 新执行: {queries_executed}")
        print(f"总延迟: {total_latency:.2f} 秒\n")
        
        return total_latency
    
    def get_reward(self, state):
        """
        获取当前状态的奖励
        
        奖励 = -总延迟（最小化延迟等同于最大化奖励）
        """
        total_latency = self.calculate_total_latency(state)
        reward = -total_latency
        return reward


def create_online_train_reward_fn(db_manager, latency_cache, queries):
    """
    创建在线训练奖励函数
    
    这是一个工厂函数，返回一个可以直接传入 PartitioningEnv 的奖励函数
    """
    calculator = OnlineTrainRewardCalculator(db_manager, latency_cache, queries)
    
    def reward_fn(state):
        return calculator.get_reward(state)
    
    return reward_fn, calculator

