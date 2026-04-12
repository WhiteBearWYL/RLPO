
"""
数据加载工具模块
用于从指定文件夹自动提取数据库schema、表大小和查询信息
"""
import os
import json
import re


def load_schema(folder_path):
    """
    从 schema.txt 加载数据库 schema
    
    参数:
        folder_path: 包含 schema.txt 的文件夹路径
    
    返回:
        表结构字典，格式: {表名: {属性名: 类型}}
    """
    schema_path = os.path.join(folder_path, "schema.txt")
    
    with open(schema_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    # 解析 JSON 格式的 schema
    schema = json.loads(content)
    
    # 转换为小写，保持一致性
    lower_schema = {}
    for table_name, attrs in schema.items():
        lower_table = table_name.lower()
        lower_attrs = {attr.lower(): dtype for attr, dtype in attrs.items()}
        lower_schema[lower_table] = lower_attrs
    
    return lower_schema


def load_table_sizes(folder_path):
    """
    从 row_count.txt 加载表的行数
    
    参数:
        folder_path: 包含 row_count.txt 的文件夹路径
    
    返回:
        表大小字典，格式: {表名: 行数}
    """
    row_count_path = os.path.join(folder_path, "row_count.txt")
    
    with open(row_count_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    # 解析格式: "表名" : "行数", "表名":"行数", ...
    table_sizes = {}
    # 去除引号
    content = content.replace('"', '')
    # 按逗号分割
    pairs = content.split(",")
    
    for pair in pairs:
        pair = pair.strip()
        if not pair:
            continue
        # 按冒号分割
        parts = pair.split(":", 1)
        if len(parts) == 2:
            table_name = parts[0].strip().lower()
            row_count = int(parts[1].strip())
            table_sizes[table_name] = row_count
    
    return table_sizes


def extract_table_aliases(sql):
    """
    从 SQL 查询中提取表别名映射（简化版）
    
    参数:
        sql: SQL 查询字符串
    
    返回:
        表别名映射，格式: {别名: 真实表名}
    """
    alias_map = {}
    
    # TPCH 标准表列表
    tpch_tables = [
        "nation", "region", "part", "supplier", "partsupp",
        "customer", "orders", "lineitem"
    ]
    
    # 简单查找 FROM 子句中的表
    from_match = re.search(r'from\s+(.+?)(?:where|group|order|$)', sql, re.IGNORECASE | re.DOTALL)
    if from_match:
        from_part = from_match.group(1)
        # 移除括号和换行
        from_part = re.sub(r'[\(\)\n]', ' ', from_part)
        
        # 逐个查找 TPCH 表
        for table in tpch_tables:
            # 匹配: table 或 table alias
            pattern = r'\b' + table + r'\b(?:\s+(\w+))?'
            matches = re.findall(pattern, from_part, re.IGNORECASE)
            for alias in matches:
                if alias:
                    alias_map[alias.lower()] = table
                else:
                    alias_map[table] = table
    
    return alias_map


def extract_join_edges_from_query(sql):
    """
    从 SQL 查询中提取连接边（改进版）
    
    参数:
        sql: SQL 查询字符串
    
    返回:
        连接边列表，格式: [ (表1.属性1, 表2.属性2), ... ]
    """
    join_edges = []
    
    # TPCH 属性名到表的映射（基于前缀）
    # c_ -> customer, o_ -> orders, l_ -> lineitem, etc.
    attr_prefix_to_table = {
        'c_': 'customer',
        'o_': 'orders',
        'l_': 'lineitem',
        'p_': 'part',
        'ps_': 'partsupp',
        's_': 'supplier',
        'n_': 'nation',
        'r_': 'region',
    }
    
    # 匹配 WHERE 子句
    where_match = re.search(r'where\s+(.+?)(?:group|order|$)', sql, re.IGNORECASE | re.DOTALL)
    if not where_match:
        return join_edges
    
    where_clause = where_match.group(1)
    
    # 匹配所有的等式条件
    # 格式: a = b，忽略空格
    eq_pattern = r'(\w+(?:\.\w+)?)\s*=\s*(\w+(?:\.\w+)?)'
    matches = re.findall(eq_pattern, where_clause, re.IGNORECASE)
    
    for left, right in matches:
        left = left.lower()
        right = right.lower()
        
        # 解析左右两边
        table1 = None
        attr1 = None
        table2 = None
        attr2 = None
        
        # 左边解析
        if '.' in left:
            # 格式: table.attr
            parts = left.split('.', 1)
            table1_candidate = parts[0]
            attr1 = parts[1]
            # 如果是别名，尝试映射
            alias_map = extract_table_aliases(sql)
            table1 = alias_map.get(table1_candidate, table1_candidate)
        else:
            # 格式: attr（无前缀）
            attr1 = left
            # 尝试从属性前缀推断表
            for prefix, tbl in attr_prefix_to_table.items():
                if attr1.startswith(prefix):
                    table1 = tbl
                    break
        
        # 右边解析
        if '.' in right:
            parts = right.split('.', 1)
            table2_candidate = parts[0]
            attr2 = parts[1]
            alias_map = extract_table_aliases(sql)
            table2 = alias_map.get(table2_candidate, table2_candidate)
        else:
            attr2 = right
            for prefix, tbl in attr_prefix_to_table.items():
                if attr2.startswith(prefix):
                    table2 = tbl
                    break
        
        # 如果成功解析了两个不同的表，添加连接边
        if table1 and table2 and table1 != table2:
            edge1 = table1 + "." + attr1
            edge2 = table2 + "." + attr2
            join_edges.append((edge1, edge2))
    
    return join_edges


def load_queries(folder_path):
    """
    从 train_queries 文件夹加载所有查询
    
    参数:
        folder_path: 包含 train_queries 文件夹的路径
    
    返回:
        查询字典，格式: {查询名: SQL内容}
    """
    queries_folder = os.path.join(folder_path, "train_queries")
    queries = {}
    
    if not os.path.exists(queries_folder):
        print("警告: 查询文件夹不存在:", queries_folder)
        return queries
    
    for filename in os.listdir(queries_folder):
        if filename.endswith(".sql"):
            query_name = filename.replace(".sql", "")
            file_path = os.path.join(queries_folder, filename)
            
            with open(file_path, "r", encoding="utf-8") as f:
                sql = f.read()
            
            queries[query_name] = sql
    
    return queries


def extract_all_join_edges(queries):
    """
    从所有查询中提取所有唯一的连接边
    
    参数:
        queries: 查询字典 {查询名: SQL内容}
    
    返回:
        唯一连接边集合
    """
    all_edges = set()
    
    for query_name, sql in queries.items():
        edges = extract_join_edges_from_query(sql)
        for edge in edges:
            # 归一化边（按字母顺序排序）
            normalized_edge = tuple(sorted(edge))
            all_edges.add(normalized_edge)
    
    return all_edges


def load_dist_key(folder_path):
    """
    加载初始分布键配置（dist_key.txt）
    
    参数:
        folder_path: 数据文件夹路径
    
    返回:
        字典，包含:
            - initial_partition: 初始分区状态 {table: attr}
            - initial_replication: 初始复制状态 {table: bool}
            - replication_allowed: 是否允许复制 {table: bool}
    """
    dist_key_path = os.path.join(folder_path, "dist_key.txt")
    if not os.path.exists(dist_key_path):
        return None
    
    with open(dist_key_path, "r", encoding="utf-8") as f:
        content = f.read()
    
    import json
    data = json.loads(content)
    
    initial_partition = {}
    initial_replication = {}
    replication_allowed = {}
    
    for table_name_upper, info in data.items():
        table_name = table_name_upper.lower()
        key_attr = info.get("KEY")
        if key_attr and key_attr != "NONE":
            initial_partition[table_name] = key_attr.lower()
        
        replicated_str = info.get("REPLICATED", "FALSE")
        initial_replication[table_name] = (replicated_str.upper() == "TRUE")
        
        replicate_able_str = info.get("REPLICATE_ABLE", "TRUE")
        replication_allowed[table_name] = (replicate_able_str.upper() == "TRUE")
    
    return {
        "initial_partition": initial_partition,
        "initial_replication": initial_replication,
        "replication_allowed": replication_allowed,
    }


def build_training_data(folder_path):
    """
    构建完整的训练数据
    
    参数:
        folder_path: 数据文件夹路径
    
    返回:
        完整的训练数据字典，包含:
            - table_attrs: 表属性
            - table_sizes: 表大小
            - queries: 查询字典
            - join_edges: 所有连接边
            - workload_dict: 工作负载字典（简单的频率分配）
            - initial_partition: 初始分区状态
            - initial_replication: 初始复制状态
    """
    print("正在从文件夹加载数据:", folder_path)
    
    # 加载 schema
    schema = load_schema(folder_path)
    print("  - 加载了", len(schema), "个表的 schema")
    
    # 加载表大小
    table_sizes = load_table_sizes(folder_path)
    print("  - 加载了", len(table_sizes), "个表的大小")
    
    # 加载查询
    queries = load_queries(folder_path)
    print("  - 加载了", len(queries), "个查询")
    
    # 提取连接边
    join_edges = extract_all_join_edges(queries)
    print("  - 提取了", len(join_edges), "个唯一连接边")
    
    # 加载初始状态（dist_key.txt）
    initial_state_info = load_dist_key(folder_path)
    if initial_state_info:
        print("  - 加载了初始状态配置")
    
    # 构建工作负载字典（简单起见，所有查询频率相同）
    workload_dict = {}
    for i, query_name in enumerate(sorted(queries.keys())):
        q_key = "Q" + str(i + 1) + ": " + query_name
        workload_dict[q_key] = 10
    
    # 构建 table_attrs（只包含整数类型的属性）
    table_attrs = {}
    for table_name, attrs in schema.items():
        attr_dict = {}
        for attr_name, dtype in attrs.items():
            attr_dict[attr_name] = False
        
        # 从初始状态设置分区键
        if initial_state_info and table_name in initial_state_info["initial_partition"]:
            partition_attr = initial_state_info["initial_partition"][table_name]
            if partition_attr in attr_dict:
                attr_dict[partition_attr] = True
        else:
            # 如果没有初始状态，尝试自动选择
            candidate_keys = []
            for attr_name, dtype in attrs.items():
                is_integer = "INTEGER" in dtype.upper() or "INT" in dtype.upper()
                if is_integer and attr_name.endswith("key"):
                    candidate_keys.append(attr_name)
            
            if candidate_keys:
                primary_key = None
                table_prefix = table_name[0] + "_"
                for key in candidate_keys:
                    if key.startswith(table_prefix):
                        primary_key = key
                        break
                if not primary_key:
                    primary_key = candidate_keys[0]
                attr_dict[primary_key] = True
        
        table_attrs[table_name] = attr_dict
    
    # 构建 replication_allowed（从初始状态或默认）
    if initial_state_info:
        replication_allowed = initial_state_info["replication_allowed"]
    else:
        replication_allowed = {table_name: True for table_name in table_attrs.keys()}
    
    # 构建 join_set
    join_set = set()
    for edge in join_edges:
        join_set.add((edge[0], edge[1], False))
    
    # 目标边 = 所有连接边
    target_edges = list(join_edges)
    
    result = {
        "table_attrs": table_attrs,
        "table_sizes": table_sizes,
        "replication_allowed": replication_allowed,
        "join_set": join_set,
        "queries": queries,
        "workload_dict": workload_dict,
        "target_edges": target_edges,
    }
    
    if initial_state_info:
        result["initial_partition"] = initial_state_info["initial_partition"]
        result["initial_replication"] = initial_state_info["initial_replication"]
    
    return result


def create_simple_reward_fn(target_active_edges):
    """
    简单版本奖励函数
    """
    def reward_fn(state):
        reward = 0.0
        active_edges = set()
        for a1, a2, is_active in state.join_state:
            if is_active:
                active_edges.add((a1, a2))
        for edge in target_active_edges:
            normalized_edge = tuple(sorted(edge))
            if normalized_edge in active_edges:
                reward += 10.0
        reward -= 0.1
        return reward
    return reward_fn


def create_cost_based_reward_fn(table_sizes, workload_edges):
    """
    基于成本模型的奖励函数（离线训练阶段）
    
    成本计算逻辑：
    - 如果连接中的任一表被复制：成本 = 两个表中最大表的行数
    - 如果连接边未激活且没有表被复制：成本 = 表A行数 × 表B行数（嵌套循环连接）
    - 如果连接边已激活且没有表被复制：成本 = 表A行数 + 表B行数（共分区连接）
    
    奖励 = -总成本（因为要最小化成本）
    
    参数：
        table_sizes: 字典，表名 -&gt; 行数
        workload_edges: 列表，工作负载中的连接边 [(表1.属性1, 表2.属性2), ...]
    """
    def reward_fn(state):
        total_cost = 0.0
        
        # 收集激活的边
        active_edges = set()
        for a1, a2, is_active in state.join_state:
            if is_active:
                active_edges.add(tuple(sorted((a1, a2))))
        
        # 对每个工作负载中的连接计算成本
        for edge in workload_edges:
            normalized_edge = tuple(sorted(edge))
            attr1, attr2 = normalized_edge
            
            # 从属性名中提取表名
            table1 = attr1.split('.')[0]
            table2 = attr2.split('.')[0]
            
            # 获取表的行数（如果没有定义，使用默认值）
            size1 = table_sizes.get(table1, 10000)
            size2 = table_sizes.get(table2, 10000)
            
            # 检查是否有表被复制
            is_replicated1 = state.replication_state.get(table1, False)
            is_replicated2 = state.replication_state.get(table2, False)
            
            if is_replicated1 or is_replicated2:
                # 有表被复制：成本 = 两个表中最小表的行数
                cost = max(size1, size2)
            elif normalized_edge in active_edges:
                # 边已激活且无表复制：共分区连接，成本 = 行数之和
                cost = size1 + size2
            else:
                # 边未激活且无表复制：嵌套循环连接，成本 = 行数之积
                cost = size1 * size2
            
            total_cost += cost
        
        # 奖励 = -总成本（因为我们要最小化成本）
        # 添加缩放因子，让奖励值更合理
        reward = -total_cost / 1000000.0
        
        return reward
    
    return reward_fn


if __name__ == "__main__":
    # 简单测试
    folder_path = "text_material/tpch"
    data = build_training_data(folder_path)
    
    print("\n=== 数据加载完成 ===")
    print("\n表属性:")
    for table, attrs in data["table_attrs"].items():
        print("  " + table + ":", attrs)
    
    print("\n表大小:")
    for table, size in data["table_sizes"].items():
        print("  " + table + ":", size)
    
    print("\n连接边:")
    for edge in data["target_edges"]:
        print("  " + edge[0] + " &lt;-&gt; " + edge[1])
    
    print("\n工作负载:")
    for qname, freq in data["workload_dict"].items():
        print("  " + qname + ":", freq)

