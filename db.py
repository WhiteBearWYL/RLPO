import queue
import psycopg2
import psycopg2.extensions
import select
import random
import asyncio
import copy
from typing import List, Tuple, Any, TextIO
from psycopg2 import sql
from config import *
import statistics
import re

def get_explain_analyze_sql(sql: str) -> str:
    """
    Add 'explain analyze' to the given SQL query content.
    
    Args:
        sql : SQL query content
    
    Returns:
        SQL content : SQL query string with 'explain analyze'
    """
    # Check for 'explain analyze'
    expalin_analyze_pos = sql.lower().find('explain analyze')
    if expalin_analyze_pos != -1:
        return  sql
    
    # Check for 'explain'
    explain_pos = sql.lower().find('explain')
    if explain_pos != -1:
        sql_content_with_analyze = sql[:explain_pos + 7] + " analyze" + sql[explain_pos + 7:]  # 7 = len('explain')
        return sql_content_with_analyze
    
    select_pos = sql.lower().find('select')
    if select_pos != -1:
        # Add 'explain analyze' before 'select'
        sql_content_with_explain = sql[:select_pos] + "explain analyze " + sql[select_pos:]
        return sql_content_with_explain
    else:
        return sql

def get_explain_sql(sql: str) -> str:
    """
    Add 'explain' to the given SQL query content.
    
    Args:
        sql : SQL query content
    
    Returns:
        SQL content : SQL query string with 'explain'
    """
    # Check for 'explain'
    explain_pos = sql.lower().find('explain')
    if explain_pos != -1:
        return  sql
    
    select_pos = sql.lower().find('select')
    if select_pos != -1:
        # Add 'explain' before 'select'
        sql_content_with_explain = sql[:select_pos] + "explain " + sql[select_pos:]
        return sql_content_with_explain
    else:
        return sql

def get_db_dist_keys_reset_sqls(now_dist_keys: dict, schema: dict = get_schema(), f: TextIO = None) -> list:
    """
    Generate SQL statements to reset distribution keys for all tables in the database.

    Args:
        now_dist_keys (dict): Current distribution keys of all tables in the database.
        schema (dict): Database schema.

    Returns:
        list: List of SQL statements for resetting distribution keys.
    """
    if not isinstance(now_dist_keys, dict):
        return None
    if not isinstance(schema, dict):
        return None

    reset_sqls = []
    for table_name, dist_key in now_dist_keys.items():
        if table_name.upper() in schema.keys():
            first_column = list(schema[table_name.upper()].keys())[0]  # Take the first column in schema
            if dist_key.upper() != first_column:
                reset_sqls.append(f"ALTER TABLE {table_name} SET DISTRIBUTED BY ({first_column});")
        else:
            print(f"Table {table_name} not found in schema", file=f, flush=True)
    return reset_sqls


def extract_alt_dist_keys(server: dict, solution: dict[str, str]) -> bool:
    """
    Simplify the 'solution' dict based on the current distribution key settings of the target server.

    Args:
        server (dict): Server info with current distribution keys.
        solution (dict[str, str]): Dictionary of desired distribution key changes.

    Returns:
        bool: False if all target keys already match current settings, True otherwise.
    """
    now_dist_keys = server["dist_keys"]
    remove_keys = []
    for key, value in solution.items():
        if now_dist_keys[key.lower()].lower() == value.lower():
            remove_keys.append(key)
    for key in remove_keys:
        del solution[key]
    return len(solution) != 0


def get_similar_template(sql_template: str,
    templates_records: dict[str, dict[str, int | float | list[int]]]) -> str:
    """
    Select the most similar query template to the given SQL template from stored records.

    Args:
        sql_template (str): SQL query template to compare.
        templates_records (dict): Dictionary containing stored template statistics.

    Returns:
        str: The most similar template, or None if none found.
    """
    schema = get_schema()
    row_counts = get_row_counts_by_file()
    join_conditions = extract_join_conditions(sql_template, schema)
    group_attributes = extract_group_by_attributes(sql_template, schema)
    where_conditions = extract_where_conditions(sql_template, schema)
    max_score = 0
    sim_template = ""

    for template, temp_dict in templates_records.items():
        score = 0
        if template == sql_template:
            continue
        for join_condition in join_conditions:
            if join_condition in temp_dict["join_conditions"]:
                join_att1 = join_condition[0].upper()
                row_num1 = next((row_counts[t] for t, attrs in schema.items() if join_att1 in attrs), 0)
                join_att2 = join_condition[1].upper()
                row_num2 = next((row_counts[t] for t, attrs in schema.items() if join_att2 in attrs), 0)
                score += (row_num1 * row_num2)
        for group_attribute in group_attributes:
            if group_attribute in temp_dict["group_attributes"]:
                group_attribute = group_attribute.upper()
                row_num1 = next((row_counts[t] for t, attrs in schema.items() if group_attribute in attrs), 0)
                score += row_num1
        for where_condition in where_conditions:
            if where_condition in temp_dict["where_conditions"]:
                where_condition = where_condition.upper()
                row_num1 = next((row_counts[t] for t, attrs in schema.items() if where_condition in attrs), 0)
                score += row_num1
        if score > max_score:
            max_score = score
            sim_template = template
    return sim_template if max_score > 0 else None


def get_explain_queries(servers: list[dict], query: QueryStruct) -> list[QueryStruct]:
    """
    Generate EXPLAIN query list to obtain query plans under different distribution key configurations.

    Args:
        servers (list[dict]): List of server configurations.
        query (QueryStruct): Original query structure.

    Returns:
        list[QueryStruct]: A list of new QueryStruct objects representing EXPLAIN queries.
    """
    explain_queries = []
    query_worker_id = query.worker_id
    query_dk_id = servers[query_worker_id]["dist_keys_records_idx"]
    tested_dist_keys_idx = []

    if not query.need_store:
        tested_dist_keys_idx.append(query_dk_id)

    for i, server in enumerate(servers):
        if not server["suspend"] and server["dist_keys_records_idx"] not in tested_dist_keys_idx:
            new_query = copy.deepcopy(query)
            new_query.type = QueryTypeEnum.EXPLAIN
            new_query.id *= -1
            new_query.dist_keys_idx = server["dist_keys_records_idx"]
            tested_dist_keys_idx.append(server["dist_keys_records_idx"])
            new_query.sql = get_explain_sql(query.sql)
            new_query.worker_id = i
            explain_queries.append(new_query)
    return explain_queries


def get_copy_query(servers, query, template_records) -> QueryStruct:
    """
    Generate a COPY_SQL type query targeting a server with a different distribution key configuration.

    Args:
        servers (list[dict]): List of server information.
        query (QueryStruct): Query object to duplicate.
        template_records (dict): Template records to track tested distribution keys.

    Returns:
        QueryStruct | None: A new query for testing another distribution key, or None if unavailable.
    """
    diff_dist_keys_workers = get_different_dist_key_workers(servers)
    query_worker_id = query.worker_id
    query_dk_id = servers[query_worker_id]["dist_keys_records_idx"]

    for worker_id in diff_dist_keys_workers:
        if worker_id == query_worker_id:
            continue
        if servers[worker_id]["suspend"]:
            continue
        if servers[worker_id]["dist_keys_records_idx"] == query_dk_id:
            continue
        if servers[worker_id]["dist_keys_records_idx"] in template_records[query.template]["dist_key_tested_idxs"]:
            continue

        new_query = copy.deepcopy(query)
        new_query.worker_id = worker_id
        new_query.dist_keys_idx = servers[worker_id]["dist_keys_records_idx"]
        template_records[query.template]["dist_key_tested_idxs"].append(new_query.dist_keys_idx)
        new_query.type = QueryTypeEnum.COPY_SQL
        new_query.id *= -1
        return new_query
    return None


def deal_the_done_sql_query(
    query: QueryStruct,
    templates_records: dict[str, dict[str, int | float | list[int]]],
    sql_template: str,
    f: TextIO = None
) -> bool:
    """
    Process SQL queries whose state is DONE.

    Args:
        query (QueryStruct): SQL query object containing all metadata.
        templates_records (dict): Template statistics dictionary (see main.py).
        sql_template (str): The query’s SQL template.
        f (TextIO): Log file handle.

    Returns:
        bool: True if an LLM should be called next, False otherwise.
    """
    if query.state != QueryStateEnum.DONE:
        print("Control flow error: Non-DONE SQL query passed to deal_the_done_sql_query", file=f, flush=True)
        return False
    if query.type not in (QueryTypeEnum.SQL, QueryTypeEnum.COPY_SQL):
        print("Control flow error: Non-SQL/COPY_SQL query passed to deal_the_done_sql_query", file=f, flush=True)

    query_cost_time = round(query.end_time - query.start_time2, 2)

    # Update min/max/average execution time statistics
    if query_cost_time < templates_records[sql_template]["min_time"] or templates_records[sql_template]["min_time"] == -1:
        templates_records[sql_template]["min_time"] = query_cost_time
        templates_records[sql_template]["min_time_dist_key_idx"] = query.dist_keys_idx
    if query_cost_time > templates_records[sql_template]["max_time"]:
        templates_records[sql_template]["max_time"] = query_cost_time
        templates_records[sql_template]["max_time_dist_key_idx"] = query.dist_keys_idx

    if query.type == QueryTypeEnum.SQL:
        sum_time = templates_records[sql_template]["avg_time"] * templates_records[sql_template]["counts"]
        sum_time += query_cost_time
        new_avg_time = round(sum_time / (templates_records[sql_template]["counts"] + 1), 2)
        templates_records[sql_template]["avg_time"] = new_avg_time
        templates_records[sql_template]["counts"] += 1

    if query.dist_keys_idx in templates_records[sql_template]["dist_key_times"]:
        before_avg_time = templates_records[sql_template]["dist_key_avg_time"][query.dist_keys_idx]
        before_times = templates_records[sql_template]["dist_key_times"][query.dist_keys_idx]
        templates_records[sql_template]["dist_key_times"][query.dist_keys_idx] += 1
        templates_records[sql_template]["dist_key_avg_time"][query.dist_keys_idx] = round(
            (before_avg_time * before_times + query_cost_time) / (before_times + 1), 2
        )
    else:
        templates_records[sql_template]["dist_key_avg_time"][query.dist_keys_idx] = query_cost_time
        templates_records[sql_template]["dist_key_times"][query.dist_keys_idx] = 1

    query.state = QueryStateEnum.INITIAL
    if templates_records[sql_template]["counts"] == 1:
        return True



def create_async_connection(dsn: str) -> tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
    """
    Create an asynchronous database connection and wait until the handshake is complete.

    Args:
        dsn (str): Data source name string used for establishing the database connection.

    Returns:
        tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor]:
            - conn: The established asynchronous connection object.
            - cur: The cursor object associated with the connection.
    """
    try:
        # Create asynchronous connection
        conn = psycopg2.connect(dsn, async_=True)
        # Wait until the connection is ready
        poll_until_ready(conn)
        # Create cursor
        cur = conn.cursor()
        return conn, cur
    except psycopg2.Error as e:
        print(f"Exception occurred while creating async connection: {e}")
        raise
    except Exception as e:
        print(f"Unexpected exception while creating async connection: {e}")
        raise


def poll_until_ready(conn: psycopg2.extensions.connection) -> None:
    """
    Poll an asynchronous connection until it returns POLL_OK.

    Args:
        conn (psycopg2.extensions.connection): The asynchronous database connection.

    Returns:
        None
    """
    while True:
        state = conn.poll()
        if state == psycopg2.extensions.POLL_OK:
            return
        elif state == psycopg2.extensions.POLL_READ:
            select.select([conn.fileno()], [], [])
        elif state == psycopg2.extensions.POLL_WRITE:
            select.select([], [conn.fileno()], [])
        else:
            raise psycopg2.OperationalError("Polling error")


def sync_execute_sql(
    cur: psycopg2.extensions.cursor,
    sql: str,
    thread_name: str,
    log_file: object
) -> str:
    """
    Execute a SQL query synchronously using the given cursor and return the result.

    Args:
        cur (psycopg2.extensions.cursor): Database cursor object.
        sql (str): SQL query to execute.
        thread_name (str): Name of the thread executing the query.
        log_file (file-like): File handle used to log messages.

    Returns:
        str: "error" if execution failed, otherwise "" (empty string).
    """
    try:
        cur.execute(sql)
        print(f"{thread_name} successfully executed {sql}", file=log_file, flush=True)
        return ""
    except psycopg2.Error as e:
        print(f"{thread_name} failed to execute {sql}: {e}", file=log_file, flush=True)
        return "error"


def dispatch_next_query(conn, cur, sql, conn_status):
    """
    Dispatch the next query to a connection/cursor pair if available,
    otherwise mark the connection as 'done'.

    Args:
        conn: Database connection object.
        cur: Cursor object.
        sql (str): SQL query statement.
        conn_status (dict): Connection status dictionary, marking each
                            connection as 'idle', 'executing', or 'done'.
    """
    try:
        cur.execute("BEGIN;")
        poll_until_ready(conn)
        cur.execute(sql)
        conn_status[conn] = 'executing'
    except Exception as e:
        conn_status[conn] = 'error'
        print(f"Error executing query on connection {conn}: {e}")


def get_all_tables_dist_keys(cursor: psycopg2.extensions.cursor) -> dict[str, str]:
    """
    Retrieve distribution key column names for all tables in a Greenplum database.

    Args:
        cursor (psycopg2.extensions.cursor): Database cursor object.

    Returns:
        dict[str, str]: A mapping from table names to distribution key columns.
                        Returns an empty dict if the query fails.
    """
    try:
        query = """
            SELECT 
                c.relname AS table_name,
                CASE 
                    WHEN p.policytype = 'r' THEN 'REPLICATED'
                    ELSE string_agg(a.attname, ', ' ORDER BY a.attnum)
                END AS dist_key
            FROM pg_class c
            JOIN pg_namespace n ON c.relnamespace = n.oid
            JOIN gp_distribution_policy p ON c.oid = p.localoid
            LEFT JOIN pg_attribute a 
                ON c.oid = a.attrelid 
                AND a.attnum = ANY(p.distkey)
            WHERE n.nspname NOT IN ('pg_catalog', 'information_schema')
            AND c.relkind = 'r'
            GROUP BY n.nspname, c.relname, p.policytype
            ORDER BY n.nspname, c.relname;
        """
        cursor.execute(query)
        results = cursor.fetchall()
        dist_keys = {row[0]: row[1] for row in results}
        return dist_keys
    except Exception as e:
        print(f"Error occurred: {e}")
        return {}


def get_dist_keys(table_name: str, cursor: psycopg2.extensions.cursor) -> list[str]:
    """
    Retrieve distribution key columns for a specific table in Greenplum.

    Args:
        table_name (str): Table name.
        cursor (psycopg2.extensions.cursor): Database cursor.

    Returns:
        list[str]: List of distribution key column names, or empty list on error.
    """
    try:
        query = """
            SELECT a.attname AS distribution_key 
            FROM pg_catalog.pg_class c 
            JOIN pg_catalog.gp_distribution_policy p ON c.oid = p.localoid 
            JOIN pg_catalog.pg_attribute a ON c.oid = a.attrelid AND a.attnum = ANY(p.distkey) 
            WHERE c.relname = %s;
        """
        cursor.execute(query, (table_name,))
        results = cursor.fetchall()
        return [row[0] for row in results]
    except Exception as e:
        print(f"Error occurred: {e}")
        return []

def get_row_counts(table_name: str, cursor: psycopg2.extensions.cursor) -> list[int]:
    """
    Retrieve the row counts per segment for a given table in Greenplum.

    Args:
        table_name (str): The name of the table.
        cursor (psycopg2.extensions.cursor): Database cursor object.

    Returns:
        list[int]: List of row counts on each segment. Returns an empty list on failure.
    """
    try:
        query = f"""
            SELECT count(*) AS row_count
            FROM {table_name}
            GROUP BY gp_segment_id 
            ORDER BY gp_segment_id;
        """
        cursor.execute(query, (table_name,))
        results = cursor.fetchall()
        row_counts = [int(row[0]) for row in results]
        return row_counts
    except Exception as e:
        print(f"Error occurred: {e}")
        return []


def get_database_connection_and_cursor(db_config: dict) -> tuple[psycopg2.extensions.connection, psycopg2.extensions.cursor, str | None]:
    """
    Attempt to connect to the database and return the connection, cursor, and error message.

    Args:
        db_config (dict): Database configuration parameters, e.g.:
            {
                "dbname": "your_database",
                "user": "your_user",
                "password": "your_password",
                "host": "your_host",
                "port": "your_port"
            }

    Returns:
        tuple:
            - connection: psycopg2 connection object if successful, otherwise None.
            - cursor: psycopg2 cursor object if successful, otherwise None.
            - error_message: Error string if failed, None if successful.
    """
    try:
        connection = psycopg2.connect(**db_config)
        cursor = connection.cursor()
        return connection, cursor, None
    except psycopg2.Error as e:
        return None, None, f"Database connection failed: {e}"


def get_distinct_count_and_variance(connection, cursor, table_name: str, column_name: str) -> Tuple[int, float]:
    """
    Retrieve the distinct count and variance of value frequencies for a column in a table.

    Args:
        connection: Database connection.
        cursor: Database cursor.
        table_name (str): Table name.
        column_name (str): Column name.

    Returns:
        Tuple[int, float]: (Distinct value count, variance of frequency distribution).
    """
    query = f"""
        SELECT COUNT(*) AS freq
        FROM {table_name}
        GROUP BY {column_name}
    """
    try:
        cursor.execute(query)
        freqs = [row[0] for row in cursor.fetchall()]
        distinct_count = len(freqs)
        variance = statistics.variance(freqs) if distinct_count > 1 else 0.0
        return distinct_count, round(variance, 2)
    except Exception as e:
        print(f"Failed to retrieve distinct_count: {e}")
        return 0, 0.0

def extract_join_conditions(sql_query: str, schema: dict) -> set:
    """
    Extract JOIN conditions from a SQL query.

    Args:
        sql_query (str): SQL query string.
        schema (dict): Database schema containing table columns.

    Returns:
        set: A set of tuples, each containing two field names used in JOIN conditions.
    """
    
    sql_query = sql_query.lower()
    
    # Extract all column names and store them in a set
    column_names = set()
    for table, columns in schema.items():
        column_names.update(col.lower() for col in columns.keys())
    
    # Extract all equality conditions from WHERE and nested SELECT statements
    join_conditions = set()
    conditions = re.findall(r'\b([a-z_]+)\s*=\s*([a-z_]+)\b', sql_query, re.IGNORECASE)
    
    for col1, col2 in conditions:
        if col1 in column_names and col2 in column_names:
            join_conditions.add(tuple(sorted((col1, col2))))
    
    return join_conditions

def extract_group_by_attributes(sql_query: str, schema: dict) -> set:
    """
    Extract GROUP BY attributes from a SQL query.

    Args:
        sql_query (str): SQL query string.
        schema (dict): Database schema containing table columns.

    Returns:
        set: A set of field names used in GROUP BY.
    """
    sql_query = sql_query.lower()
    
    # Extract all column names from schema and store them in a set
    column_names = set()
    for table, columns in schema.items():
        column_names.update(col.lower() for col in columns.keys())
    
    group_by_match = re.search(r'group\s+by\s+([^;]+?)(\s+having|\s+order\s+by|$)', sql_query, re.IGNORECASE | re.DOTALL)
    if not group_by_match:
        return set()
    
    attributes = set()
    for attr in group_by_match.group(1).split(','):
        attr = attr.strip()
        if attr in column_names:
            attributes.add(attr)
    
    return attributes

def extract_where_conditions(sql_query: str, schema: dict) -> set:
    """
    Extract WHERE filter conditions from a SQL query.

    Args:
        sql_query (str): SQL query string.
        schema (dict): Database schema containing table columns.

    Returns:
        set: A set of field names involved in WHERE filter conditions.
    """
    sql_query = sql_query.lower()
    
    # Extract all column names from schema and store them in a set
    column_names = set()
    for table, columns in schema.items():
        column_names.update(col.lower() for col in columns.keys())
    
    where_match = re.search(r'where\s+([^;]+?)(\s+group\s+by|\s+order\s+by|\s+limit|$)', sql_query, re.IGNORECASE | re.DOTALL)
    if not where_match:
        return set()
    
    join_conditions = extract_join_conditions(sql_query, schema)
    attributes = set()
    
    for condition in re.split(r'\s+and\s+', where_match.group(1)):
        tokens = re.findall(r'\b[a-z_]+\b', condition)  # Extract potential column names
        field_found = False
        condition_fields = set()
        for token in tokens:
            if token in column_names:
                condition_fields.add(token)
                field_found = True
        
        if field_found and tuple(sorted(condition_fields)) not in join_conditions:
            attributes.update(condition_fields)
    
    return attributes

def get_sql_template(sql: str) -> str:
    """
    Transform SQL string by replacing string literals and numbers with placeholders.

    Args:
        sql : Original SQL query string (str)

    Returns:
        transformed_sql : Transformed SQL string (str), 
                          where all string literals are replaced with '$', 
                          and all numbers are replaced with '?'
    """
    # Replace all string literals (enclosed in single quotes) with '$'
    sql = re.sub(r"'[^']*'", "$", sql)
    
    # Replace all numbers with '?' placeholder
    sql = re.sub(r'\d+', '?', sql)
    
    return sql

def parse_plan_line(line: str) -> tuple:
    """
    Parse a single line in a query plan, extracting operator type, cost, rows, width, and extra info.

    Args:
        line : A single line of query plan (str)

    Returns:
        node_type : Operator type (str)
        cost : Cost range (str)
        rows : Number of rows (int)
        width : Data width (int)
        extra_info : Additional information (dict)
    """
    line_stripped = line.strip()

    op_match = re.match(r'^([A-Z][A-Za-z0-9\s:_\-]+)(?=\(|$)', line_stripped)
    node_type = op_match.group(1).strip() if op_match else ''

    cost = None
    rows = None
    width = None

    cost_rows_width_pattern = re.compile(
        r'\(cost=([\d\.]+..[\d\.]+)\s+rows=(\d+)\s+width=(\d+)\)'
    )
    crw_match = cost_rows_width_pattern.search(line_stripped)
    if crw_match:
        cost = crw_match.group(1)
        rows = int(crw_match.group(2))
        width = int(crw_match.group(3))

    extra_info = {}
    other_parentheses = re.findall(r'\(([^()]*?)\)', line_stripped)
    for item in other_parentheses:
        if "cost=" in item or "rows=" in item or "width=" in item:
            continue
        parts = [p.strip() for p in item.split(';')]
        for part in parts:
            kv = part.split(':')
            if len(kv) == 2:
                extra_info[kv[0].strip()] = kv[1].strip()
            elif 'slice' in part.lower():
                extra_info['slice'] = part.replace('slice', '').strip()
    return node_type, cost, rows, width, extra_info

def parse_anal_line(line: str) -> tuple:
    """
    Parse a single line in an EXPLAIN ANALYZE plan, extracting operator type, actual time range, rows, cost range, and extra info.

    Args:
        line : A single line of query plan (str)

    Returns:
        node_type : Operator type (str)
        actual_time : Actual execution time range (str)
        rows : Actual number of rows (int)
        cost : Estimated cost range (str)
        extra_info : Additional information (dict)
    """
    line_stripped = line.strip()

    op_match = re.match(r'^([A-Z][A-Za-z0-9\s:_\-]+)(?=\(|$)', line_stripped)
    node_type = op_match.group(1).strip() if op_match else ''
    
    cost_rows_width_pattern = re.compile(
        r'\(cost=([\d\.]+..[\d\.]+)\s+rows=(\d+)\s+width=(\d+)\)'
    )
    cost = None
    crw_match = cost_rows_width_pattern.search(line_stripped)
    if crw_match:
        cost = crw_match.group(1)
    
    actual_time = None
    rows = None
    loops = None
    time_rows_loops_pattern = re.compile(
        r'\(actual time=([\d\.]+..[\d\.]+)\s+rows=(\d+)\s+loops=(\d+)\)'
    )
    crw_match = time_rows_loops_pattern.search(line_stripped)
    if crw_match:
        actual_time = crw_match.group(1)
        rows = int(crw_match.group(2))
        loops = int(crw_match.group(3))

    extra_info = {}
    other_parentheses = re.findall(r'\(([^()]*?)\)', line_stripped)
    for item in other_parentheses:
        if "actual time=" in item or "rows=" in item :
            continue
        parts = [p.strip() for p in item.split(';')]
        for part in parts:
            kv = part.split(':')
            if len(kv) == 2:
                extra_info[kv[0].strip()] = kv[1].strip()
            elif 'slice' in part.lower():
                extra_info['slice'] = part.replace('slice', '').strip()
    return node_type, actual_time, rows, cost, extra_info

def parse_plan_attributes(line: str) -> dict:
    """
    Parse attribute lines in a query plan.

    Args:
        line : A single line of query plan (str)

    Returns:
        attributes : Attribute dictionary (dict)
    """
    line_stripped = line.strip()
    pattern_keys = [
        'Merge Key', 'Sort Key', 'Hash Cond', 'Join Filter', 'Filter', 'Hash Key',
        'Group Key', 'Planned Partitions'
    ]
    for key in pattern_keys:
        if line_stripped.startswith(key + ':'):
            _, val = line_stripped.split(':', 1)
            return {key: val.strip()}
    return {}

def build_plan_tree(plan_lines: list) -> list:
    """
    Build a tree structure of the query plan.

    Args:
        plan_lines : List of plan lines, each element being a string or single-element tuple (list)

    Returns:
        root_nodes : The built tree structure containing all root nodes (list)
        
    Example:
        plan_tree = build_plan_tree(plan_lines)
        print("Starting serialization of Plan Tree...")
        serialized_tree = json.dumps(plan_tree, indent=2, ensure_ascii=False)
        print("Serialization completed.")
    """
    root_nodes = []
    stack = []

    for raw_line in plan_lines:
        if isinstance(raw_line, tuple):
            raw_line = raw_line[0] if raw_line else ""

        if not raw_line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(' '))
        line_stripped = raw_line.strip()

        is_new_operator = ('->' in line_stripped) or re.search(r'\(cost=[\d\.]', line_stripped)

        if is_new_operator:
            line_without_arrow = line_stripped.replace('->', '').strip()
            node_type, cost, rows, width, extra_info = parse_plan_line(line_without_arrow)

            node = {
                'node_type': node_type or '', 
                'cost': cost,
                'rows': rows,
                # 'width': width,
                'attributes': dict(extra_info),
                'children': []
            }

            while stack and stack[-1][1] >= indent:
                stack.pop()

            if stack:
                stack[-1][0]['children'].append(node)
            else:
                root_nodes.append(node)

            stack.append((node, indent))
        else:
            attr_dict = parse_plan_attributes(line_stripped)
            if attr_dict:
                if stack:
                    stack[-1][0]['attributes'].update(attr_dict)
            elif stack:
                top_node = stack[-1][0]
                if 'extra_lines' not in top_node['attributes']:
                    top_node['attributes']['extra_lines'] = []
                top_node['attributes']['extra_lines'].append(line_stripped)

    return root_nodes

def build_anal_tree(anal_lines: list) -> list:
    """
    Build a tree structure of an EXPLAIN ANALYZE query plan.

    Args:
        anal_lines : List of EXPLAIN ANALYZE output lines, 
            each element being a string or single-element tuple (list)

    Returns:
        root_nodes : The built tree structure containing all root nodes (list)
    """
    root_nodes = []
    stack = []

    for raw_line in anal_lines:
        if isinstance(raw_line, tuple):
            raw_line = raw_line[0] if raw_line else ""

        if not raw_line.strip():
            continue

        indent = len(raw_line) - len(raw_line.lstrip(' '))
        line_stripped = raw_line.strip()

        is_new_operator = ('->' in line_stripped) or re.search(r'\(actual time=[\d\.]', line_stripped)

        if is_new_operator:
            line_without_arrow = line_stripped.replace('->', '').strip()
            node_type, actual_time, rows, cost, extra_info = parse_anal_line(line_without_arrow)

            node = {
                'node_type': node_type or '', 
                'actual_time': actual_time,
                'rows': rows,
                'cost': cost,
                'attributes': dict(extra_info),
                'children': []
            }

            while stack and stack[-1][1] >= indent:
                stack.pop()

            if stack:
                stack[-1][0]['children'].append(node)
            else:
                root_nodes.append(node)

            stack.append((node, indent))
        else:
            attr_dict = parse_plan_attributes(line_stripped)
            if attr_dict:
                if stack:
                    stack[-1][0]['attributes'].update(attr_dict)
            elif stack:
                top_node = stack[-1][0]
                if 'extra_lines' not in top_node['attributes']:
                    top_node['attributes']['extra_lines'] = []
                top_node['attributes']['extra_lines'].append(line_stripped)

    return root_nodes
