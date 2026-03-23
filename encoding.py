from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple, Set, Optional, Any
import numpy as np


QualifiedAttr = str                  # e.g. "orders.customer_id"
JoinTriple = Tuple[QualifiedAttr, QualifiedAttr, bool]


class EncodingError(ValueError):
    pass


def _split_qualified_attr(attr: str) -> Tuple[str, str]:
    """
    将全限定属性名 'table.attr' 拆成 (table, attr)。
    """
    if "." not in attr:
        raise EncodingError(
            f"属性 '{attr}' 不是全限定名，请使用 'table.attr' 格式。"
        )
    table, col = attr.split(".", 1)
    if not table or not col:
        raise EncodingError(f"非法全限定属性名: '{attr}'")
    return table, col


def _normalize_edge_key(a1: QualifiedAttr, a2: QualifiedAttr) -> Tuple[QualifiedAttr, QualifiedAttr]:
    """
    将无向 join edge 规范化为唯一顺序的二元组。
    例如:
        ('orders.customer_id', 'customer.customer_id')
    与
        ('customer.customer_id', 'orders.customer_id')
    会得到同一个 key。
    """
    _split_qualified_attr(a1)
    _split_qualified_attr(a2)

    if a1 == a2:
        raise EncodingError(f"join edge 两端属性不能相同: '{a1}'")

    return tuple(sorted((a1, a2)))


@dataclass(frozen=True)
class EncodingCatalog:
    """
    固定编码目录。
    只要 catalog 不变，向量的每一维语义和位置都不变。
    """
    table_order: List[str]
    qualified_attr_order: List[QualifiedAttr]
    join_edge_order: List[Tuple[QualifiedAttr, QualifiedAttr]]
    workload_order: List[str]

    # 子向量内部索引
    attr_to_idx: Dict[QualifiedAttr, int]
    table_to_rep_idx: Dict[str, int]
    edge_to_idx: Dict[Tuple[QualifiedAttr, QualifiedAttr], int]
    query_to_idx: Dict[str, int]

    # 全向量维度切片
    rep_offset: int
    part_offset: int
    edge_offset: int
    workload_offset: int

    num_replication_bits: int
    num_partition_bits: int
    num_edge_bits: int
    num_workload_bits: int

    # 全向量 index -> label
    index_to_label: List[str]

    @property
    def state_dim(self) -> int:
        return (
            self.num_replication_bits
            + self.num_partition_bits
            + self.num_edge_bits
            + self.num_workload_bits
        )


@dataclass
class EncodedState:
    vector: np.ndarray
    replication_vec: np.ndarray
    partition_vec: np.ndarray
    edge_vec: np.ndarray
    workload_vec: np.ndarray


class StateEncoder:
    """
    状态编码模块。

    state = [replication bits | partition bits | edge bits | workload vector]

    约束：
    1. 每张表最多只有一个分区键属性为 True
    2. join_set 中第三项 bool 表示该 edge 当前是否激活
    3. join 属性必须是全限定名 table.attr
    """

    def __init__(
        self,
        table_attrs: Dict[str, Dict[str, bool]],
        replication_allowed: Dict[str, bool],
        join_set: Set[JoinTriple],
        workload_dict: Dict[str, int],
        normalize_workload: bool = True,
    ) -> None:
        self.table_attrs = table_attrs
        self.replication_allowed = replication_allowed
        self.join_set = join_set
        self.workload_dict = workload_dict
        self.normalize_workload = normalize_workload

        self._validate_inputs()
        self.catalog = self._build_catalog()

    # =========================
    # validation
    # =========================

    def _validate_inputs(self) -> None:
        # 1. replication_allowed 和 table_attrs 表集合对齐
        table_names = set(self.table_attrs.keys())
        rep_table_names = set(self.replication_allowed.keys())

        missing_tables = table_names - rep_table_names
        extra_tables = rep_table_names - table_names

        if missing_tables:
            raise EncodingError(f"replication_allowed 缺少这些表: {sorted(missing_tables)}")
        if extra_tables:
            raise EncodingError(f"replication_allowed 包含未知表: {sorted(extra_tables)}")

        # 2. 检查表属性，并强制每表最多一个分区键
        known_qualified_attrs = set()

        for table, attrs in self.table_attrs.items():
            if not attrs:
                raise EncodingError(f"表 '{table}' 没有属性定义。")

            true_count = 0
            for attr, is_partition_key in attrs.items():
                if not isinstance(is_partition_key, bool):
                    raise EncodingError(
                        f"表 '{table}' 的属性 '{attr}' 的值必须是 bool。"
                    )
                if is_partition_key:
                    true_count += 1
                known_qualified_attrs.add(f"{table}.{attr}")

            if true_count > 1:
                raise EncodingError(
                    f"表 '{table}' 有多个分区键为 True。"
                    f" 每张表最多只能有一个分区键。"
                )

        # 3. 检查 join_set
        normalized_edges = set()
        for left_attr, right_attr, is_active in self.join_set:
            if left_attr not in known_qualified_attrs:
                raise EncodingError(f"join_set 中属性不存在: '{left_attr}'")
            if right_attr not in known_qualified_attrs:
                raise EncodingError(f"join_set 中属性不存在: '{right_attr}'")
            if not isinstance(is_active, bool):
                raise EncodingError(
                    f"连接 ({left_attr}, {right_attr}, {is_active}) 的第三项必须是 bool。"
                )

            edge_key = _normalize_edge_key(left_attr, right_attr)
            if edge_key in normalized_edges:
                raise EncodingError(f"join_set 中存在重复边: {edge_key}")
            normalized_edges.add(edge_key)

        # 4. 检查 workload
        for q, cnt in self.workload_dict.items():
            if not isinstance(q, str) or not q:
                raise EncodingError("workload 中查询模板 key 必须是非空字符串。")
            if not isinstance(cnt, int) or cnt < 0:
                raise EncodingError(f"workload 中查询 '{q}' 的频次必须是非负整数。")

    # =========================
    # catalog
    # =========================

    def _build_catalog(self) -> EncodingCatalog:
        table_order = sorted(self.table_attrs.keys())

        qualified_attr_order: List[QualifiedAttr] = []
        for table in table_order:
            for attr in sorted(self.table_attrs[table].keys()):
                qualified_attr_order.append(f"{table}.{attr}")

        join_edge_order = sorted(
            {_normalize_edge_key(a1, a2) for a1, a2, _ in self.join_set}
        )

        workload_order = sorted(self.workload_dict.keys())

        attr_to_idx = {attr: i for i, attr in enumerate(qualified_attr_order)}
        table_to_rep_idx = {table: i for i, table in enumerate(table_order)}
        edge_to_idx = {edge: i for i, edge in enumerate(join_edge_order)}
        query_to_idx = {q: i for i, q in enumerate(workload_order)}

        num_replication_bits = len(table_order)
        num_partition_bits = len(qualified_attr_order)
        num_edge_bits = len(join_edge_order)
        num_workload_bits = len(workload_order)

        rep_offset = 0
        part_offset = rep_offset + num_replication_bits
        edge_offset = part_offset + num_partition_bits
        workload_offset = edge_offset + num_edge_bits

        index_to_label: List[str] = []

        # replication section
        for table in table_order:
            index_to_label.append(f"replicate_allow[{table}]")

        # partition section
        for qattr in qualified_attr_order:
            index_to_label.append(f"partkey_using[{qattr}]")

        # edge section
        for a1, a2 in join_edge_order:
            index_to_label.append(f"edge_using[{a1}={a2}]")

        # workload section
        for q in workload_order:
            index_to_label.append(f"query_frequency[{q}]")

        return EncodingCatalog(
            table_order=table_order,
            qualified_attr_order=qualified_attr_order,
            join_edge_order=join_edge_order,
            workload_order=workload_order,
            attr_to_idx=attr_to_idx,
            table_to_rep_idx=table_to_rep_idx,
            edge_to_idx=edge_to_idx,
            query_to_idx=query_to_idx,
            rep_offset=rep_offset,
            part_offset=part_offset,
            edge_offset=edge_offset,
            workload_offset=workload_offset,
            num_replication_bits=num_replication_bits,
            num_partition_bits=num_partition_bits,
            num_edge_bits=num_edge_bits,
            num_workload_bits=num_workload_bits,
            index_to_label=index_to_label,
        )

    # =========================
    # encoding helpers
    # =========================

    def encode_replication_state(
        self,
        current_replication_state: Optional[Dict[str, bool]] = None,
    ) -> np.ndarray:
        """
        key = table
        value = 当前该表是否被复制
        若不传，则默认全 False
        """
        vec = np.zeros(self.catalog.num_replication_bits, dtype=np.float32)

        if current_replication_state is None:
            current_replication_state = {}

        for table in self.catalog.table_order:
            idx = self.catalog.table_to_rep_idx[table]
            is_replicated = bool(current_replication_state.get(table, False))

            if is_replicated and not self.replication_allowed[table]:
                raise EncodingError(f"表 '{table}' 不允许复制，但给定状态中为 True。")

            vec[idx] = 1.0 if is_replicated else 0.0

        return vec

    def encode_partition_state(
        self,
        current_partition_state: Optional[Dict[str, Dict[str, bool]]] = None,
    ) -> np.ndarray:
        """
        key = table
        value = {attr -> bool}
        若不传，则使用初始化时的 table_attrs 中定义的状态。
        """
        vec = np.zeros(self.catalog.num_partition_bits, dtype=np.float32)

        source = current_partition_state if current_partition_state is not None else self.table_attrs

        for table in self.catalog.table_order:
            if table not in source:
                raise EncodingError(f"current_partition_state 缺少表 '{table}'")
            for attr in self.table_attrs[table].keys():
                if attr not in source[table]:
                    raise EncodingError(f"current_partition_state 缺少属性 '{table}.{attr}'")

            true_count = sum(1 for v in source[table].values() if bool(v))
            if true_count > 1:
                raise EncodingError(
                    f"表 '{table}' 在 current_partition_state 中有多个分区键为 True。"
                )

        for qattr in self.catalog.qualified_attr_order:
            table, attr = _split_qualified_attr(qattr)
            idx = self.catalog.attr_to_idx[qattr]
            vec[idx] = 1.0 if bool(source[table][attr]) else 0.0

        return vec

    def encode_edge_state(
        self,
        current_join_state: Optional[Set[JoinTriple]] = None,
    ) -> np.ndarray:
        """
        current_join_state 中每个元素:
            (left_attr, right_attr, is_active)

        第三项表示该 edge 当前是否激活。
        若不传，则使用初始化时 join_set。
        """
        vec = np.zeros(self.catalog.num_edge_bits, dtype=np.float32)

        source = current_join_state if current_join_state is not None else self.join_set

        seen = set()
        for a1, a2, is_active in source:
            key = _normalize_edge_key(a1, a2)

            if key not in self.catalog.edge_to_idx:
                raise EncodingError(f"未知 join edge: {key}")
            if key in seen:
                raise EncodingError(f"current_join_state 中存在重复边: {key}")
            seen.add(key)

            idx = self.catalog.edge_to_idx[key]
            vec[idx] = 1.0 if bool(is_active) else 0.0

        return vec

    def encode_workload(
        self,
        workload_dict: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        """
        按固定顺序将 workload 编码为频率向量。
        缺失查询视为 0。
        第一版要求查询维度固定，不允许出现 catalog 外的新查询。
        """
        source = workload_dict if workload_dict is not None else self.workload_dict
        vec = np.zeros(self.catalog.num_workload_bits, dtype=np.float32)

        for q in source:
            if q not in self.catalog.query_to_idx:
                raise EncodingError(
                    f"encode_workload 收到了 catalog 中不存在的新查询模板: '{q}'"
                )

        for q in self.catalog.workload_order:
            idx = self.catalog.query_to_idx[q]
            vec[idx] = float(source.get(q, 0))

        if self.normalize_workload:
            total = float(vec.sum())
            if total > 0.0:
                vec = vec / total

        return vec

    # =========================
    # full state
    # =========================

    def encode_state(
        self,
        current_replication_state: Optional[Dict[str, bool]] = None,
        current_partition_state: Optional[Dict[str, Dict[str, bool]]] = None,
        current_join_state: Optional[Set[JoinTriple]] = None,
        workload_dict: Optional[Dict[str, int]] = None,
    ) -> EncodedState:
        rep_vec = self.encode_replication_state(current_replication_state)
        part_vec = self.encode_partition_state(current_partition_state)
        edge_vec = self.encode_edge_state(current_join_state)
        workload_vec = self.encode_workload(workload_dict)

        full = np.concatenate([rep_vec, part_vec, edge_vec, workload_vec], axis=0)

        return EncodedState(
            vector=full,
            replication_vec=rep_vec,
            partition_vec=part_vec,
            edge_vec=edge_vec,
            workload_vec=workload_vec,
        )

    # =========================
    # layout / explain
    # =========================

    def describe_layout(self, verbose: bool = True) -> Dict[str, Any]:
        """
        返回编码布局信息。
        verbose=True 时会给出每个 index 的明确语义。
        """
        c = self.catalog
        layout = {
            "state_dim": c.state_dim,
            "sections": {
                "replication": {
                    "start": c.rep_offset,
                    "end": c.part_offset,
                    "size": c.num_replication_bits,
                },
                "partition": {
                    "start": c.part_offset,
                    "end": c.edge_offset,
                    "size": c.num_partition_bits,
                },
                "edge": {
                    "start": c.edge_offset,
                    "end": c.workload_offset,
                    "size": c.num_edge_bits,
                },
                "workload": {
                    "start": c.workload_offset,
                    "end": c.state_dim,
                    "size": c.num_workload_bits,
                },
            },
        }

        if verbose:
            layout["index_mapping"] = [
                {"index": i, "label": label}
                for i, label in enumerate(c.index_to_label)
            ]

        return layout

    def explain_vector(self, vector: np.ndarray) -> List[Tuple[int, str, float]]:
        """
        返回 [(index, label, value), ...]
        """
        if len(vector) != self.catalog.state_dim:
            raise EncodingError(
                f"向量长度不匹配: got={len(vector)}, expected={self.catalog.state_dim}"
            )

        return [
            (i, self.catalog.index_to_label[i], float(vector[i]))
            for i in range(len(vector))
        ]

    def explain_nonzero(
        self,
        vector: np.ndarray,
        threshold: float = 1e-8,
    ) -> List[Tuple[int, str, float]]:
        """
        只返回非零项，便于调试。
        """
        explained = self.explain_vector(vector)
        return [
            (idx, label, value)
            for idx, label, value in explained
            if abs(value) > threshold
        ]

    def pretty_print_vector(
        self,
        vector: np.ndarray,
        threshold: float = -1.0,
    ) -> None:
        """
        以可读形式打印向量。
        threshold > 0 时只打印 abs(value) > threshold 的项。
        """
        rows = self.explain_vector(vector)
        for idx, label, value in rows:
            if threshold > 0 and abs(value) <= threshold:
                continue
            print(f"{idx:4d} | {label:60s} | {value:.6f}")