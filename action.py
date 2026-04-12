
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import numpy as np

from encoding import EncodingError, EncodingCatalog, _normalize_edge_key


QualifiedAttr = str
EdgeKey = Tuple[QualifiedAttr, QualifiedAttr]


class ActionType(str, Enum):
    PARTITION_TABLE = "PARTITION_TABLE"
    REPLICATE_TABLE = "REPLICATE_TABLE"
    ACTIVATE_EDGE = "ACTIVATE_EDGE"
    DEACTIVATE_EDGE = "DEACTIVATE_EDGE"


@dataclass(frozen=True)
class Action:
    """
    单步动作定义。
    不负责状态转移，只描述“要做什么”。
    """
    action_type: ActionType
    table: Optional[str] = None
    attr: Optional[QualifiedAttr] = None
    edge: Optional[EdgeKey] = None

    def __post_init__(self) -> None:
        # 基本字段一致性检查
        if self.action_type == ActionType.PARTITION_TABLE:
            if self.table is None or self.attr is None:
                raise EncodingError("PARTITION_TABLE 动作必须提供 table 和 attr。")
            if self.edge is not None:
                raise EncodingError("PARTITION_TABLE 动作不应包含 edge。")

        elif self.action_type == ActionType.REPLICATE_TABLE:
            if self.table is None:
                raise EncodingError("REPLICATE_TABLE 动作必须提供 table。")
            if self.attr is not None or self.edge is not None:
                raise EncodingError("REPLICATE_TABLE 动作不应包含 attr 或 edge。")

        elif self.action_type in (ActionType.ACTIVATE_EDGE, ActionType.DEACTIVATE_EDGE):
            if self.edge is None:
                raise EncodingError(f"{self.action_type} 动作必须提供 edge。")
            if self.table is not None or self.attr is not None:
                raise EncodingError(f"{self.action_type} 动作不应包含 table 或 attr。")

        else:
            raise EncodingError(f"未知动作类型: {self.action_type}")

    def to_readable_dict(self) -> Dict[str, Any]:
        return {
            "action_type": self.action_type.value,
            "table": self.table,
            "attr": self.attr,
            "edge": self.edge,
        }

    def __str__(self) -> str:
        if self.action_type == ActionType.PARTITION_TABLE:
            return f"PartitionTable(table={self.table}, attr={self.attr})"
        if self.action_type == ActionType.REPLICATE_TABLE:
            return f"ReplicateTable(table={self.table})"
        if self.action_type == ActionType.ACTIVATE_EDGE:
            return f"ActivateEdge(edge={self.edge})"
        if self.action_type == ActionType.DEACTIVATE_EDGE:
            return f"DeactivateEdge(edge={self.edge})"
        return self.to_readable_dict().__str__()


@dataclass(frozen=True)
class ActionCatalog:
    """
    固定动作编码目录。
    向量结构:
        [action_type | target_table | target_attr | target_edge]
    """
    action_type_order: List[ActionType]
    table_order: List[str]
    qualified_attr_order: List[QualifiedAttr]
    edge_order: List[EdgeKey]

    action_type_to_idx: Dict[ActionType, int]
    table_to_idx: Dict[str, int]
    attr_to_idx: Dict[QualifiedAttr, int]
    edge_to_idx: Dict[EdgeKey, int]

    type_offset: int
    table_offset: int
    attr_offset: int
    edge_offset: int

    index_to_label: List[str]

    @property
    def action_dim(self) -> int:
        return (
            len(self.action_type_order)
            + len(self.table_order)
            + len(self.qualified_attr_order)
            + len(self.edge_order)
        )


class ActionEncoder:
    """
    动作编码器。
    负责：
    1. 基于 encoding catalog 构造动作目录
    2. 将 Action 编码为固定长度向量
    3. 枚举“静态候选动作”

    不负责：
    - 动作是否合法（依赖当前 state 的部分）
    - 执行动作后的状态转移
    """

    def __init__(self, encoding_catalog: EncodingCatalog, replication_allowed: Dict[str, bool]) -> None:
        self.ec = encoding_catalog
        self.replication_allowed = replication_allowed
        self.catalog = self._build_catalog()

    def _build_catalog(self) -> ActionCatalog:
        action_type_order = [
            ActionType.PARTITION_TABLE,
            ActionType.REPLICATE_TABLE,
            ActionType.ACTIVATE_EDGE,
            ActionType.DEACTIVATE_EDGE,
        ]

        table_order = list(self.ec.table_order)
        qualified_attr_order = list(self.ec.qualified_attr_order)
        edge_order = list(self.ec.join_edge_order)

        action_type_to_idx = {a: i for i, a in enumerate(action_type_order)}
        table_to_idx = {t: i for i, t in enumerate(table_order)}
        attr_to_idx = {a: i for i, a in enumerate(qualified_attr_order)}
        edge_to_idx = {e: i for i, e in enumerate(edge_order)}

        type_offset = 0
        table_offset = type_offset + len(action_type_order)
        attr_offset = table_offset + len(table_order)
        edge_offset = attr_offset + len(qualified_attr_order)

        index_to_label: List[str] = []

        for a in action_type_order:
            index_to_label.append(f"type[{a.value}]")
        for t in table_order:
            index_to_label.append(f"table[{t}]")
        for qa in qualified_attr_order:
            index_to_label.append(f"attr[{qa}]")
        for e in edge_order:
            index_to_label.append(f"edge[{e[0]}={e[1]}]")

        return ActionCatalog(
            action_type_order=action_type_order,
            table_order=table_order,
            qualified_attr_order=qualified_attr_order,
            edge_order=edge_order,
            action_type_to_idx=action_type_to_idx,
            table_to_idx=table_to_idx,
            attr_to_idx=attr_to_idx,
            edge_to_idx=edge_to_idx,
            type_offset=type_offset,
            table_offset=table_offset,
            attr_offset=attr_offset,
            edge_offset=edge_offset,
            index_to_label=index_to_label,
        )

    # =========================
    # encode
    # =========================

    def encode_action(self, action: Action) -> np.ndarray:
        c = self.catalog
        vec = np.zeros(c.action_dim, dtype=np.float32)

        # 1) action type
        vec[c.type_offset + c.action_type_to_idx[action.action_type]] = 1.0

        # 2) table slot
        if action.table is not None:
            if action.table not in c.table_to_idx:
                raise EncodingError(f"未知表: {action.table}")
            vec[c.table_offset + c.table_to_idx[action.table]] = 1.0

        # 3) attr slot
        if action.attr is not None:
            if action.attr not in c.attr_to_idx:
                raise EncodingError(f"未知属性: {action.attr}")
            vec[c.attr_offset + c.attr_to_idx[action.attr]] = 1.0

        # 4) edge slot
        if action.edge is not None:
            edge_key = _normalize_edge_key(action.edge[0], action.edge[1])
            if edge_key not in c.edge_to_idx:
                raise EncodingError(f"未知 edge: {edge_key}")
            vec[c.edge_offset + c.edge_to_idx[edge_key]] = 1.0

        return vec

    # =========================
    # candidate action generation
    # =========================

    def enumerate_all_actions(self) -> List[Action]:
        """
        枚举静态候选动作全集。
        这里不做基于当前 state 的合法性过滤。
        """
        actions: List[Action] = []

        # 1) PartitionTable(table, attr)
        for qattr in self.ec.qualified_attr_order:
            table, _ = qattr.split(".", 1)
            actions.append(
                Action(
                    action_type=ActionType.PARTITION_TABLE,
                    table=table,
                    attr=qattr,
                )
            )

        # 2) ReplicateTable(table)
        for table in self.ec.table_order:
            if self.replication_allowed.get(table, False):
                actions.append(
                    Action(
                        action_type=ActionType.REPLICATE_TABLE,
                        table=table,
                    )
                )

        # 3) ActivateEdge(edge), DeactivateEdge(edge)
        for edge in self.ec.join_edge_order:
            actions.append(
                Action(
                    action_type=ActionType.ACTIVATE_EDGE,
                    edge=edge,
                )
            )
            actions.append(
                Action(
                    action_type=ActionType.DEACTIVATE_EDGE,
                    edge=edge,
                )
            )

        return actions

    # =========================
    # explain / layout
    # =========================

    def describe_layout(self, verbose: bool = True) -> Dict[str, Any]:
        c = self.catalog
        layout = {
            "action_dim": c.action_dim,
            "sections": {
                "action_type": {
                    "start": c.type_offset,
                    "end": c.table_offset,
                    "size": len(c.action_type_order),
                },
                "table": {
                    "start": c.table_offset,
                    "end": c.attr_offset,
                    "size": len(c.table_order),
                },
                "attr": {
                    "start": c.attr_offset,
                    "end": c.edge_offset,
                    "size": len(c.qualified_attr_order),
                },
                "edge": {
                    "start": c.edge_offset,
                    "end": c.action_dim,
                    "size": len(c.edge_order),
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
        if len(vector) != self.catalog.action_dim:
            raise EncodingError(
                f"向量长度不匹配: got={len(vector)}, expected={self.catalog.action_dim}"
            )

        return [
            (i, self.catalog.index_to_label[i], float(vector[i]))
            for i in range(len(vector))
        ]

    def explain_nonzero(self, vector: np.ndarray, threshold: float = 1e-8) -> List[Tuple[int, str, float]]:
        return [
            (idx, label, val)
            for idx, label, val in self.explain_vector(vector)
            if abs(val) > threshold
        ]

    def pretty_print_vector(self, vector: np.ndarray, threshold: float = -1.0) -> None:
        rows = self.explain_vector(vector)
        for idx, label, value in rows:
            if threshold >= 0 and abs(value) <= threshold:
                continue
            print(f"{idx:4d} | {label:50s} | {value:.6f}")
