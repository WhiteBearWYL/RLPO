
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Set, Tuple, Optional, List, Callable, Any
import copy

from encoding import (
    QualifiedAttr,
    JoinTriple,
    EncodingError,
    StateEncoder,
    _split_qualified_attr,
    _normalize_edge_key,
)
from action import Action, ActionType, ActionEncoder


EdgeKey = Tuple[QualifiedAttr, QualifiedAttr]


@dataclass
class PartitionEnvState:
    """
    环境中的显式状态。
    """
    replication_state: Dict[str, bool]
    partition_state: Dict[str, Dict[str, bool]]
    join_state: Set[JoinTriple]
    workload_dict: Dict[str, int]
    step_count: int = 0

    def clone(self):
        return PartitionEnvState(
            replication_state=copy.deepcopy(self.replication_state),
            partition_state=copy.deepcopy(self.partition_state),
            join_state=set(self.join_state),
            workload_dict=copy.deepcopy(self.workload_dict),
            step_count=self.step_count,
        )


class PartitioningEnv:
    """
    环境职责：
    - 动作合法性检查
    - 状态联动更新
    - reward 计算
    - episode 控制

    当前语义：
    1. Partition/Replicate/ActivateEdge 会自动重算相关 edge
    2. DeactivateEdge 仅把当前 edge 置 False
    3. DeactivateEdge 不持久抑制，后续若再次发生相关 partition 变化，
       该 edge 会根据当前 partitioning 被重新计算
    """

    def __init__(
        self,
        state_encoder,
        action_encoder,
        tmax=100,
        reward_fn=None,
        initial_partition=None,
        initial_replication=None,
    ):
        self.state_encoder = state_encoder
        self.action_encoder = action_encoder
        self.tmax = tmax
        self.reward_fn = reward_fn
        self.initial_partition = initial_partition
        self.initial_replication = initial_replication

        self.table_attrs = self.state_encoder.table_attrs
        self.replication_allowed = self.state_encoder.replication_allowed
        self.base_join_set = self.state_encoder.join_set
        self.initial_workload = self.state_encoder.workload_dict

        self.tables = list(self.state_encoder.catalog.table_order)
        self.qualified_attrs = list(self.state_encoder.catalog.qualified_attr_order)
        self.edge_order = list(self.state_encoder.catalog.join_edge_order)

        self._table_to_incident_edges = self._build_table_to_incident_edges()
        self.state = self._build_initial_state()

    # ============================================================
    # initialization
    # ============================================================

    def _build_table_to_incident_edges(self):
        result = {t: [] for t in self.tables}
        for edge in self.edge_order:
            a1, a2 = edge
            t1, _ = _split_qualified_attr(a1)
            t2, _ = _split_qualified_attr(a2)
            result[t1].append(edge)
            if t2 != t1:
                result[t2].append(edge)
        return result

    def _build_initial_state(self):
        # 初始复制状态
        if self.initial_replication:
            replication_state = {t: self.initial_replication.get(t, False) for t in self.tables}
        else:
            replication_state = {t: False for t in self.tables}

        # 初始分区状态
        partition_state = {
            table: {attr: False for attr in attrs.keys()}
            for table, attrs in self.table_attrs.items()
        }
        
        # 从 table_attrs 设置默认分区键（仅对未复制的表）
        for table, attrs in self.table_attrs.items():
            if not replication_state[table]:
                for attr, is_default_key in attrs.items():
                    if is_default_key:
                        partition_state[table][attr] = True
        
        # 从初始状态覆盖（仅对未复制的表）
        if self.initial_partition:
            for table, attr in self.initial_partition.items():
                if table in partition_state and attr in partition_state[table]:
                    if not replication_state[table]:
                        # 先清除该表的所有分区键
                        for a in partition_state[table]:
                            partition_state[table][a] = False
                        # 然后设置指定的分区键
                        partition_state[table][attr] = True

        join_state = set()
        for a1, a2, is_active in self.base_join_set:
            key = _normalize_edge_key(a1, a2)
            join_state.add((key[0], key[1], bool(is_active)))

        workload_dict = copy.deepcopy(self.initial_workload)

        state = PartitionEnvState(
            replication_state=replication_state,
            partition_state=partition_state,
            join_state=join_state,
            workload_dict=workload_dict,
            step_count=0,
        )

        self._validate_state_shape_only(state)
        return state

    # ============================================================
    # public api
    # ============================================================

    def reset(self):
        self.state = self._build_initial_state()
        return self.state.clone()

    def get_state(self):
        return self.state.clone()

    def encode_state(self, state=None):
        s = self.state if state is None else state
        return self.state_encoder.encode_state(
            current_replication_state=s.replication_state,
            current_partition_state=s.partition_state,
            current_join_state=s.join_state,
            workload_dict=s.workload_dict,
        )

    def legal_actions(self, state=None):
        s = self.state if state is None else state
        all_actions = self.action_encoder.enumerate_all_actions()
        return [a for a in all_actions if self.is_action_legal(a, s)]

    def is_done(self, state=None):
        s = self.state if state is None else state
        return s.step_count >= self.tmax

    def step(self, action):
        if self.is_done(self.state):
            raise EncodingError("当前 episode 已结束，不能继续 step。")

        if not self.is_action_legal(action, self.state):
            raise EncodingError("非法动作: " + str(action))

        next_state = self.state.clone()

        if action.action_type == ActionType.PARTITION_TABLE:
            self._apply_partition_table(next_state, action.table, action.attr)

        elif action.action_type == ActionType.REPLICATE_TABLE:
            self._apply_replicate_table(next_state, action.table)

        elif action.action_type == ActionType.ACTIVATE_EDGE:
            self._apply_activate_edge(next_state, action.edge)

        elif action.action_type == ActionType.DEACTIVATE_EDGE:
            self._apply_deactivate_edge(next_state, action.edge)

        else:
            raise EncodingError("未知动作类型: " + str(action.action_type))

        next_state.step_count += 1
        self._validate_state_shape_only(next_state)

        reward = self._compute_reward(next_state)
        done = self.is_done(next_state)
        info = {
            "action": str(action),
            "step_count": next_state.step_count,
        }

        self.state = next_state
        return next_state.clone(), reward, done, info

    # ============================================================
    # legality checking
    # ============================================================

    def is_action_legal(self, action, state=None):
        s = self.state if state is None else state
        try:
            if action.action_type == ActionType.PARTITION_TABLE:
                return self._is_partition_table_legal(s, action.table, action.attr)
            if action.action_type == ActionType.REPLICATE_TABLE:
                return self._is_replicate_table_legal(s, action.table)
            if action.action_type == ActionType.ACTIVATE_EDGE:
                return self._is_activate_edge_legal(s, action.edge)
            if action.action_type == ActionType.DEACTIVATE_EDGE:
                return self._is_deactivate_edge_legal(s, action.edge)
            return False
        except EncodingError:
            return False

    def _is_partition_table_legal(self, state, table, qattr):
        if table is None or qattr is None:
            return False

        t, attr = _split_qualified_attr(qattr)
        if t != table:
            return False
        if table not in self.table_attrs:
            return False
        if attr not in self.table_attrs[table]:
            return False

        # 已经是该 partition 且未复制，视为空操作
        if (
            state.replication_state.get(table, False) is False
            and self._get_current_partition_attr(state, table) == qattr
        ):
            return False

        return True

    def _is_replicate_table_legal(self, state, table):
        if table is None:
            return False
        if table not in self.table_attrs:
            return False
        if not self.replication_allowed.get(table, False):
            return False

        if state.replication_state.get(table, False):
            return False

        return True

    def _is_activate_edge_legal(self, state, edge):
        if edge is None:
            return False

        edge = _normalize_edge_key(edge[0], edge[1])
        if edge not in self.edge_order:
            return False

        # 已经 active，则视为空操作
        if self._is_edge_active(state, edge):
            return False

        return True

    def _is_deactivate_edge_legal(self, state, edge):
        if edge is None:
            return False

        edge = _normalize_edge_key(edge[0], edge[1])
        if edge not in self.edge_order:
            return False

        return self._is_edge_active(state, edge)

    # ============================================================
    # transition rules
    # ============================================================

    def _apply_partition_table(self, state, table, qattr):
        """
        规则：
        - rep[T] = False
        - 清空 T 的所有分区键
        - part[T.a] = True
        - 自动重算与 T 相关的所有 edge
        """
        t, _ = _split_qualified_attr(qattr)
        if t != table:
            raise EncodingError("PartitionTable 参数不一致: table=" + table + ", attr=" + qattr)

        state.replication_state[table] = False
        self._clear_table_partition(state, table)
        self._set_partition_attr(state, table, qattr)
        self._recompute_edges_for_tables(state, {table})

    def _apply_replicate_table(self, state, table):
        """
        规则：
        - rep[T] = True
        - 清空 T 的所有分区键
        - 自动重算与 T 相关的所有 edge（都会变 False）
        """
        if not self.replication_allowed.get(table, False):
            raise EncodingError("表 '" + table + "' 不允许复制。")

        state.replication_state[table] = True
        self._clear_table_partition(state, table)
        self._recompute_edges_for_tables(state, {table})

    def _apply_activate_edge(self, state, edge):
        """
        规则：
        - 将两端表改为对应分区键
        - 两端表取消 replicate
        - 自动重算与两端表相关的所有 edge
        """
        edge = _normalize_edge_key(edge[0], edge[1])
        a1, a2 = edge
        t1, _ = _split_qualified_attr(a1)
        t2, _ = _split_qualified_attr(a2)

        state.replication_state[t1] = False
        state.replication_state[t2] = False

        self._clear_table_partition(state, t1)
        self._clear_table_partition(state, t2)

        self._set_partition_attr(state, t1, a1)
        self._set_partition_attr(state, t2, a2)

        self._recompute_edges_for_tables(state, {t1, t2})

    def _apply_deactivate_edge(self, state, edge):
        """
        规则：
        - 仅将该 edge 置 False
        - 不自动修改表分区键
        - 不做持久抑制
        """
        edge = _normalize_edge_key(edge[0], edge[1])
        self._set_edge_active(state, edge, False)

    # ============================================================
    # reward
    # ============================================================

    def _compute_reward(self, state):
        if self.reward_fn is not None:
            return float(self.reward_fn(state))
        return 0.0

    # ============================================================
    # state utilities
    # ============================================================

    def _get_current_partition_attr(self, state, table):
        chosen = []
        for attr, val in state.partition_state[table].items():
            if bool(val):
                chosen.append(table + "." + attr)

        if len(chosen) > 1:
            raise EncodingError("表 '" + table + "' 同时存在多个分区键: " + str(chosen))
        if len(chosen) == 0:
            return None
        return chosen[0]

    def _clear_table_partition(self, state, table):
        for attr in state.partition_state[table]:
            state.partition_state[table][attr] = False

    def _set_partition_attr(self, state, table, qattr):
        t, attr = _split_qualified_attr(qattr)
        if t != table:
            raise EncodingError("_set_partition_attr 参数不一致: table=" + table + ", qattr=" + qattr)
        if attr not in state.partition_state[table]:
            raise EncodingError("表 '" + table + "' 不包含属性 '" + attr + "'")

        self._clear_table_partition(state, table)
        state.partition_state[table][attr] = True

    def _is_edge_active(self, state, edge):
        edge = _normalize_edge_key(edge[0], edge[1])
        for a1, a2, is_active in state.join_state:
            if _normalize_edge_key(a1, a2) == edge:
                return bool(is_active)
        raise EncodingError("edge 不存在于 join_state 中: " + str(edge))

    def _set_edge_active(self, state, edge, is_active):
        edge = _normalize_edge_key(edge[0], edge[1])
        found = False
        new_join_state = set()

        for a1, a2, old_active in state.join_state:
            key = _normalize_edge_key(a1, a2)
            if key == edge:
                new_join_state.add((key[0], key[1], bool(is_active)))
                found = True
            else:
                new_join_state.add((key[0], key[1], bool(old_active)))

        if not found:
            raise EncodingError("edge 不存在于 join_state 中: " + str(edge))

        state.join_state = new_join_state

    def _derived_edge_active(self, state, edge):
        """
        根据当前 partition/replication 状态推导一条 edge 是否应为 active。
        """
        edge = _normalize_edge_key(edge[0], edge[1])
        a1, a2 = edge
        t1, _ = _split_qualified_attr(a1)
        t2, _ = _split_qualified_attr(a2)

        if state.replication_state[t1] or state.replication_state[t2]:
            return False

        p1 = self._get_current_partition_attr(state, t1)
        p2 = self._get_current_partition_attr(state, t2)

        return (p1 == a1) and (p2 == a2)

    def _recompute_edges_for_tables(self, state, tables):
        """
        对与指定表相关的所有 edge 按当前 partition/replication 自动重算。
        """
        new_join_state = set()

        for a1, a2, old_active in state.join_state:
            key = _normalize_edge_key(a1, a2)
            t1, _ = _split_qualified_attr(key[0])
            t2, _ = _split_qualified_attr(key[1])

            if t1 in tables or t2 in tables:
                is_active = self._derived_edge_active(state, key)
                new_join_state.add((key[0], key[1], is_active))
            else:
                new_join_state.add((key[0], key[1], bool(old_active)))

        state.join_state = new_join_state

    # ============================================================
    # validation
    # ============================================================

    def _validate_state_shape_only(self, state):
        # replication_state 完整
        for t in self.tables:
            if t not in state.replication_state:
                raise EncodingError("state.replication_state 缺少表 '" + t + "'")

        # partition_state 完整，且每表最多一个 True
        for t in self.tables:
            if t not in state.partition_state:
                raise EncodingError("state.partition_state 缺少表 '" + t + "'")

            attrs = state.partition_state[t]
            expected_attrs = set(self.table_attrs[t].keys())
            got_attrs = set(attrs.keys())
            if expected_attrs != got_attrs:
                raise EncodingError(
                    "表 '" + t + "' 的 partition_state 属性集合不一致: expected=" + str(expected_attrs) + ", got=" + str(got_attrs)
                )

            true_count = sum(1 for v in attrs.values() if bool(v))
            if true_count > 1:
                raise EncodingError("表 '" + t + "' 同时存在多个分区键。")

            if state.replication_state[t] and true_count != 0:
                raise EncodingError("表 '" + t + "' 已复制，但仍有分区键。")

        # join_state 必须覆盖所有 edge 且不重复
        seen = set()
        expected_edges = set(self.edge_order)
        got_edges = set()

        for a1, a2, is_active in state.join_state:
            key = _normalize_edge_key(a1, a2)
            if key in seen:
                raise EncodingError("join_state 中存在重复 edge: " + str(key))
            seen.add(key)
            got_edges.add(key)

            if key not in expected_edges:
                raise EncodingError("join_state 中存在未知 edge: " + str(key))
            if not isinstance(is_active, bool):
                raise EncodingError("edge 状态必须为 bool: " + str(key))

        if got_edges != expected_edges:
            missing = expected_edges - got_edges
            extra = got_edges - expected_edges
            raise EncodingError(
                "join_state 覆盖不完整。missing=" + str(missing) + ", extra=" + str(extra)
            )

    # ============================================================
    # debug helpers
    # ============================================================

    def state_to_dict(self, state=None):
        s = self.state if state is None else state
        return {
            "replication_state": copy.deepcopy(s.replication_state),
            "partition_state": copy.deepcopy(s.partition_state),
            "join_state": sorted(list(s.join_state)),
            "workload_dict": copy.deepcopy(s.workload_dict),
            "step_count": s.step_count,
        }

    def pretty_print_state(self, state=None):
        s = self.state if state is None else state

        print("=== Replication State ===")
        for t in self.tables:
            print(t + ": " + str(s.replication_state[t]))

        print("\n=== Partition State ===")
        for t in self.tables:
            chosen = self._get_current_partition_attr(s, t)
            print(t + ": " + str(chosen))

        print("\n=== Join State ===")
        for a1, a2, is_active in sorted(s.join_state):
            print(a1 + " <-> " + a2 + ": " + str(is_active))

        print("\n=== Workload ===")
        for q, c in s.workload_dict.items():
            print(q + ": " + str(c))

        print("\nstep_count = " + str(s.step_count))

