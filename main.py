import threading
import queue
from encoding import *

table_attrs = {
    "orders": {
        "order_id": False,
        "customer_id": True,
        "nation_id": False,
    },
    "customer": {
        "customer_id": True,
        "nation_id": False,
    },
    "nation": {
        "nation_id": True,
        "name": False,
    },
}

replication_allowed = {
    "orders": False,
    "customer": True,
    "nation": True,
}

join_set = {
    ("orders.customer_id", "customer.customer_id", True),
    ("customer.nation_id", "nation.nation_id", False),
}

workload_dict = {
    "Q1: orders join customer": 100,
    "Q2: customer join nation": 40,
}

encoder = StateEncoder(
    table_attrs=table_attrs,
    replication_allowed=replication_allowed,
    join_set=join_set,
    workload_dict=workload_dict,
    normalize_workload=True,
)

layout = encoder.describe_layout(verbose=True)
print(layout)

encoded = encoder.encode_state(
    current_replication_state={
        "orders": False,
        "customer": True,
        "nation": False,
    }
)

print("state_dim =", encoded.vector.shape[0])
encoder.pretty_print_vector(encoded.vector, threshold=0)