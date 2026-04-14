
"""
RLPO - 主程序入口
根据配置文件执行离线训练、在线训练或使用 Agent 输出结果
"""
import os
import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from encoding import StateEncoder
from action import ActionEncoder
from env import PartitioningEnv
from agent import DQNAgent
from data_loader import (
    build_training_data,
    create_cost_based_reward_fn,
)
from online_train import (
    QueryLatencyCache,
    DatabaseManager,
    create_online_train_reward_fn,
)

import config as cfg


def setup_environment():
    """根据配置设置环境和数据"""
    workload_name = cfg.WORKLOAD_NAME
    data_folder = cfg.DATA_FOLDER
    
    print("=" * 80)
    print("正在加载数据...")
    print("=" * 80)
    
    training_data = build_training_data(data_folder)
    
    table_attrs = training_data["table_attrs"]
    table_sizes = training_data["table_sizes"]
    replication_allowed = training_data["replication_allowed"]
    join_set = training_data["join_set"]
    workload_dict = training_data["workload_dict"]
    target_edges = training_data["target_edges"]
    initial_partition = training_data.get("initial_partition", None)
    initial_replication = training_data.get("initial_replication", None)
    
    print("\n正在创建编码器和环境...")
    state_encoder = StateEncoder(
        table_attrs=table_attrs,
        replication_allowed=replication_allowed,
        join_set=join_set,
        workload_dict=workload_dict,
        normalize_workload=True,
    )
    
    action_encoder = ActionEncoder(
        encoding_catalog=state_encoder.catalog,
        replication_allowed=replication_allowed,
    )
    
    reward_fn = create_cost_based_reward_fn(table_sizes, target_edges)
    
    return {
        "table_attrs": table_attrs,
        "table_sizes": table_sizes,
        "replication_allowed": replication_allowed,
        "join_set": join_set,
        "workload_dict": workload_dict,
        "target_edges": target_edges,
        "initial_partition": initial_partition,
        "initial_replication": initial_replication,
        "state_encoder": state_encoder,
        "action_encoder": action_encoder,
        "reward_fn": reward_fn,
        "queries": workload_dict,
    }


def offline_train(env_data):
    """执行离线训练"""
    print("\n" + "=" * 80)
    print("离线训练模式")
    print("=" * 80)
    
    offline_config = cfg.OFFLINE_TRAIN
    
    state_encoder = env_data["state_encoder"]
    action_encoder = env_data["action_encoder"]
    reward_fn = env_data["reward_fn"]
    initial_partition = env_data["initial_partition"]
    initial_replication = env_data["initial_replication"]
    
    env = PartitioningEnv(
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        tmax=offline_config["tmax"],
        reward_fn=reward_fn,
        initial_partition=initial_partition,
        initial_replication=initial_replication,
    )
    
    state_dim = state_encoder.catalog.state_dim
    agent = DQNAgent(
        state_dim=state_dim,
        action_encoder=action_encoder,
        hidden_dim=offline_config["hidden_dim"],
        lr=offline_config["lr"],
        gamma=offline_config["gamma"],
        epsilon_start=offline_config["epsilon_start"],
        epsilon_end=offline_config["epsilon_end"],
        epsilon_decay=offline_config["epsilon_decay"],
        target_update=offline_config["target_update"],
        buffer_capacity=offline_config["buffer_capacity"],
    )
    
    last_agent_path = cfg.LAST_AGENT_PATH
    if last_agent_path and os.path.exists(last_agent_path):
        print("\n正在加载已有模型:", last_agent_path)
        agent.load(last_agent_path)
    
    num_episodes = offline_config["num_episodes"]
    batch_size = offline_config["batch_size"]
    
    print("\n开始训练...")
    print("训练回合数:", num_episodes)
    print("\n" + "-" * 80)
    
    episode_rewards = []
    episode_losses = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        state = env.reset()
        encoded_state = env.encode_state(state)
        total_reward = 0.0
        total_loss = 0.0
        steps = 0
        done = False
        
        while not done:
            legal_actions = env.legal_actions(state)
            if not legal_actions:
                break
            
            action, action_vec = agent.select_action(encoded_state, legal_actions, training=True)
            next_state, reward, done, info = env.step(action)
            next_encoded_state = env.encode_state(next_state)
            
            agent.replay_buffer.push(
                encoded_state.vector,
                action_vec,
                reward,
                next_encoded_state.vector,
                done,
            )
            
            loss = agent.update(batch_size)
            total_loss += loss
            
            encoded_state = next_encoded_state
            state = next_state
            total_reward += reward
            steps += 1
        
        episode_rewards.append(total_reward)
        episode_losses.append(total_loss / max(steps, 1))
        episode_lengths.append(steps)
        
        if (episode + 1) % 200 == 0:
            avg_reward = np.mean(episode_rewards[-200:])
            avg_loss = np.mean(episode_losses[-200:])
            avg_length = np.mean(episode_lengths[-200:])
            print("Episode", episode + 1, "/", num_episodes,
                  "| Avg Reward:", round(avg_reward, 2),
                  "| Avg Loss:", round(avg_loss, 4),
                  "| Avg Length:", round(avg_length, 1),
                  "| Epsilon:", round(agent.epsilon, 3))
    
    print("-" * 80)
    print("\n训练完成!")
    
    # 保存模型
    workload_name = cfg.WORKLOAD_NAME
    results_dir = os.path.join("results", workload_name)
    os.makedirs(results_dir, exist_ok=True)
    
    agent_path = os.path.join(results_dir, "dqn_agent.pth")
    agent.save(agent_path)
    print("\n模型已保存到:", agent_path)
    
    # 保存训练曲线
    fig, axes = plt.subplots(3, 1, figsize=(12, 12))
    
    window = 200
    if len(episode_rewards) >= window:
        smoothed_rewards = np.convolve(episode_rewards, np.ones(window) / window, mode="valid")
        axes[0].plot(range(window, len(episode_rewards) + 1), smoothed_rewards, label="Smoothed (window=" + str(window) + ")")
    axes[0].plot(episode_rewards, alpha=0.3, label="Raw")
    axes[0].set_xlabel("Episode")
    axes[0].set_ylabel("Reward")
    axes[0].set_title("Training Rewards")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    if len(episode_losses) >= window:
        smoothed_losses = np.convolve(episode_losses, np.ones(window) / window, mode="valid")
        axes[1].plot(range(window, len(episode_losses) + 1), smoothed_losses, label="Smoothed (window=" + str(window) + ")")
    axes[1].plot(episode_losses, alpha=0.3, label="Raw")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Training Losses")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    if len(episode_lengths) >= window:
        smoothed_lengths = np.convolve(episode_lengths, np.ones(window) / window, mode="valid")
        axes[2].plot(range(window, len(episode_lengths) + 1), smoothed_lengths, label="Smoothed (window=" + str(window) + ")")
    axes[2].plot(episode_lengths, alpha=0.3, label="Raw")
    axes[2].set_xlabel("Episode")
    axes[2].set_ylabel("Episode Length")
    axes[2].set_title("Episode Lengths")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    curves_path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(curves_path)
    print("训练曲线已保存到:", curves_path)
    
    import pickle
    stats_path = os.path.join(results_dir, "training_stats.pkl")
    with open(stats_path, "wb") as f:
        pickle.dump({
            "rewards": episode_rewards,
            "losses": episode_losses,
            "lengths": episode_lengths,
        }, f)
    print("训练统计已保存到:", stats_path)


def online_train(env_data):
    """执行在线训练"""
    print("\n" + "=" * 80)
    print("在线训练模式")
    print("=" * 80)
    
    offline_config = cfg.OFFLINE_TRAIN
    online_config = cfg.ONLINE_TRAIN
    
    state_encoder = env_data["state_encoder"]
    action_encoder = env_data["action_encoder"]
    queries = env_data["queries"]
    initial_partition = env_data["initial_partition"]
    initial_replication = env_data["initial_replication"]
    
    try:
        print("\n正在初始化数据库连接...")
        db_manager = DatabaseManager()
        
        print("正在初始化查询延迟缓存...")
        latency_cache = QueryLatencyCache("query_latency_cache.json")
        
        print("正在创建在线训练奖励函数...")
        reward_fn, reward_calculator = create_online_train_reward_fn(
            db_manager, latency_cache, queries
        )
        
        print("\n正在创建强化学习环境...")
        env = PartitioningEnv(
            state_encoder=state_encoder,
            action_encoder=action_encoder,
            tmax=offline_config["tmax"],
            reward_fn=reward_fn,
            initial_partition=initial_partition,
            initial_replication=initial_replication,
        )
        
        state_dim = state_encoder.catalog.state_dim
        agent = DQNAgent(
            state_dim=state_dim,
            action_encoder=action_encoder,
            hidden_dim=offline_config["hidden_dim"],
            lr=offline_config["lr"],
            gamma=offline_config["gamma"],
            epsilon_start=offline_config["epsilon_start"],
            epsilon_end=offline_config["epsilon_end"],
            epsilon_decay=offline_config["epsilon_decay"],
            target_update=offline_config["target_update"],
            buffer_capacity=offline_config["buffer_capacity"],
        )
        
        last_agent_path = cfg.LAST_AGENT_PATH
        if last_agent_path and os.path.exists(last_agent_path):
            print("\n正在加载已有模型:", last_agent_path)
            agent.load(last_agent_path)
        
        num_steps = online_config["num_steps"]
        batch_size = online_config["batch_size"]
        
        print("\n开始在线训练...")
        print("训练步数:", num_steps)
        print("\n" + "-" * 80)
        
        state = env.reset()
        encoded_state = env.encode_state(state)
        total_reward = 0.0
        total_loss = 0.0
        
        for step in range(num_steps):
            legal_actions = env.legal_actions(state)
            action, action_vec = agent.select_action(encoded_state, legal_actions, training=True)
            next_state, reward, done, info = env.step(action)
            next_encoded_state = env.encode_state(next_state)
            
            agent.replay_buffer.push(
                encoded_state.vector, 
                action_vec, 
                reward, 
                next_encoded_state.vector, 
                done
            )
            
            loss = agent.update(batch_size)
            total_loss += loss
            total_reward += reward
            
            if (step + 1) % 50 == 0:
                avg_reward = total_reward / (step + 1)
                avg_loss = total_loss / (step + 1)
                print(
                    f"Step {step + 1}/{num_steps} | "
                    f"Avg Reward: {avg_reward:.2f} | "
                    f"Avg Loss: {avg_loss:.4f} | "
                    f"Epsilon: {agent.epsilon:.3f}"
                )
            
            encoded_state = next_encoded_state
            state = next_state
            
            if done:
                state = env.reset()
                encoded_state = env.encode_state(state)
        
        print("-" * 80)
        print("\n训练完成!")
        
        workload_name = cfg.WORKLOAD_NAME
        results_dir = os.path.join("results", workload_name)
        os.makedirs(results_dir, exist_ok=True)
        
        agent_path = os.path.join(results_dir, "dqn_agent_online.pth")
        agent.save(agent_path)
        print("\n模型已保存到:", agent_path)
        
        print("\n正在关闭数据库连接...")
        db_manager.close()
        
        print("\n在线训练完成!")
        
    except Exception as e:
        print(f"\n在线训练失败: {e}")
        import traceback
        traceback.print_exc()


def inference(env_data):
    """使用 Agent 输出结果"""
    print("\n" + "=" * 80)
    print("推理模式 - 使用 Agent 输出分区推荐")
    print("=" * 80)
    
    agent_path = cfg.LAST_AGENT_PATH
    if not agent_path or not os.path.exists(agent_path):
        print("错误: 找不到训练好的 Agent！")
        print("请先进行离线训练，或在配置文件中设置正确的 LAST_AGENT_PATH。")
        return
    
    state_encoder = env_data["state_encoder"]
    action_encoder = env_data["action_encoder"]
    reward_fn = env_data["reward_fn"]
    initial_partition = env_data["initial_partition"]
    initial_replication = env_data["initial_replication"]
    offline_config = cfg.OFFLINE_TRAIN
    inference_config = cfg.INFERENCE
    
    env = PartitioningEnv(
        state_encoder=state_encoder,
        action_encoder=action_encoder,
        tmax=100,
        reward_fn=reward_fn,
        initial_partition=initial_partition,
        initial_replication=initial_replication,
    )
    
    state_dim = state_encoder.catalog.state_dim
    agent = DQNAgent(
        state_dim=state_dim,
        action_encoder=action_encoder,
        hidden_dim=offline_config["hidden_dim"],
        lr=offline_config["lr"],
        gamma=offline_config["gamma"],
        epsilon_start=0.01,
        epsilon_end=0.01,
        epsilon_decay=1.0,
        target_update=offline_config["target_update"],
        buffer_capacity=offline_config["buffer_capacity"],
    )
    
    print("\n正在加载模型:", agent_path)
    agent.load(agent_path)
    print("模型加载成功!")
    
    print("\n开始推理...")
    state = env.reset()
    print("\n初始状态:")
    env.pretty_print_state(state)
    
    print("\n" + "-" * 80)
    print("执行智能体选择的动作...")
    print("-" * 80)
    
    max_steps = inference_config.get("max_steps", 50)
    for step in range(max_steps):
        encoded_state = env.encode_state(state)
        legal_actions = env.legal_actions(state)
        
        if not legal_actions:
            print("没有可用的动作了!")
            break
        
        action, _ = agent.select_action(encoded_state, legal_actions, training=False)
        next_state, reward, done, info = env.step(action)
        
        print("Step", step + 1, ":", str(action), "| Reward:", round(reward, 2))
        
        state = next_state
        
        if done:
            break
    
    print("\n" + "=" * 80)
    print("推理完成! 最终状态:")
    print("=" * 80)
    env.pretty_print_state(state)
    
    # 输出到 results/{workload}/z_{current_time}.txt
    workload_name = cfg.WORKLOAD_NAME
    results_dir = os.path.join("results", workload_name)
    os.makedirs(results_dir, exist_ok=True)
    
    current_time = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(results_dir, "z_" + current_time + "infer.txt")
    
    print("\n正在将结果写入:", output_path)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("RLPO - Inference Results\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("=== Replication State ===\n")
        for table in sorted(state.replication_state.keys()):
            f.write(table + ": " + str(state.replication_state[table]) + "\n")
        
        f.write("\n")
        
        f.write("=== Partition State ===\n")
        for table in sorted(state.partition_state.keys()):
            chosen_attr = None
            for attr, is_set in state.partition_state[table].items():
                if is_set:
                    chosen_attr = attr
                    break
            if chosen_attr:
                f.write(table + ": " + table + "." + chosen_attr + "\n")
            else:
                f.write(table + ": None (Replicated)\n")
        
        f.write("\n")
        
        f.write("=== Join State ===\n")
        for a1, a2, is_active in sorted(state.join_state):
            f.write(a1 + " <-> " + a2 + ": " + str(is_active) + "\n")
        
        f.write("\n")
        f.write("=" * 80 + "\n")
    
    print("结果已保存到:", output_path)


def main():
    """主函数"""
    print("=" * 80)
    print("RLPO - Learning a Partitioning Advisor for Cloud Databases")
    print("=" * 80)
    
    mode = cfg.MODE
    
    env_data = setup_environment()
    
    if mode == "offline_train":
        offline_train(env_data)
    elif mode == "online_train":
        online_train(env_data)
    elif mode == "inference":
        inference(env_data)
    else:
        print("错误: 未知模式:", mode)
        print("支持的模式: offline_train, inference")


if __name__ == "__main__":
    main()

