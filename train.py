import argparse
import yaml
import torch

from source_code.Agent.PPO import PPOAgent
from source_code.Agent.training import train_ppo_agent, save_training_run


if __name__ == "__main__":
    torch.set_num_threads(1)  # con batch=1 il threading intra-op è solo overhead, peggio ancora su cluster
    torch.distributions.Distribution.set_default_validate_args(False)
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./config.yaml', help="Path to config file")
    args = parser.parse_args()

    try:
        with open(args.config, 'r') as f:
            cfg = yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file {args.config} not found. Exiting.")
        raise SystemExit(1)


    print("Initializing Proximal Policy Optimization (PPO-Clip) Agent...")
    ppo_agent = PPOAgent(
        state_dim=cfg['PPO']['state_dim'],
        action_dim=cfg['PPO']['action_dim'],
        hidden_dim=cfg['PPO']['hidden_dim'],
        clip_ratio=cfg['PPO']['clip_ratio'],
        epochs=cfg['PPO']['epochs'],
        shared_net=cfg['PPO']['shared_net'],
    )
    # If needed to fine tune a pre-trained model
    #ppo_agent.load("checkpoints/really_good_model_copy.pt")

    returns_ppo = train_ppo_agent(cfg, ppo_agent)
    ppo_agent.save("checkpoints/ppo_sailing.pt")

    save_training_run(returns= returns_ppo, model_path="checkpoints/ppo_sailing.pt", cfg=cfg)

