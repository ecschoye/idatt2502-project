import argparse

from DDQN.training.ddqn_training import DDQNLogger, DDQNRenderer, DDQNTrainer
from PPO.ppo_trainer import PPOTrainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Project Options")
    parser.add_argument(
        "option",
        nargs="?",
        default="ddqn",
        type=str,
        help="Select an option: 'ddqn' for DDQN training, "
        "'ppo' for PPO training, 'render-ddqn' to render DDQN, "
        "'render-ppo' to render PPO, 'log-ddqn' to log DDQN model, ",
    )
    parser.add_argument(
        "--log",
        action="store_true",
        help="Enable logging (only applicable with 'ddqn')",
    )
    parser.add_argument(
        "--log-model",
        action="store_true",
        help="Enable model logging (only applicable with 'ddqn')",
    )

    args = parser.parse_args()

    if args.option == "ddqn":
        print("DDQN")
        DDQNTrainer().train(log=args.log)
        if args.log_model:
            print("Log model")
            DDQNLogger().log()
    elif args.option == "ppo":
        print("PPO")
        PPOTrainer().train(log=args.log)
    elif args.option == "render-ddqn":
        print("Rendering ddqn")
        DDQNRenderer().render()
    elif args.option == "render-ppo":
        print("Rendering ppo")
        PPOTrainer().test()
    elif args.option == "log-ddqn":
        print("Logging DDQN Model")
        DDQNLogger().log()
    elif args.option == "log-ppo":
        print("Logging PPO Model")
        PPOTrainer().log_model()
    else:
        print(
            "Invalid option. Please select an option "
            "(ddqn, ppo, render_ddqn, render_ppo)."
        )
