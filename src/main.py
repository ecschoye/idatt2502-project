import argparse

from src.training.ddqn_training import (
    train_mario,
    render_mario,
    log_model_version
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Machine Learning Project Options")
    parser.add_argument("option", nargs="?", default="ddqn", type=str, help="Select an option (ddqn, ppo, render_ddqn, render_ppo)")
    parser.add_argument("--log", action="store_true", help="Enable logging")
    parser.add_argument("--log-model", action="store_true", help="Enable model logging")

    args = parser.parse_args()

    if args.option == "ddqn":
        print("DDQN")
        train_mario(log=args.log)
        if args.log_model:
            print("Log model")
            log_model_version()
    elif args.option == "ppo":
        print("PPO")
        #train_ppo(log=args.log) ADD PPO TRANING
        if args.log_model:
            print("Log model")
            #log_model_version() ADD PPO LOGING MODEL
    elif args.option == "render-ddqn":
        print("Rendering ddqn")
        render_mario()
    elif args.option == "render-ppo":
        print("Rendering ppo")
        #render_ppo() ADD PPO RENDER METHOD
    else:
        print("Invalid option. Please select an option (ddqn, ppo, render_ddqn, render_ppo).")