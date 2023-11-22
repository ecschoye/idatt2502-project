# idatt2502-project
- written by Edvard Schøyen, Daniel Skymoen, Jarand Romestrand, Kristian Vaula Jensen

## Asignement
### Various reinforcement learning tasks
#### Assignment proposals:
- Choose an environment - test different algorithms
- Choose an algorithm - test different environments
- Create an environment
- Explore different configurations of an algorithm
- Try a hardcore environment: MineRL, Atari, Doom​

We choose to try a hardcore environment, and the environment we ended up with was super mario bros.

Read more about the environment here: https://pypi.org/project/gym-super-mario-bros/

## Problems statement
Compare the performence of DDQN and PPO in the Super Mario
Bros evniroment.

## Requirements
Here are the requirements to run the project:
- Python >=3.9, <3.12

Prefered requirements for easier setup and running of project:
- Make

### Windows
- Visual Studio C++ build tools

## Help
To list all the Make commands use the command under:
```bash
make help
```

## How to run
The guide for running the project assumes that you have Make installed. If you don't have it, you will find the commands you need in the Makefile

### Setup
This function creates a virtual environment for all the dependecies required for running the project and installs them.

``` bash
make setup
```

Pyhton3 is set to standard and if your computer uses python instead of python3 you can overwrite this with the command under:

```bash
make setup PYTHON=python
```

#### Neptun
An .env file is required to use Neptun for logging the training of a model.
Example of the contents needed for Neptun:
```bash
NEPTUNE_API_TOKEN="YOUR_API_KEY"
NEPTUNE_PROJECT_NAME="YOUR_NEPTUN_PROJECT_NAME"
```

### Running the project
The commands under activates the environment created in the seupup section and then runs the python code.

#### Train the models
This function activates the environment and then trains the ddqn.

``` bash
make ddqn
```

This function activates the environment and then trains the ppo.

``` bash
make ppo
```

#### Render trained models
This function activates the environment and then renders a trained model of ddqn.
``` bash
make render-ddqn
```

This function activates the environment and then renders a trained model of ppo.
``` bash
make render-ppo
```

#### Options
You can specify flags for both the ppo and ddqn to log the training to Neptun.
You can use these args on both ppo and ddqn. The args can also be combined. 
To specify flags for logging use the commands under:

##### Log training
Logs the traning to make graphs on Neptun
```bash
make ddqn args="--log"
```

##### Log the completed model
Logs the model to make graphs on Neptun
```bash
make ddqn args="--log-model"
```

##### Log training and the completed model
Logs the training and the model to make graphs on Neptun
```bash
make ddqn args="--log --log-model"
```

##### Python options
Pyhton3 is set to standard and if your computer uses python instead of python3 you can overwrite this with the command under:

```bash
make ddqn PYTHON=python
```

#### Deactivate the environment with:
``` bash
deactivate
```

### Cleanup
This commands removes the virtual environment created in the setup section. You can also remove this manually.

```bash
make clean
```

## Example runs
Here are some example runs:

### DDQN
Example runs with DDQN:

<img src="gifs/ddqn_super_mario_1.gif" alt="DDQN Run" width="400"/>

<img src="gifs/ddqn_super_mario_2.gif" alt="DDQN Run" width="400"/>

### PPO
An example run with PPO:

<img src="gifs/ppo_super_mario_1.gif" alt="PPO Run" width="400"/>

<img src="gifs/ppo_super_mario_2.gif" alt="PPO Run" width="400"/>
