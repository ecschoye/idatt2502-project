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

We chose to try a hardcore envoirement, and the envoirement we ended up with was super mario bros.

Read more about the envoirement here: https://pypi.org/project/gym-super-mario-bros/

## Problems statement
Compareing the performence of DDQN and PPO in the evniroment Super Mario
Bros.

## Requirements
Here are the requirements to run the project:
- Python >= 3.10.8
- Preferably Make (For a easier setup of project)

## How to run
The commands used for the project assumes that you have Make installed. If you don't have it, you will find the commands you need in the Makefile

### Setup
This function creates a virtual envoirement for all the dependecies required for running the project and installs them.

``` bash
make setup
```

Pyhton3 is set to standard and if your computer uses python instead of python3 you can overwrite this with the command under:

```bash
make setup PYTHON=python
```

You may encounter an issue where imports are not working as they should. All you need to is use this command:

```bash
make imports
```

### Run
Make sure that you use the virtueal envoirement created in the setup section.
#### To activate the envoirement in commandline:
- Windows:
``` bash
cd venv/scripts && activate && cd .. && cd ..
```
- Other:
```bash
cd venv/bin && activate && cd .. && cd ..
```

#### Train and render
This function trains the ddqn.

``` bash
make ddqn
```

This function trains the ppo.

``` bash
make ppo
```

This function renders a trained model of ddqn.
``` bash
make render-ddqn
```

This function renders a trained model of ppo.
``` bash
make render-ppo
```

#### Options
You can specify flags for both the ppo and ddqn to log the training to Neptun.
You can use these args on both ppo and ddqn. The args can also be combined. 
To specify flags for logging use the commands under:

Logs the traning to make graphs
```bash
make ddqn args="--log"
```

Logs the model to make graphs
```bash
make ddqn args="--log-model"
```

Pyhton3 is set to standard and if your computer uses python instead of python3 you can overwrite this with the command under:

```bash
make ddqn PYTHON=python
```

#### Deactivate the envoirement with:
``` bash
deactivate
```

### Cleanup
This commands removes the virtual envoirment created in the setup section. You can also remove this manually.

```bash
make clean
```
