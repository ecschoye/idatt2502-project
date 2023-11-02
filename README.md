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
FILL IN FINAL PROBLEM STATEMENT

## Requirements
Here are the requirements to run the project

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

You may encounter an issue where imports are not working as they should. All you need to is use this command in the active virtual envoirement:

```bash
make imports
```

### Run
FILL IN HOW TO RUN BOTH PPO AND DDQN 

### Cleanup
This commands removes the virtual envoirment created in the setup section. You can also remove this manually.

```bash
make clean
```