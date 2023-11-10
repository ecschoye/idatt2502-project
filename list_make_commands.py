import re

"""List all commands in a Makefile"""
makefile_path = 'Makefile'
with open(makefile_path, 'r') as file:
    lines = file.readlines()

pattern = re.compile(r'^([a-zA-Z_-]+):.*?## (.*)$')
commands = [pattern.search(line) for line in lines]
commands = [match.groups() for match in commands if match]

for command, description in sorted(commands):
    print(f"{command:30} {description}")
