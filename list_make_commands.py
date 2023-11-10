import re

def list_make_commands(makefile_path):
    with open(makefile_path, 'r') as file:
        lines = file.readlines()

    pattern = re.compile(r'^([a-zA-Z_-]+):.*?## (.*)$')
    commands = [pattern.search(line) for line in lines]
    commands = [match.groups() for match in commands if match]

    for command, description in sorted(commands):
        print(f"{command:30} {description}")

if __name__ == "__main__":
    list_make_commands('Makefile')
