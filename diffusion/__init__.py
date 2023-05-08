from dotenv import load_dotenv
import os


def load_env_variables(path=os.path.join("..", ".env")):
    load_dotenv(path)
    env_variables = {}
    for key, value in os.environ.items():
        env_variables[key] = value
    return env_variables