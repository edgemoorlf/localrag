import os
from os.path import join, dirname
from dotenv import load_dotenv

dotenv_path = join(dirname(__file__), '.env')
load_dotenv(dotenv_path)

BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY")
print(BRAVE_API_KEY)