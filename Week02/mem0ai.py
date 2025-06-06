# Imports 
import os
from dotenv import load_dotenv
from mem0 import MemoryClient, Memory

# api key 
load_dotenv()
api_key = os.getenv('MEM0_API_KEY')
if api_key:
    print(f"Memo API key is found: {api_key[:10]}")
else:
    print("Unable to fetch API key...")

# Creating Memory Client
try:
    client = MemoryClient(api_key=api_key)
    print("...Client Craeted Successfully...")
except Exception as e:
    print(f"Error creating client: {e}")

## MEMORY OPERATIONS ##
# 1. Create Memories
message = [
    {'role':'user', 'content':"Hi, I am Anuj kumar. I am a vegitarian and I really like to explore new type of food/Recipies. But keep in mind I am allergic to nuts"},
    {'role':'assistant', 'content':"Hello Anuj! I've noted that you're a vegetarian and have a nut allergy. I'll keep this in mind for any food-related recommendations or discussions."}
]
try:
    res = client.add(message, user_id='anuj001', metadata={'food':'vegan'})
    print(f"=== Memory added successfully ===")
except Exception as e:
    print(f"=== Unable to add memory, Error has occured: {e} ===")

# 2. Search Memories
query = "What is my Name, And is my hobby."
try: 
    res = client.search(query, user_id='anuj001')
    print(type(res))
    for i in res:
        print(i)
        print("="*100)
except Exception as e:
    print(f"Error: {e}")

print(client.users())
print("="*50+" Get All Memories "+"="*50)
for i in client.get_all(user_id='anuj001'):
    for key,value in i.items():
        print(f"{key} --> {value}")
    print("*"*100)