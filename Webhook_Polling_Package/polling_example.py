
import time, requests

FAKE_API = "https://jsonplaceholder.typicode.com/posts/1"

while True:
    print(requests.get(FAKE_API).json())
    time.sleep(5)
