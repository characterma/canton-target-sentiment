from locust import HttpUser, task, TaskSet
import locust.stats
import os
from pathlib import Path

locust.stats.CSV_STATS_INTERVAL_SEC = 1 # default is 1 second
locust.stats.CSV_STATS_FLUSH_INTERVAL_SEC = 10 # Determines how often the datais flushed to disk, default is 10 seconds 

# api setting
HOST = "http://0.0.0.0:8080"
URL = "/target_sentiment"
TEST_INPUT = {
    "content": "## Headline ##\n大家最近買左D乜 分享下?\n## Content ##\n引用:\n原帖由 綠茶寶寶 於 2018-12-31 08:40 PM 發表\n買左3盒胭脂\nFit me 胭脂睇youtuber推介話好用，用完覺得麻麻\n原來fit me麻麻 我買左YSL 支定妝噴霧 用完覺得無想像咁好\n\n", 
    "start_ind": 141, 
    "end_ind": 144
}

class Testlen(TaskSet):
    @task
    def test(self):
        response = self.client.post(url=URL, json=TEST_INPUT)
        print("Response status code:", response.status_code)
        
class HttpRequester(HttpUser):
    tasks = [Testlen]

def runTest(u, r, file, t):
    import subprocess
    cmd = "locust -f locustfile.py --host={} --headless --csv={} --csv-full-history -u{} -r {} -t {}".format(HOST, file, u, r, t)
    # print(os.)
    # subprocess.call("pwd")
    subprocess.call(cmd, shell=True)
         
if __name__ == '__main__':
    # normal test
    u = 1
    r = 1
    file = "normal_test"
    t = "1m"
    runTest(u, r, file, t)
    # stress test
    u_list = [100, 200]
    r = 1
    t = "1m"
    for u in u_list:
        file = "stress_test_u{}".format(u)
        runTest(u, r, file, t)