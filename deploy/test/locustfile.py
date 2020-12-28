from locust import HttpUser, task, TaskSet, between
import locust.stats
import os
from pathlib import Path
import argparse
import subprocess
import time

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
        # print("Response status code:", response.status_code)
        
class HttpRequester(HttpUser):
    wait_time = between(1, 2)
    tasks = [Testlen]

def runTest(u, r, file, t):
    """https://docs.locust.io/en/stable/configuration.html
    Args:
        u (int): The number of concurrent users
        r (int): The rate per second in which users are spawned.
        t (str): Stop after the specified amount of time. (1m: 1 minute, 60s: 60 seconds)
    """
    import subprocess
    cmd = "locust -f locustfile.py --host={} --headless --csv={} --csv-full-history -u{} -r {} -t {}".format(HOST, file, u, r, t)
    subprocess.call(cmd, shell=True)
         
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", type=int, default=1)
    parser.add_argument("-u", type=int, default=200)
    parser.add_argument("--max_wait_time", type=int, default=0)
    parser.add_argument("--min_wait_time", type=int, default=0)
    parser.add_argument("--api", type=str, default="")
    args = parser.parse_args()

    

    # single user test
    file = "normal_test_u1_r1_t10s_{}".format(args.api)
    runTest(1, 1, file, "10s")

    # stress test
    u = args.u
    r = args.r
    t = str(int(u / r + 60)) + "s"

    HttpRequester = between(args.min_wait_time, args.max_wait_time)
    
    file = "stress_test_u{}_r{}_t{}_{}".format(u, r, t, args.api)

    subprocess.Popen("python ./log_cpu_mem.py '{}'".format(args.api, str(int(u / r + 60) + 20)), shell=True, close_fds=False)
    print("started logging")
    time.sleep(10)
    print("starting load test")
    runTest(u, r, file, t)
    time.sleep(10)