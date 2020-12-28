import time
import string
import sys
import subprocess

def get_cpumem(file_name):
    d = [i for i in subprocess.getoutput(f"ps aux | grep {file_name}").split("\n")]
    cpu_usg = sum([float(x.split()[2]) for x in d])
    mem_usg = sum([float(x.split()[3]) for x in d])
    return (cpu_usg, mem_usg) if d else None

if __name__ == '__main__':
    if not len(sys.argv) == 2:
        print("usage: %s PID" % sys.argv[0])
        exit(2)

    file_name = "serve_{}.py".format(sys.argv[1])

    with open(f"./cpu_mem_{sys.argv[1]}.txt", "w+") as f:
        f.write("%CPU\t%MEM") 
        f.write("\n")
        try:
            while True:
                x, y = get_cpumem(file_name)
                f.write(("%.2f\t%.2f" % (x, y)))
                f.write("\n")
                time.sleep(0.5)
        except KeyboardInterrupt:
            exit(0)