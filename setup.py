from setuptools import setup


required_packages = []
dependency_links = []
with open("requirements.txt", "r") as file:
      for ln in file.readlines():
            ln = ln.strip()
            print(ln)
            if ln.startswith("--find-links"):
                  dependency_links.append(ln.split(" ")[1])
            else:
                  required_packages.append(ln)


# assert(False)

setup(name='nlp_pipeline',
      version='1.0',
      description='',
      author='WisersAI NLP',
      author_email='',
      # dependency_links=["https://download.pytorch.org/whl/torch_stable.html"],
      # install_requires=required_packages,
      packages=["nlp_pipeline"],
)