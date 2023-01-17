from setuptools import setup


dev_packages = [
      "captum==0.4.1",
      "scikit-learn==0.23.1", 
      "tensorboard==2.8.0", 
      "seqeval", 
      "xlsxwriter", 
      "papermill", 
      "scrapbook"
]

skip_lines = [
      "--find-links https://download.pytorch.org/whl/torch_stable.html", 
      "torch==1.6.0+cu101"
]


required_packages = []
with open("requirements.txt", "r") as file:
      for ln in file.readlines():
            ln = ln.strip()
            if ln in skip_lines:
                  continue

            if ln not in dev_packages:
                  required_packages.append(ln)


setup(name='nlp_pipeline',
      version='1.0',
      description='',
      author='WisersAI NLP',
      author_email='',
      install_requires=required_packages,
      extras_require={
            "deploy": [], 
            "dev": dev_packages, 
      },
      packages=["nlp_pipeline"],
)