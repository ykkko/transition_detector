from setuptools import setup, find_packages


with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

with open("VERSION") as f:
    version = f.readline().rstrip()

setup(name='transition_detector',
      version=version,
      url='https://gitlab.mtsai.tk/ai/ml/CV/mts_media/transition_detector',
      license='LICENSE',
      author='Yury Kochnev',
      author_email='yvkochn2@mts.ru',
      description='Finds transitions such as cut, fadein and fadeout in a video',
      packages=find_packages(),
      long_description=long_description,
      long_description_content_type='text/markdown',
      zip_safe=False,
      install_requires=['numpy',
                        'opencv-python',
                        'tqdm'])
