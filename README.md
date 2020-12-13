# Carpoel-at-age-three-DQN-algorithm

## 本代码基于AIStudio环境运行！！！
深度学习入门 | 三岁在飞桨带你入门深度学习—Carpoel，利用PARL复现基于神经网络与DQN算法（真的是0基础）
# 深度学习入门 | 三岁在飞桨带你入门深度学习—Carpoel，利用PARL复现基于神经网络与DQN算法（真的是0基础）
大家好，这里是三岁，众所周知三岁是编程届小白，为了给大家贡献一个“爬山”的模板，
三岁利用最基础的深度学习“hello world”项目给大家解析，及做示范。
三岁老规矩，白话，简单，入门，基础
如果有什么不准确，不正确的地方希望大家可以提出来！
（代码源于[强化学习7日打卡营-世界冠军带你从零实践>PARL强化学习公开课Lesson3_DQN](https://aistudio.baidu.com/aistudio/projectdetail/569647)）
* 以下项目适用于CPU环境
## 参考资料
* B站视频地址：[https://www.bilibili.com/video/bv1v54y1v7Qf](https://www.bilibili.com/video/bv1v54y1v7Qf)
* AI 社区文章地址：[https://ai.baidu.com/forum/topic/show/962531](https://ai.baidu.com/forum/topic/show/962531)
* CSDN文章地址：[https://editor.csdn.net/md?articleId=107393006](https://editor.csdn.net/md?articleId=107393006)
* 三岁推文地址：[https://mp.weixin.qq.com/s/6-6RR0XuvTNuXKhX7fFXaQ](https://mp.weixin.qq.com/s/6-6RR0XuvTNuXKhX7fFXaQ)
* 参考论文：[https://www.nature.com/articles/nature14236](https://www.nature.com/articles/nature14236)
* DQNgithub地址：[https://github.com/PaddlePaddle/PARL/tree/develop/examples](https://github.com/PaddlePaddle/PARL/tree/develop/examples)
* 参考视频：[https://www.bilibili.com/video/BV1yv411i7xd?p=12](https://www.bilibili.com/video/BV1yv411i7xd?p=12)
* Carpoel参考资料：[https://gym.openai.com/envs/CartPole-v1/](https://gym.openai.com/envs/CartPole-v1/)
* PARL官方地址：[https://github.com/PaddlePaddle/PARL](https://github.com/PaddlePaddle/PARL)

那么我们接下来就开始爬山吧，记得唱着小白船然后录像呦，三岁会帮你调整一下jio的位置的【滑稽】


### 环境预设
根据实际情况把AI Studio的环境进行修改使其更加符合代码的运行。


```python
!pip uninstall -y parl  # 说明：AIStudio预装的parl版本太老，容易跟其他库产生兼容性冲突，建议先卸载
!pip uninstall -y pandas scikit-learn # 提示：在AIStudio中卸载这两个库再import parl可避免warning提示，不卸载也不影响parl的使用

!pip install gym
!pip install paddlepaddle==1.6.3
!pip install parl==1.3.1

# 建议下载paddle系列产品时添加百度源 -i https://mirror.baidu.com/pypi/simple
# python -m pip install paddlepaddle -i https://mirror.baidu.com/pypi/simple
# pip install parl==1.3.1  -i https://mirror.baidu.com/pypi/simple

# 说明：安装日志中出现两条红色的关于 paddlehub 和 visualdl 的 ERROR 与parl无关，可以忽略，不影响使用
```

    Uninstalling parl-1.1.2:
      Successfully uninstalled parl-1.1.2
    Uninstalling pandas-0.23.4:
      Successfully uninstalled pandas-0.23.4
    Uninstalling scikit-learn-0.20.0:
      Successfully uninstalled scikit-learn-0.20.0
    Looking in indexes: https://pypi.mirrors.ustc.edu.cn/simple/
    Requirement already satisfied: gym in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (0.12.1)
    Requirement already satisfied: requests>=2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym) (2.22.0)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym) (1.15.0)
    Requirement already satisfied: numpy>=1.10.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym) (1.16.4)
    Requirement already satisfied: pyglet>=1.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym) (1.4.5)
    Requirement already satisfied: scipy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from gym) (1.3.0)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.0->gym) (2019.9.11)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.0->gym) (3.0.4)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.0->gym) (2.8)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.0->gym) (1.25.6)
    Requirement already satisfied: future in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pyglet>=1.2.0->gym) (0.18.0)
    Looking in indexes: https://pypi.mirrors.ustc.edu.cn/simple/
    Collecting paddlepaddle==1.6.3
    [?25l  Downloading https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/96/28/e72bebb3c9b3d98eb9b15d9f6d85150f3cbd63e695e59882ff9f04846686/paddlepaddle-1.6.3-cp37-cp37m-manylinux1_x86_64.whl (90.9MB)
    [K     |████████████████████████████████| 90.9MB 488kB/s eta 0:00:011    |███████████████▌                | 44.0MB 481kB/s eta 0:01:38
    [?25hRequirement already satisfied: nltk; python_version >= "3.5" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (3.4.5)
    Requirement already satisfied: prettytable in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (0.7.2)
    Requirement already satisfied: six in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (1.15.0)
    Requirement already satisfied: protobuf>=3.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (3.10.0)
    Requirement already satisfied: numpy>=1.12; python_version >= "3.5" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (1.16.4)
    Requirement already satisfied: scipy; python_version >= "3.5" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (1.3.0)
    Requirement already satisfied: objgraph in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (3.4.1)
    Requirement already satisfied: decorator in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (4.4.0)
    Requirement already satisfied: matplotlib; python_version >= "3.6" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (2.2.3)
    Requirement already satisfied: Pillow in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (7.1.2)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (4.1.1.26)
    Requirement already satisfied: requests>=2.20.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (2.22.0)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (5.1.2)
    Requirement already satisfied: funcsigs in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (1.0.2)
    Requirement already satisfied: rarfile in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (3.1)
    Requirement already satisfied: graphviz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlepaddle==1.6.3) (0.13)
    Requirement already satisfied: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from protobuf>=3.1.0->paddlepaddle==1.6.3) (41.4.0)
    Requirement already satisfied: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib; python_version >= "3.6"->paddlepaddle==1.6.3) (0.10.0)
    Requirement already satisfied: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib; python_version >= "3.6"->paddlepaddle==1.6.3) (1.1.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib; python_version >= "3.6"->paddlepaddle==1.6.3) (2019.3)
    Requirement already satisfied: python-dateutil>=2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib; python_version >= "3.6"->paddlepaddle==1.6.3) (2.8.0)
    Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib; python_version >= "3.6"->paddlepaddle==1.6.3) (2.4.2)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.20.0->paddlepaddle==1.6.3) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.20.0->paddlepaddle==1.6.3) (3.0.4)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.20.0->paddlepaddle==1.6.3) (2019.9.11)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests>=2.20.0->paddlepaddle==1.6.3) (1.25.6)
    Installing collected packages: paddlepaddle
      Found existing installation: paddlepaddle 1.8.0
        Uninstalling paddlepaddle-1.8.0:
          Successfully uninstalled paddlepaddle-1.8.0
    Successfully installed paddlepaddle-1.6.3
    Looking in indexes: https://pypi.mirrors.ustc.edu.cn/simple/
    Collecting parl==1.3.1
    [?25l  Downloading https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/62/79/590af38a920792c71afb73fad7583967928b4d0ba9fca76250d935c7fda8/parl-1.3.1-py2.py3-none-any.whl (521kB)
    [K     |████████████████████████████████| 522kB 17.8MB/s eta 0:00:01
    [?25hRequirement already satisfied: tb-nightly==1.15.0a20190801 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from parl==1.3.1) (1.15.0a20190801)
    Requirement already satisfied: visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from parl==1.3.1) (2.0.0b4)
    Requirement already satisfied: pyzmq==18.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from parl==1.3.1) (18.0.1)
    Requirement already satisfied: flask>=1.0.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from parl==1.3.1) (1.1.1)
    Collecting flask-cors (from parl==1.3.1)
      Downloading https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/78/38/e68b11daa5d613e3a91e4bf3da76c94ac9ee0d9cd515af9c1ab80d36f709/Flask_Cors-3.0.8-py2.py3-none-any.whl
    Requirement already satisfied: click in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from parl==1.3.1) (7.0)
    Collecting psutil>=5.6.2 (from parl==1.3.1)
    [?25l  Downloading https://mirrors.tuna.tsinghua.edu.cn/pypi/web/packages/aa/3e/d18f2c04cf2b528e18515999b0c8e698c136db78f62df34eee89cee205f1/psutil-5.7.2.tar.gz (460kB)
    [K     |████████████████████████████████| 460kB 52.1MB/s eta 0:00:01
    [?25hRequirement already satisfied: tensorboardX==1.8 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from parl==1.3.1) (1.8)
    Requirement already satisfied: pyarrow==0.13.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from parl==1.3.1) (0.13.0)
    Requirement already satisfied: scipy>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from parl==1.3.1) (1.3.0)
    Requirement already satisfied: termcolor>=1.1.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from parl==1.3.1) (1.1.0)
    Requirement already satisfied: cloudpickle==1.2.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from parl==1.3.1) (1.2.1)
    Requirement already satisfied: absl-py>=0.4 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from tb-nightly==1.15.0a20190801->parl==1.3.1) (0.8.1)
    Requirement already satisfied: six>=1.10.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from tb-nightly==1.15.0a20190801->parl==1.3.1) (1.15.0)
    Requirement already satisfied: grpcio>=1.6.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from tb-nightly==1.15.0a20190801->parl==1.3.1) (1.26.0)
    Requirement already satisfied: wheel>=0.26; python_version >= "3" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from tb-nightly==1.15.0a20190801->parl==1.3.1) (0.33.6)
    Requirement already satisfied: setuptools>=41.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from tb-nightly==1.15.0a20190801->parl==1.3.1) (41.4.0)
    Requirement already satisfied: werkzeug>=0.11.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from tb-nightly==1.15.0a20190801->parl==1.3.1) (0.16.0)
    Requirement already satisfied: markdown>=2.6.8 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from tb-nightly==1.15.0a20190801->parl==1.3.1) (3.1.1)
    Requirement already satisfied: protobuf>=3.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from tb-nightly==1.15.0a20190801->parl==1.3.1) (3.10.0)
    Requirement already satisfied: numpy>=1.12.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from tb-nightly==1.15.0a20190801->parl==1.3.1) (1.16.4)
    Requirement already satisfied: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (2.22.0)
    Requirement already satisfied: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (1.21.0)
    Requirement already satisfied: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (3.8.2)
    Requirement already satisfied: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (1.0.0)
    Requirement already satisfied: opencv-python in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (4.1.1.26)
    Requirement already satisfied: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (7.1.2)
    Requirement already satisfied: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.0.4->parl==1.3.1) (1.1.0)
    Requirement already satisfied: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.0.4->parl==1.3.1) (2.10.3)
    Requirement already satisfied: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (2019.9.11)
    Requirement already satisfied: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (2.8)
    Requirement already satisfied: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (3.0.4)
    Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (1.25.6)
    Requirement already satisfied: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (0.23)
    Requirement already satisfied: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (1.4.10)
    Requirement already satisfied: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (1.3.4)
    Requirement already satisfied: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (2.0.1)
    Requirement already satisfied: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (1.3.0)
    Requirement already satisfied: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (16.7.9)
    Requirement already satisfied: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (0.10.0)
    Requirement already satisfied: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (5.1.2)
    Requirement already satisfied: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (0.6.1)
    Requirement already satisfied: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (2.6.0)
    Requirement already satisfied: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (2.2.0)
    Requirement already satisfied: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (2019.3)
    Requirement already satisfied: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (2.8.0)
    Requirement already satisfied: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.0.4->parl==1.3.1) (1.1.1)
    Requirement already satisfied: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (0.6.0)
    Requirement already satisfied: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl>=2.0.0b; python_version >= "3" and platform_system == "Linux"->parl==1.3.1) (7.2.0)
    Building wheels for collected packages: psutil
      Building wheel for psutil (setup.py) ... [?25ldone
    [?25h  Created wheel for psutil: filename=psutil-5.7.2-cp37-cp37m-linux_x86_64.whl size=268459 sha256=ef019c256f341f219f0260fa4acfe2863e8f0bf1f96ed95b3f910a2a4c44a74c
      Stored in directory: /home/aistudio/.cache/pip/wheels/a8/74/a2/9f54383a7c48678163f965a5d2f4acb794417e60ab0d7351f8
    Successfully built psutil
    Installing collected packages: flask-cors, psutil, parl
    Successfully installed flask-cors-3.0.8 parl-1.3.1 psutil-5.7.2


### Step2  导入依赖
如果依赖导入失败有可能没有下载第三方库可以在此前加上代码块
！pip instudio 第三方库名 

如果安装失败可以加上镜像

![](https://ai-studio-static-online.cdn.bcebos.com/8ee07cbb24874e96bdac93b07e3a22e3b6b465981c904129990af727701bd0b0)
```
镜像源地址：

百度：https://mirror.baidu.com/pypi/simple

清华：https://pypi.tuna.tsinghua.edu.cn/simple

阿里云：http://mirrors.aliyun.com/pypi/simple/

中国科技大学 https://pypi.mirrors.ustc.edu.cn/simple/

华中理工大学：http://pypi.hustunique.com/

山东理工大学：http://pypi.sdutlinux.org/

豆瓣：http://pypi.douban.com/simple/

例：!pip instudio jieba -i https://mirror.baidu.com/pypi/simple
```


```python
import parl
from parl import layers
import paddle.fluid as fluid
import copy
import numpy as np
import os
import gym
from parl.utils import logger
```

# 流程解析
![](https://ai-studio-static-online.cdn.bcebos.com/25f4800e0c3b44bc9ddc908a4d3dba1793dc74f0f5a045e799771f0466b88e5b)


### Step3 设置超参数


```python
LEARN_FREQ = 5 # 训练频率，不需要每一个step都learn，攒一些新增经验后再learn，提高效率
MEMORY_SIZE = 20000    # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再开启训练
BATCH_SIZE = 32   # 每次给agent learn的数据数量，从replay memory随机里sample一批数据出来
LEARNING_RATE = 0.001 # 学习率
GAMMA = 0.99 # reward 的衰减因子，一般取 0.9 到 0.999 不等
```

# 搭建PARL
什么是PARL呢？让我们看看官方文档怎么说

![飞桨官网描述](https://ai-studio-static-online.cdn.bcebos.com/1f72929e036549d493dee85c2834b8a98e330c4ac6a447f08b647cab09e12bfa)         飞桨官网描述
![PARL GitHub描述](https://ai-studio-static-online.cdn.bcebos.com/9a19623bdf41411ca07c089d3cf0f64752e0eb51a58a47e694bc4e4614079078)       PARL GitHub描述

* PARL主要是基于Model、Algorithm、Agent三个代码块来实现，其中Model和Agent是用户自定义操作。
* Model：是网络结构：要三层网络还是四层网络都是在Model中去定义（下文是三层的网络结构）
* Agent：是PARL与环境的一个接口，通过对模板的修改即可运用到各个不同的环境中去。

* 至于Algorithm是内部已经封装好了的，直接加入参数运行即可，主要是算法的模块的展现


### Step4 搭建Model、Algorithm、Agent架构
* `Agent`把产生的数据传给`algorithm`，`algorithm`根据`model`的模型结构计算出`Loss`，使用`SGD`或者其他优化器不断的优化，`PARL`这种架构可以很方便的应用在各类深度强化学习问题中。

#### （1）Model
* `Model`用来定义前向(`Forward`)网络，用户可以自由的定制自己的网络结构。


```python
class Model(parl.Model):
    def __init__(self, act_dim):
        hid1_size = 128
        hid2_size = 128
        # 3层全连接网络
        self.fc1 = layers.fc(size=hid1_size, act='relu')
        self.fc2 = layers.fc(size=hid2_size, act='relu')
        self.fc3 = layers.fc(size=act_dim, act=None)

    def value(self, obs):
        # 定义网络
        # 输入state，输出所有action对应的Q，[Q(s,a1), Q(s,a2), Q(s,a3)...]
        h1 = self.fc1(obs)
        h2 = self.fc2(h1)
        Q = self.fc3(h2)
        return Q
```

## Q算法的“藏身之地”
###  优势
DQN算法较普通算法在经验回放和固定Q目标有了较大的改进

* 1、经验回放：他充分利用了off-colicp的优势，通过训练把结果（成绩）存入Q表格，然后随机从表格中取出一条结果进行优化。这样子一方面可以：减少样本之间的关联性另一方面：提高样本的利用率
注：训练结果会存进Q表格，当Q表格满了以后，存进来的数据会把最早存进去的数据“挤出去”（弹出）
* 2、固定Q目标他解决了算法更新不平稳的问题。
和监督学习做比较，监督学习的最终值要逼近实际结果，这个结果是固定的，但是我们的DQN却不是，他的目标值是经过神经网络以后的一个值，那么这个值是变动的不好拟合，怎么办，DQN团队想到了一个很好的办法，让这个值在一定时间里面保持不变，这样子这个目标就可以确定了，然后目标值更新以后更加接近实际结果，可以更好的进行训练。

#### （2）Algorithm
* `Algorithm` 定义了具体的算法来更新前向网络(`Model`)，也就是通过定义损失函数来更新`Model`，和算法相关的计算都放在`algorithm`中。




```python
# from parl.algorithms import DQN # 也可以直接从parl库中导入DQN算法

class DQN(parl.Algorithm):
    def __init__(self, model, act_dim=None, gamma=None, lr=None):
        """ DQN algorithm
        
        Args:
            model (parl.Model): 定义Q函数的前向网络结构
            act_dim (int): action空间的维度，即有几个action
            gamma (float): reward的衰减因子
            lr (float): learning rate 学习率.
        """
        self.model = model
        self.target_model = copy.deepcopy(model)

        assert isinstance(act_dim, int)
        assert isinstance(gamma, float)
        assert isinstance(lr, float)
        self.act_dim = act_dim
        self.gamma = gamma
        self.lr = lr

    def predict(self, obs):
        """ 使用self.model的value网络来获取 [Q(s,a1),Q(s,a2),...]
        """
        return self.model.value(obs)

    def learn(self, obs, action, reward, next_obs, terminal):
        """ 使用DQN算法更新self.model的value网络
        """
        # 从target_model中获取 max Q' 的值，用于计算target_Q
        next_pred_value = self.target_model.value(next_obs)
        best_v = layers.reduce_max(next_pred_value, dim=1)
        best_v.stop_gradient = True  # 阻止梯度传递
        terminal = layers.cast(terminal, dtype='float32')
        target = reward + (1.0 - terminal) * self.gamma * best_v

        pred_value = self.model.value(obs)  # 获取Q预测值
        # 将action转onehot向量，比如：3 => [0,0,0,1,0]
        action_onehot = layers.one_hot(action, self.act_dim)
        action_onehot = layers.cast(action_onehot, dtype='float32')
        # 下面一行是逐元素相乘，拿到action对应的 Q(s,a)
        # 比如：pred_value = [[2.3, 5.7, 1.2, 3.9, 1.4]], action_onehot = [[0,0,0,1,0]]
        #  ==> pred_action_value = [[3.9]]
        pred_action_value = layers.reduce_sum(
            layers.elementwise_mul(action_onehot, pred_value), dim=1)

        # 计算 Q(s,a) 与 target_Q的均方差，得到loss
        cost = layers.square_error_cost(pred_action_value, target)
        cost = layers.reduce_mean(cost)
        optimizer = fluid.optimizer.Adam(learning_rate=self.lr)  # 使用Adam优化器
        optimizer.minimize(cost)
        return cost

    def sync_target(self):
        """ 把 self.model 的模型参数值同步到 self.target_model
        """
        self.model.sync_weights_to(self.target_model)

```

#### （3）Agent
* `Agent` 负责算法与环境的交互，在交互过程中把生成的数据提供给`Algorithm`来更新模型(`Model`)，数据的预处理流程也一般定义在这里。


```python
class Agent(parl.Agent):
    def __init__(self,
                 algorithm,
                 obs_dim,
                 act_dim,
                 e_greed=0.1,
                 e_greed_decrement=0):
        assert isinstance(obs_dim, int)
        assert isinstance(act_dim, int)
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        super(Agent, self).__init__(algorithm)

        self.global_step = 0
        self.update_target_steps = 200  # 每隔200个training steps再把model的参数复制到target_model中

        self.e_greed = e_greed  # 有一定概率随机选取动作，探索
        self.e_greed_decrement = e_greed_decrement  # 随着训练逐步收敛，探索的程度慢慢降低

    def build_program(self):
        self.pred_program = fluid.Program()
        self.learn_program = fluid.Program()

        with fluid.program_guard(self.pred_program):  # 搭建计算图用于 预测动作，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            self.value = self.alg.predict(obs)

        with fluid.program_guard(self.learn_program):  # 搭建计算图用于 更新Q网络，定义输入输出变量
            obs = layers.data(
                name='obs', shape=[self.obs_dim], dtype='float32')
            action = layers.data(name='act', shape=[1], dtype='int32')
            reward = layers.data(name='reward', shape=[], dtype='float32')
            next_obs = layers.data(
                name='next_obs', shape=[self.obs_dim], dtype='float32')
            terminal = layers.data(name='terminal', shape=[], dtype='bool')
            self.cost = self.alg.learn(obs, action, reward, next_obs, terminal)

    def sample(self, obs):
        sample = np.random.rand()  # 产生0~1之间的小数
        if sample < self.e_greed:
            act = np.random.randint(self.act_dim)  # 探索：每个动作都有概率被选择
        else:
            act = self.predict(obs)  # 选择最优动作
        self.e_greed = max(
            0.01, self.e_greed - self.e_greed_decrement)  # 随着训练逐步收敛，探索的程度慢慢降低
        return act

    def predict(self, obs):  # 选择最优动作
        obs = np.expand_dims(obs, axis=0)
        pred_Q = self.fluid_executor.run(
            self.pred_program,
            feed={'obs': obs.astype('float32')},
            fetch_list=[self.value])[0]
        pred_Q = np.squeeze(pred_Q, axis=0)
        act = np.argmax(pred_Q)  # 选择Q最大的下标，即对应的动作
        return act

    def learn(self, obs, act, reward, next_obs, terminal):
        # 每隔200个training steps同步一次model和target_model的参数
        if self.global_step % self.update_target_steps == 0:
            self.alg.sync_target()
        self.global_step += 1

        act = np.expand_dims(act, -1)
        feed = {
            'obs': obs.astype('float32'),
            'act': act.astype('int32'),
            'reward': reward,
            'next_obs': next_obs.astype('float32'),
            'terminal': terminal
        }
        cost = self.fluid_executor.run(
            self.learn_program, feed=feed, fetch_list=[self.cost])[0]  # 训练一次网络
        return cost
```

### Step5 ReplayMemory
* 经验池：用于存储多条经验，实现 经验回放。
![](https://ai-studio-static-online.cdn.bcebos.com/340817e194974c24ac63dd569fba336cd500d120729e4354825fb9c3501108ab)



```python
import random
import collections
import numpy as np


class ReplayMemory(object):
    def __init__(self, max_size):
        self.buffer = collections.deque(maxlen=max_size)

    # 增加一条经验到经验池中
    def append(self, exp):
        self.buffer.append(exp)

    # 从经验池中选取N条经验出来
    def sample(self, batch_size):
        mini_batch = random.sample(self.buffer, batch_size)
        obs_batch, action_batch, reward_batch, next_obs_batch, done_batch = [], [], [], [], []

        for experience in mini_batch:
            s, a, r, s_p, done = experience
            obs_batch.append(s)
            action_batch.append(a)
            reward_batch.append(r)
            next_obs_batch.append(s_p)
            done_batch.append(done)

        return np.array(obs_batch).astype('float32'), \
            np.array(action_batch).astype('float32'), np.array(reward_batch).astype('float32'),\
            np.array(next_obs_batch).astype('float32'), np.array(done_batch).astype('float32')

    def __len__(self):
        return len(self.buffer)

```

### Step6 Training && Test（训练&&测试）
![](https://ai-studio-static-online.cdn.bcebos.com/3b9f168ee09a46dc9962decb0d2a60f7d35196d459824b9f923b163a0b4bbe4e)
![](https://ai-studio-static-online.cdn.bcebos.com/4ffa39f252744a3791c04c5d8381824d4f7447a592d4409da658045061bcd0b9)
* 训练和评估的一个模块


```python
# 训练一个episode
def run_episode(env, agent, rpm):
    total_reward = 0
    obs = env.reset()
    step = 0
    while True:
        step += 1
        action = agent.sample(obs)  # 采样动作，所有动作都有概率被尝试到
        next_obs, reward, done, _ = env.step(action)
        rpm.append((obs, action, reward, next_obs, done))

        # train model
        if (len(rpm) > MEMORY_WARMUP_SIZE) and (step % LEARN_FREQ == 0):
            (batch_obs, batch_action, batch_reward, batch_next_obs,
             batch_done) = rpm.sample(BATCH_SIZE)
            train_loss = agent.learn(batch_obs, batch_action, batch_reward,
                                     batch_next_obs,
                                     batch_done)  # s,a,r,s',done

        total_reward += reward
        obs = next_obs
        if done:
            break
    return total_reward


# 评估 agent, 跑 5 个episode，总reward求平均
def evaluate(env, agent, render=False):
    eval_reward = []
    for i in range(5):
        obs = env.reset()
        episode_reward = 0
        while True:
            action = agent.predict(obs)  # 预测动作，只选最优动作
            obs, reward, done, _ = env.step(action)
            episode_reward += reward
            if render:
                env.render()
            if done:
                break
        eval_reward.append(episode_reward)
    return np.mean(eval_reward)

```

### Step7 创建环境和Agent，创建经验池，启动训练，保存模型

# 主函数
* 讲不清，理还乱，看一波图解，冷静冷静！！！
![](https://ai-studio-static-online.cdn.bcebos.com/dab5d244b98c41108a84e37b79db0009109ed32d4d734ca7b04b68e2e301da09)
![](https://ai-studio-static-online.cdn.bcebos.com/a297e56a6de045169e4a04391c22aa80bf7cc9d9608446fd97bef4dcd16431c8)
![](https://ai-studio-static-online.cdn.bcebos.com/8bc3b4cd1913479480baa286b120f45b49cb8ac7c36645df8423d873a7bd6d07)



```python
env = gym.make('CartPole-v0')  # CartPole-v0: 预期最后一次评估总分 > 180（最大值是200）
action_dim = env.action_space.n  # CartPole-v0: 2
obs_shape = env.observation_space.shape  # CartPole-v0: (4,)

rpm = ReplayMemory(MEMORY_SIZE)  # DQN的经验回放池

# 根据parl框架构建agent
model = Model(act_dim=action_dim)
algorithm = DQN(model, act_dim=action_dim, gamma=GAMMA, lr=LEARNING_RATE)
agent = Agent(
    algorithm,
    obs_dim=obs_shape[0],
    act_dim=action_dim,
    e_greed=0.1,  # 有一定概率随机选取动作，探索
    e_greed_decrement=1e-6)  # 随着训练逐步收敛，探索的程度慢慢降低

# 加载模型
# save_path = './dqn_model.ckpt'
# agent.restore(save_path)

# 先往经验池里存一些数据，避免最开始训练的时候样本丰富度不够
while len(rpm) < MEMORY_WARMUP_SIZE:
    run_episode(env, agent, rpm)

max_episode = 2000

# 开始训练
episode = 0
while episode < max_episode:  # 训练max_episode个回合，test部分不计算入episode数量
    # train part
    for i in range(0, 50):
        total_reward = run_episode(env, agent, rpm)
        episode += 1

    # test part
    eval_reward = evaluate(env, agent, render=False)  # render=True 查看显示效果
    logger.info('episode:{}    e_greed:{}   test_reward:{}'.format(
        episode, agent.e_greed, eval_reward))

# 训练结束，保存模型
save_path = './dqn_model.ckpt'
agent.save(save_path)
```

    [32m[07-27 18:19:17 MainThread @machine_info.py:88][0m Cannot find available GPU devices, using CPU now.
    [32m[07-27 18:19:17 MainThread @machine_info.py:88][0m Cannot find available GPU devices, using CPU now.
    [32m[07-27 18:19:18 MainThread @machine_info.py:88][0m Cannot find available GPU devices, using CPU now.
    [32m[07-27 18:19:19 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:50    e_greed:0.09930599999999931   test_reward:11.0
    [32m[07-27 18:19:21 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:100    e_greed:0.09877699999999878   test_reward:9.4
    [32m[07-27 18:19:23 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:150    e_greed:0.09828699999999829   test_reward:9.4
    [32m[07-27 18:19:24 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:200    e_greed:0.09781299999999782   test_reward:9.2
    [32m[07-27 18:19:26 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:250    e_greed:0.09733199999999734   test_reward:9.2
    [32m[07-27 18:19:27 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:300    e_greed:0.09683499999999684   test_reward:9.6
    [32m[07-27 18:19:29 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:350    e_greed:0.09634199999999635   test_reward:9.6
    [32m[07-27 18:19:30 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:400    e_greed:0.09583999999999585   test_reward:9.6
    [32m[07-27 18:19:32 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:450    e_greed:0.09533799999999534   test_reward:11.0
    [32m[07-27 18:19:34 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:500    e_greed:0.09476499999999477   test_reward:9.4
    [32m[07-27 18:19:37 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:550    e_greed:0.09395199999999396   test_reward:44.4
    [32m[07-27 18:19:43 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:600    e_greed:0.0921909999999922   test_reward:21.2
    [32m[07-27 18:20:05 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:650    e_greed:0.08648299999998649   test_reward:186.0
    [32m[07-27 18:20:39 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:700    e_greed:0.07723099999997723   test_reward:199.0
    [32m[07-27 18:21:12 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:750    e_greed:0.06832399999996833   test_reward:188.6
    [32m[07-27 18:21:42 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:800    e_greed:0.06003099999996003   test_reward:124.6
    [32m[07-27 18:22:09 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:850    e_greed:0.05293599999995294   test_reward:165.6
    [32m[07-27 18:22:31 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:900    e_greed:0.04666999999994667   test_reward:110.2
    [32m[07-27 18:22:56 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:950    e_greed:0.040123999999940124   test_reward:111.0
    [32m[07-27 18:23:21 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1000    e_greed:0.03343099999993343   test_reward:133.6
    [32m[07-27 18:23:47 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1050    e_greed:0.026627999999926627   test_reward:130.0
    [32m[07-27 18:24:08 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1100    e_greed:0.02101999999992102   test_reward:130.0
    [32m[07-27 18:24:31 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1150    e_greed:0.015293999999915868   test_reward:179.4
    [32m[07-27 18:25:03 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1200    e_greed:0.01   test_reward:183.0
    [32m[07-27 18:25:33 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1250    e_greed:0.01   test_reward:130.8
    [32m[07-27 18:25:59 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1300    e_greed:0.01   test_reward:172.8
    [32m[07-27 18:26:25 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1350    e_greed:0.01   test_reward:152.2
    [32m[07-27 18:26:49 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1400    e_greed:0.01   test_reward:127.8
    [32m[07-27 18:27:19 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1450    e_greed:0.01   test_reward:182.0
    [32m[07-27 18:27:52 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1500    e_greed:0.01   test_reward:120.6
    [32m[07-27 18:28:25 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1550    e_greed:0.01   test_reward:133.6
    [32m[07-27 18:28:45 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1600    e_greed:0.01   test_reward:14.0
    [32m[07-27 18:28:49 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1650    e_greed:0.01   test_reward:15.8
    [32m[07-27 18:29:09 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1700    e_greed:0.01   test_reward:164.2
    [32m[07-27 18:29:47 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1750    e_greed:0.01   test_reward:200.0
    [32m[07-27 18:30:26 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1800    e_greed:0.01   test_reward:200.0
    [32m[07-27 18:31:03 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1850    e_greed:0.01   test_reward:182.4
    [32m[07-27 18:31:40 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1900    e_greed:0.01   test_reward:193.8
    [32m[07-27 18:32:17 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:1950    e_greed:0.01   test_reward:193.4
    [32m[07-27 18:32:53 MainThread @<ipython-input-9-eeec60765889>:38][0m episode:2000    e_greed:0.01   test_reward:170.2


![](https://ai-studio-static-online.cdn.bcebos.com/94a7460d1da44164a7805c5680a2779689b1dc4c78494104aa3d2bda075411fa)


# 运行经历
这一串代码是从课件上copy下来的，所以里面的参数及数据都是已经调整好了的，但是在线下跑数据发现每一次跑也并非到最后都是200也就是每一次的结果都是不一定的，然后我开了显示模式，里面的测试画面和结果会被显示和打印，前面的速度会比后面的快，因为分数低时间自然就短了。
更据这段时间对深度学习的学习，基本上代码运行没有问题后那么调整几个超参基本上可以得到一个比较好的拟合效果。


# 心得体会
本次“爬山”，发现之前的一些盲点在本次有了解决，，之前对深度学习、PARL、算法等都有了最新的认识，虽然还是那个小白但是认识提上去了，以后还是可以更加努力的去奋斗的，这就是传说中的回头看的时候就知道自己以前是多么的无知了。

# 这里是三岁，请大家多多指教啊！
