# vix_option

---
## 使用须知
- run convert_data.py to prepare data
- gen data/qlib_data/instruments/trable.txt
  - copy from all.txt
  - instruments: just remain VIX_1M-VIX_6M

## 环境配置

- Python版本：3.8
- 环境及依赖包管理器：[PDM](https://pdm.fming.dev/latest/)

### PDM安装

**初次使用PDM需要为当前系统全局安装PDM工具，如果已安装过PDM，即可跳过本节**

安装步骤：

1. 安装前确保系统全局的python版本为3.8，并且已添加到系统环境变量中，各系统下安装全局Python的方法如下：
    - Windows：
        - [官方Python3.8安装包](https://www.python.org/ftp/python/3.8.10/python-3.8.10-amd64.exe)
        - 安装过程中选中“添加到当前用户PATH的选项”
    - Mac:
        - 建议使用[Homebrew](https://brew.sh/)安装，安装Homebrew命令：

          ```
          /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
          ```
        - 配置homebrew镜像：[清华大学homebrew镜像使用帮助](https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/)
        - 安装python3.8：

          ```
          brew install python@3.8
          ```
    - Linux(Ubuntu):
        - 为apt配置deadsnakes ppa

          ```
          sudo add-apt-repository ppa:deadsnakes/ppa
          sudo apt update
          ```
        - 使用apt安装：

          ```
          sudo apt install python3.8
          ```

2. 检查系统全局python版本
    1. 在系统终端中打印当前python的版本号
        - **注意：对于安装了Anaconda的Mac/Linux系统，首先确保Anaconda没有设置为打开终端自动激活环境**
        - Windows(CMD/Power Shell): ```python --version```
        - Mac/Linux: ```python3.8 -- version```
    2. 检查是否可正常打印出python的版本号，且版本号为3.8.x

3. 安装[pipx](https://pypa.github.io/pipx/)工具，用于管理系统全局依赖python的工具程序
    - Windows(CMD/Power Shell):
        ```
        python -m pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
        python -m pip install --user pipx
        python -m pipx ensurepath
        ```
    - Mac/Linux:
        ```
        python3.8 -m pip config set global.index-url https://mirrors.cloud.tencent.com/pypi/simple
        python3.8 -m pip install --user pipx
        python3.8 -m pipx ensurepath
        ```

4. 为系统安装PDM工具，***执行此步前重启终端***
    - Windows(CMD/Power Shell):
        ```
        python -m pipx install pdm
        ```
    - Mac/Linux:
        ```
        python3.8 -m pipx install pdm
        ```

5. 为PDM设置pypi镜像
    ```
    pdm config pypi.url https://mirrors.cloud.tencent.com/pypi/simple
    ```

### 工程环境初始化及依赖包管理

#### 运行环境初次配置步骤：
***如果已在当前项目下创建过虚拟环境，请先手动删除虚拟环境目录，例如`venv`目录***：
1. 在PyCharm终端中执行 ```pdm install```，如果正确配置了系统全局python的话，pdm会自动利用系统全局python为当前项目创建虚拟环境目录`.venv`，同时安装好所有依赖包。
2. 在PyCharm里配置解释器路径为：
   - Windows: `{项目根目录}/.venv/Scripts/python.exe`
   - Mac/Linux: `{项目根目录}/.venv/bin/python`
3. 在PyCharm终端中执行 ```pre-commit install```初始化pre-commit模块

#### 依赖包列表更新流程：
1. 对于程序运行所需的依赖包，使用```pdm add {包名} {包名} ... --save-exact```命令添加，以pandas和flask为例：
    ```
    pdm add pandas flask --save-exact
    ```
2. 对于开发过程中所需，但程序运行非必须的包可使用```pdm add -dG dev {包名} {包名} ... --save-exact```命令添加至开发环境依赖列表。例如`jupyter`、`black`等包都属于开发工具，算法程序在服务器上运行并不依赖它们。
    ```
    pdm add -dG dev jupyter black --save-exact
    ```
3. 依赖包添加完成后，`pdm`会更新`pdm.lock`和`pyproject.toml`文件，请提交到git中。

#### 服务器环境初始化：
1. 安装配置好系统全局的`pdm`工具
2. 终端中进入工程所在目录
3. 执行```pdm install --prod --no-lock --no-editable```初始化虚拟环境
4. 激活`{项目根目录}/.venv/`下的虚拟环境，执行命令即可

---

## 开发者须知

### git基本设置
为了使代码拥有线性的提交历史从而更加清晰明朗，请将`git pull`的默认配置改为`rebase`:
```
git config --global pull.rebase true
```

### 代码同步方法
在PyCharm中从`origin/feature-xxxx-all`分支合并到个人分支时，请选`Pull into feature-xxxx-{姓名简写} Using Rebase`选项

### Push操作流程
- 每次Push代码到远端的个人分支前，都建议执行一次与all分支的代码同步
- 如果远端all分支有更新，rebase后的本地个人分支commit历史会与远端个人分支存在差异，此时push会弹出警告对话框，选择push按钮旁下拉菜单中的`force push`即可

### 规范要求
#### 一般要求
除一些特殊情况之外，每次提交都要配合一个项目协同的事项。提交的信息采用如下格式：
```
#<项目协同事项编号> 本次提交的主题

本次提交的概述

[如有需要，本次提交的具体细节用bullet point罗列]
```

特殊情况指非常简单的任务，例如纠正个别拼写上的错误。这种情况下可以不加入项目协同事项的编号，只需简单说明本次提交的主题即可。

#### 核心代码要求
- 类名称一律用大写字母开头的`upper camel case`
- 方法名称一律用小写字母的`snake case`
- 变量名称一路用小写字母的`snake case`
- 变量名称在保证可读性的前提下尽量保持剪短，例如`number_of_observations`可以写作`num_observations`，但是不宜写作`n_obs`;又例如`underlying_price`可以写作`spot`, 但是不宜写作`S`
- 公共的类和方法必须要有`doc string`
- 公共的方法所有的变量和返回类型必须有`type annotation`

---

## 单元测试环境
- 公共的类和方法尽可能要有单元测试
- ***每次push到云端之前运行单元测试以保证代码的正确性***
- 单元测试框架：pytest
- 单元测试文件目录：`tests/`
- 启动全部测试命令：`pytest tests/`
- 如果使用PyCharm可创建一个`Run/Debug Configuration`:
  - 配置类型：`Python tests` -> `pytest`
  - Script Path: `{项目根目录}/tests`
  - Working directory: `{项目根目录}`

---

## 使用多进程跑optuna
- 先create_study, use run_model_params_optuna.py 中的 STORAGE_NAME, STUDY_NAME(可变更)
- run_model_params_optuna.py --if_load_study True
- 如果是rolling, rolling_exp 和 handler.pkl 需要分配UUID
- 在环境中跑多进程，需要注意显存占用