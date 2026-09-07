# 第一次推送到 GitHub

准备状态：独立代码目录已整理，不包含原仓库历史、仿真资产或原项目其他修改。
本地 Git 可初始化为 `main`；首次提交和远端推送由下面步骤完成。
浏览器登录与终端 Git 认证是两套入口；浏览器已登录不保证终端能直接 push。

## 1. 建立空仓库

在已登录的 Chrome 打开 https://github.com/new ：

- Owner：选择你自己的账号，以页面显示的账号为准。
- Repository name：`asset-collision-spheres`（可以自行改名）。
- Description：`Material-aware collision sphere generation for 3D assets`。
- Visibility：第一版建议 Private；当前 LICENSE 为 pending，公开前先确定授权与许可证。
- 不初始化 README、.gitignore、License，本地已有这些文件。

点 Create repository 后，复制 Quick setup 中的 HTTPS 地址。
不要根据 `duudual/task2sim` 页面推断登录账号，那只是已打开仓库的所有者。

## 2. 在 VS Code 集成终端提交

先进入独立的 `asset-collision-spheres` 目录，不要在 task2sim 或 FastSim-Plugins 运行以下命令：

```bash
cd /你的项目目录/asset-collision-spheres
git init -b main
git status --short
git config user.name
git config user.email
```

如果后两项为空或不是你的身份，只设置当前仓库：

```bash
git config user.name "你的提交署名"
git config user.email "GitHub 中已验证的邮箱或个人 noreply 邮箱"
```

准备提交并核对内容：

```bash
git add .gitignore LICENSE README.md MANIFEST.in pyproject.toml constraints-tested.txt src tests examples docs
git diff --cached --stat
git diff --cached --check
git commit -m "Extract standalone asset collision sphere generation API"
```

`assets/`、`outputs/`、`.venv/`、缓存和构建文件由 `.gitignore` 排除。
首次提交应只含本独立包文件。

## 3. 连接并推送

将下一条命令的占位地址替换为刚从 GitHub 复制的真实 HTTPS 地址：

```bash
git remote add origin https://github.com/YOUR_ACCOUNT/asset-collision-spheres.git
git remote -v
git push -u origin main
```

刷新 GitHub 仓库页面，应能看到 README、src、tests 等内容。
若提示 `remote origin already exists`，先 `git remote -v` 核对，不要盲目覆盖地址。

## 4. 认证不通过时

优先在 VS Code 集成终端执行 push，按 GitHub/VS Code 弹出的登录提示完成认证。
GitHub 的 HTTPS Git 操作不接受账户密码；使用凭据管理器、GitHub CLI 登录或 SSH。
不要把密码、Token 写进 remote URL、仓库文件或发到聊天中。

如果自己已经安装了 GitHub CLI，可执行：

```bash
gh auth login --hostname github.com --git-protocol https --web
gh auth setup-git
git push -u origin main
```

本次准备时当前命令环境没有找到 `gh`，所以它是可选方案，不是前述步骤的前提。

## 5. 后续更新

```bash
source .venv/bin/activate
python -m pytest -q
git status --short
git add src tests examples docs README.md pyproject.toml
git diff --cached --stat
git commit -m "Describe the change"
git push
```

以后在这个独立目录维护碰撞球接口。原 FastSim 项目若要改用此包，需要另做显式依赖和导入迁移。

官方参考：[上传本地代码](https://docs.github.com/en/migrations/importing-source-code/using-the-command-line-to-import-source-code/adding-locally-hosted-code-to-github)、
[GitHub 认证方式](https://docs.github.com/en/authentication/keeping-your-account-and-data-secure/about-authentication-to-github)。
