# ⽀持用户拖拽矢量图的医学绘图小程序、文字不乱码

# 1. 具体效果
<img width="681" height="1201" alt="image" src="https://github.com/user-attachments/assets/0b4f32eb-e7fb-48e9-80e4-892688bb70c5" />

<img width="1280" height="957" alt="2e6887c7-ae47-4ad6-ae99-823047688939" src="https://github.com/user-attachments/assets/3efc2280-95b6-43d6-af02-2bb87c77d55f" />

<img width="1280" height="964" alt="68cbe09a-b705-4a3b-a646-a7033693d616" src="https://github.com/user-attachments/assets/f31d91ae-5863-46a6-8c52-aa1ee12cdc17" />


# 2. 功能设计
输入：
1.用户输入的自然语言要求；
2.用户上传的文档(PDF,WORD格式，不超过50MB，可不上传)；
3.用户选择的图片类型：共四种，机制通路图/实验流程图/技术路线图/科普插图
（并且针对每种类型的图片需要选择具体要求：图片风格/图片语言/信息密度/图片比例）
输出：
输出对应的医学绘图图片。输出结束后，支持用户拖拽连线和标签、支持复原。用户可保存初始图或修改后版本


# 3. 具体流程设计

## 3.1 请求接收与排队
前端将文本需求、选择的风格和文献打包发送给 FastAPI 后端。后端保存 PDF，将任务塞入队列，并返回一个 task_id。前端开始根据 task_id 轮询。

## 3.2 文献解析与Prompt规划
将用户上传的文献和自然语言要求，转化为绘图指导prompt。
- 输入 (Inputs):
  - pdf_path: 用户上传的 PDF 文件路径（可选）。
  - user_query: 用户前端输入的具体绘图要求。
  - image_type, style, language, density, aspect_ratio: 前端选择的各项参数配置。
- 处理逻辑:
  1. 使用 pdfplumber 提取文本，并交给 Qwen3-8B 提炼出关键实体、关系。
  2. 结合提炼的 summary、user_query 和系统预设的各类画风/密度指南，再次让 Qwen3-8B 输出JSON结构。
  3. 系统会强制在 Prompt 中加入 No text, no letters... 等反面提示词，确保底图纯净。
- 输出 (Outputs): 一个包含结构化绘图计划的字典 ：
{
  "summary": "文献总结文本...",
  "prompt_bundle": {
    "base_prompt": "Medical 机制通路图, 3D render... No text, no letters...",
    "negative_prompt": "text, letters, caption, watermark",
    "text_items": [
      {"id": "L1", "text": "信号起始", "target": "上游受体区域"},
      {"id": "L2", "text": "级联激活", "target": "中部通路核心"}
    ]
  },
  "prompt": "完整的组合版 prompt 字符串"

}
## 3.3 无文字底图生成
调用 Qwen-Image 模型。系统会根据前端传入的图片类型动态挂载对应的 LoRA 权重，渲染出一张纯净底图。
- 输入 (Inputs):
  - prompt: 第一步生成的 base_prompt。
  - image_type: 用于匹配挂载对应的LoRA权重。
  - width / height: 根据 aspect_ratio 映射出的具体分辨率（如 1024x1024）。
- 输出 (Outputs):
  - 无文字的医学底图。
  
## 3.4 空间布局与锚点计算
将刚生成的纯净底图和步骤 2 提取的“标签列表”喂给视觉语言模型。VL 模型通过理解图片，推理出每个医学标签最合适的摆放位置，以及连线需要指向的坐标，输出严格的JSON坐标系。
- 输入 (Inputs):
  - image_path: 第二步生成的纯净底图的物理路径。
  - text_items: 第一步生成的标签数组（包含需要写的文字和要指向的目标区域）。
  - image_type & language: 用于微调 Prompt 规则（例如“科普插图”必须带连线）。
- 处理逻辑:
  1. 将底图和带有 text_items 任务清单的 Prompt 输入给 Qwen3-VL-8B。
  2. VL模型观察底图的留白区域和关键目标结构，推理出每个标签最合适的放置坐标。
  3. 输出并解析出包含各个标签归一化坐标（0~1）和引线坐标的 JSON 结构。
- 输出 (Outputs): 具体的布局坐标计划，如果解析失败则使用默认布局

## 3.5 图文合成与后处理
使用 PIL (Python Imaging Library) 根据 VL 模型输出的相对坐标（转化为绝对像素），在底图上绘制圆角矩形、多语言文本和连接线，生成带标注的最终结果图。
- 处理逻辑:
  1. 使用 PIL (Pillow) 库打开底图，并创建一个透明蒙版。
  2. 根据 connectors 的归一化坐标计算实际像素位置，画出连接线。
  3. 根据 labels 的归一化坐标计算实际像素位置，画出带有圆角和半透明背景的文字框。
  4. 算法根据文字框的大小和文字长度，自动调整字号和换行，将文字写入框中。
- 输出 (Outputs):
  - 带有专业排版标签和指示线的最终合成图
阶段 6：交互式展示 (Frontend) 
前端获取到生成的 layout_plan JSON 和底图 URL 后，会在画布上渲染出一套可拖拽的HTML覆盖层。用户可以自由调整 VL 模型生成的坐标，并在前端点击保存，实现最终的定稿。调整后通过 Canvas 合并保存

## 3.6 异步队列链路部署
FastAPI：API 网关
- 负责直接和用户交流，轻量级的网络通信
- APP 把 PDF 和参数发给 FastAPI。FastAPI 把任务写队列里，并给用户一个id号。用户定期询问问 FastAPI是否完成，FastAPI可以马上回复。
Redis：消息中间件 & 状态库
- 连接前台（FastAPI）和后端（Celery），用户的消息打包到Redis 队列里排队。程序跑到哪一步了会实时写进 Redis中的表。FastAPI 去 Redis 里查这张表，反馈给用户。
Celery Worker：后端异步任务执行器
- 分布式任务队列，在后台工作，和Redis直接接触
- 发现有用户新需求，则加载 Qwen3 -> 解析 PDF -> 释放显存 -> 加载 Qwen-Image -> 绘图 -> 释放显存。

启动redis、Celery和FastAPI:
```
# 安装库
pip install fastapi uvicorn celery redis python-multipart
sudo apt install celery
# 启动redis、Celery Work和FastAPI（确保开三个终端，三条指令都在不同终端运行）
# 第一个终端
redis-server
# 第二个终端
cd ./backend
celery -A worker.celery_app worker --loglevel=info --concurrency=1 -P solo
# 第三个终端
cd ./backend
python main.py
后端链路已经可以运行了，但我们无法在服务器浏览器上测试，需要将端口通过SSH转发到本地，使用本地浏览器进行测试
# 请务必在自己本地的电脑设置该命令
# 该命令只用于测试链路
# 8000是后端端口，3000是前端Next.js端口
# ssh -F /dev/null -L 8000:localhost:8000 -L 3000:localhost:3000 ubuntu@@你的服务器ip
ssh -F /dev/null -L 8000:localhost:8000 -L 3000:localhost:3000 ubuntu@117.50.193.22
使用本地浏览器访问该网址http://localhost:8000/docs
```

## 3.7 前端编写
前端部署：基于Next.js
安装Node.js和相关图标库，并创建一个前端文件
```
# 安装nvm(Node.js的版本管理器)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
# 安装nvm install 20并验证是否升级成功
nvm install 20
node -v
```
创建一个前端框架
```
npx create-next-app@latest frontend --registry=https://registry.npmmirror.com --no-audit --yes
# 启动前端框架，启动后，可以在本地电脑的http://localhost:3000访问
cd frontend
# 安装用于调用后端API的请求库
npm install axios
# 安装常用的图标库
npm install lucide-react
编译并启动前端
cd ./frontend
# 每次使用前要重新扫描包
npm install --registry=https://registry.npmmirror.com
# Linux默认进程数1024，解除限制
ulimit -n 65535
# 在服务器上启动前端框架
npm run dev
```
如果要在自己的电脑上测试仅前端的效果，请使用以下测试方法
```
# 请务必在自己本地的电脑设置该命令
# 该命令只用于测试
# 8000是后端端口，3000是前端Next.js端口
# ssh -F /dev/null -L 8000:localhost:8000 -L 3000:localhost:3000 ubuntu@@你的服务器ip
ssh -F /dev/null -L 8000:localhost:8000 -L 3000:localhost:3000 ubuntu@117.50.193.22
# ssh连接后，在本地http://localhost:3000/访问应用前端
# 运行结束后，服务器的./data/uploads路径下保存用户上传的图片，
# ./data/output路径下保存模型生成的图片。同时，测试端可以看到输出结果
```

## 3.8 公网链接：基于pm2启动进程，基于Nginx转发
编译前端为速度更快的静态产物，并且通过进程管理工具让前端服务在后台24h运行
安装库pm2
```
sudo npm install pm2 -g
```
启动pm2进程
```
# 清理进程
pm2 delete all
pm2 list
# 启动后端进程
cd ./backend
# 启动FastApi进程
pm2 start /home/ubuntu/anaconda3/envs/text2image/bin/python --name "medical-api" -- main.py
# 启动celery worker进程，设置了单显卡concurrency=1，设置了--pool=solo 让 Celery 不再派生子进程
pm2 start /home/ubuntu/anaconda3/envs/text2image/bin/python --name "medical-worker" -- -m celery -A worker.celery_app worker --loglevel=info --concurrency=1 --pool=solo
# 启动前端进程
cd ../frontend
# 如果加入了新功能，调用了新的包，则使用前要重新扫描包
# npm install --registry=https://registry.npmmirror.com
# 编译前端文件并启动
npm run build
pm2 start npm --name "medical-web" -- start
# 测试时，可以用logs命令查看进程的运行日志
pm2 logs medical-worker --lines 100
```
安装Nginx库
```
sudo apt update
sudo apt install nginx -y
```
配置Niginx代理
```
# 配置转发规则文件，代码如下
sudo nano /etc/nginx/sites-available/medical_app
# 转发规则配置文件的代码
server {
    listen 80;
    server_name 117.50.193.22; 
    # Nginx 默认只允许上传 1MB 以内的文件。需要手动调高这个上限。
    client_max_body_size 50M;
    # 1. 拦截前端页面请求，转发给 Next.js (端口 3000)
    location / {
        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
    # 2. 拦截 API 请求，转发给 FastAPI (端口 8000)
    location /api/ {
        proxy_pass http://localhost:8000/api/;
        proxy_set_header Host $host;
    }
    # 3. 拦截图片资源请求，转发给 FastAPI 的静态目录
    location /output/ {
        proxy_pass http://localhost:8000/output/;
    }
}
```
启动Nginx代理
```
# 建立软链接激活配置
sudo ln -s /etc/nginx/sites-available/medical_app /etc/nginx/sites-enabled/
# 重启 Nginx
sudo systemctl restart nginx
# 启动后，在浏览器输入服务器公网IP，就可以使用应用。如果后续需要真实网址，需要去购买一个域名
# http://117.50.193.22
```


# 4. 具体开发流程
```
      ┌─────────────────────────────────────────────────────────┐
      │               前端：用户输入层(微信小程序)                │
      │      负责UI交互，收集用户自然语言、上传的文件以及四类配置项 │
      └───────────────────────┬─────────────────────────────────┘
                              ↓
      ┌─────────────────────────────────────────────────────────┐
      │               业务端：调用大模型(FastAPI)                │
      │     处理 API 请求，解析文档，RAG，生成Prompt，调用大模型  │
      └───────────────────────┬─────────────────────────────────┘
                              ↓
      ┌─────────────────────────────────────────────────────────┐
      │              大模型中枢：大模型训练(Text LLM)            │
      │        数据集确定 → 数据集清洗 → 微调训练 → 大模型部署    │
      └───────────────────────┬─────────────────────────────────┘
                              ↓
      ┌─────────────────────────────────────────────────────────┐
      │          模型推理：模型推理(Diffsynth + Qwen-Image)      │
      │数据集确定 → 数据集清洗 → 微调训练 → 调用大模型 → 返回推理结果│
      └─────────────────────────────────────────────────────────┘
```

# 5. 运行平台和库

## 5.1 运行平台
优云智算：A800，80G显存，ubuntu-nvidia系统镜像

## 5.2 基础库安装
安装python3并使python命令指向pthon3
```
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip python3-venv -y
sudo update-alternatives --install /usr/bin/python python /usr/bin/python3 1
安装anaconda
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
bash Anaconda3-2024.10-1-Linux-x86_64.sh
source ~/.bashrc
conda --version
```
安装运行库
```
pip install peft accelerate datasets protobuf transformers
pip install torch torchvision
pip install --upgrade huggingface-hub
pip install --upgrade sentence-transformers
pip install --upgrade bitsandbytes
# 前端库
pip install streamlit requests
pip install git+https://github.com/huggingface/diffusers.git
# 文生图框架DiffSynth
git clone https://github.com/modelscope/DiffSynth-Studio.git
cd DiffSynth-Studio
pip install -e .
```

# 6. 数据集收集与清洗

## 6.1 数据集调研
要剔除分辨率过低、包含过多文字说明的图片。剔除折线图、柱状图、显微镜照片。
### 6.1.1 机制通路图
这类图片强调分子间的相互作用、信号传导和细胞器功能。
WikiPathways：https://academy.wikipathways.org/
开放的生物学通路数据库，由科学社区持续更新。
适用性：提供高质量的通路图，包含详细的通路描述、实体列表。
Reactome：开源的、经过同行评审的生物学通路数据库。还可以考虑BioRender、FigDraw
### 6.1.2 实验流程图 & 技术路线图 
PubMed Central Open Access Subset(PMC-OA):
https://huggingface.co/datasets/axiong/pmc_oa_demo/viewer/default/train?sort%5Bcolumn%5D=image&sort%5Bdirection%5D=desc&sort%5Btransform%5D=width
含了数以百万计的开源生物医学论文中的原始插图及其对应的图注
### 6.1.3 科普插图
偏向3D渲染或精美的扁平化矢量图，要求生动具象。
Servier Medical Art(SMART)：https://smart.servier.com/
著名的开源医学插图库。提供大量高质量的细胞、解剖、器材等矢量元素和完整插图。
维基共享资源 (Wikimedia Commons - Medical Illustrations)：
维基百科旗下，包含海量版权开放的医学解剖、病理插图。

## 6.2 构建SFT训练集
下载的图片全部转化为png格式
自动打标：
本地部署Qwen3-VL-8B-Instruct模型进行打标
使用打标签脚本qwen3_fast_tragger.py对原始图片进行打标

打标后的数据集格式如下：
{"prompt": "科普插图, 专业3D渲染, 英文, high density, 详细医学描述：该图展示了心脏壁的解剖结构，包括心外膜（由壁层和脏层心包组成）、心肌层（心肌组织）和心内膜层。心包腔位于壁层和脏层心包之间，内含少量液体以减少摩擦。图中清晰标注了各层结构，有助于理解心脏的组织层次和功能分区。", "image": "BlausenMedical2014_Blausen_0470_HeartWall.png"}
{"prompt": "科普插图, 专业3D渲染, 中文, high density, 详细医学描述：该图像为人体脊柱的3D解剖示意图，重点突出颈椎和腰椎区域，通过放大镜特写展示颈椎椎体结构，用于科普脊柱解剖学知识。", "image": "BlausenMedical2014_Blausen_0618_LumbarSpine.png"}
{"prompt": "科普插图, 扁平矢量插画, 英文, medium density, 详细医学描述：该图展示了尿道括约肌的解剖结构，包括膀胱肌肉、尿液、括约肌肌肉和尿道。右侧插图显示了人体腹部和盆腔区域，突出显示了膀胱和尿道的位置，用于说明尿道括约肌在控制排尿中的作用。", "image": "BlausenMedical2014_Urinary_Sphincter.png"}
{"prompt": "科普插图, 扁平矢量插画, 中文, high density, 详细医学描述：该图展示了C4植物（如玉米）叶片的横切面结构，包括维管束鞘细胞、叶肉细胞、叶脉（木质部和韧皮部）、表皮细胞、机械组织等，用于说明C4光合作用的解剖学基础。", "image": "组织学示意图_C4_photosynthesis_is_less_complicated_vi.png"}

模型训练：
将原始图片数据metadata文件存放在./data/raw_images路径下，训练脚本train_science_pop.sh
训练结束后，Lora权重将会存放在./data/Qwen-Image-Trainlora路径下
加载lora：
有两种方式可以加载lora，项目使用热加载模式
1.冷加载：未开启显存管理时，Lora会融合进基础模型权重，加载后无法卸载
2.热加载：开启显存管理时，Lora不会融合进基础模型权重，推理速度变慢一些，但可以用pipe_clear_lora()卸载
训练参数列表可见DiffSynth的官方文档：https://diffsynth-studio-doc.readthedocs.io/zh-cn/latest/Model_Details/Qwen-Image.html

## 6.3 构建SFT训练集
模型的训练集基于约5000条SFT训练集，全部来自于维基百科。组成如下：
### 6.3.1 现代科普风格总图：1378张
https://commons.wikimedia.org/wiki/Category:Images_from_Blausen_Medical_Communications
Category:Images from Blausen Medical Communications
这是目前维基上质量最统一的医学绘图风格的数据集，包含超过1000多张针对各种疾病和解剖结构的3D感插图。
### 6.3.2 人体解剖图：约1000张
https://commons.wikimedia.org/wiki/Category:Medical_illustrations_by_Patrick_Lynch
Category:Medical illustrations by Patrick Lynch
另外未使用的备选数据：
矢量化解剖图 (精选纯净线条)：Category:SVG human anatomy
链接：https://commons.wikimedia.org/wiki/Category:SVG_human_anatomy
人体解剖示意图总汇 (包含多角度剖面)：Category:Human anatomy diagrams
链接：https://commons.wikimedia.org/wiki/Category:Human_anatomy_diagrams
### 6.3.3 细胞与分子生物学：约1000张
https://commons.wikimedia.org/wiki/Category:Cell_anatomy
Category:Cell anatomy
聚焦于细胞器的抽象表达和大分子结构的模式图
另外未使用的备选数据：
细胞解剖学 (涵盖各种细胞器)：Category:Cell anatom
链接：https://commons.wikimedia.org/wiki/Category:Cell_anatomy
分子与细胞生物学图解 (大类聚合)：Category:Molecular and cellular biology diagrams
链接：https://commons.wikimedia.org/wiki/Category:Molecular_and_cellular_biology_diagrams
### 6.3.4 组织学与病理学：约800张
https://commons.wikimedia.org/wiki/Category:Metabolic_pathway_diagrams
Category:Metabolic pathway diagrams
另外未使用的备选数据：
组织学示意图：Category:Histological schematic
链接：https://commons.wikimedia.org/wiki/Category:Histological_schematic
### 6.3.5 生理机制与信号通路：约1000张
https://commons.wikimedia.org/wiki/Category:Metabolic_pathway_diagrams
Category:Metabolic pathway diagrams
这类图像是“机制通路图”和“流程图”的核心素材，包含大量的箭头、逻辑节点
另外未使用的备选数据：
信号转导图 (细胞内外信息交流)：Category:Signal transduction diagrams
链接：https://commons.wikimedia.org/wiki/Category:Signal_transduction_diagrams
免疫系统模式图 (复杂的细胞互作网络)：Category:Diagrams of the immune system
链接：https://commons.wikimedia.org/wiki/Category:Diagrams_of_the_immune_system



















