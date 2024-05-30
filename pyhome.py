# ------------------------------------------------------------------------ Reference counting
from collections import namedtuple, deque
from functools import wraps
from collections import defaultdict

def manage_references(get_item_func, release_item_func, items_attr):
    """装饰器：管理引用计数的增减"""
    def decorator(cls):
        @wraps(get_item_func)
        def wrapped_get_item(self):
            items = getattr(self, items_attr)
            item, min_ref_count = get_item_func(self, items)
            if item:
                items[min_ref_count + 1].append(item)
                if not items[min_ref_count]:
                    del items[min_ref_count]
                return item
            return None

        @wraps(release_item_func)
        def wrapped_release_item(self, item):
            items = getattr(self, items_attr)
            ref_count = item.citations
            if ref_count in items:
                for i, curr_item in enumerate(items[ref_count]):
                    if release_item_func(curr_item, item):
                        items[ref_count].remove(curr_item)
                        if ref_count > 0:
                            items[ref_count - 1].append(curr_item)
                        break
                if not items[ref_count]:
                    del items[ref_count]

        cls.get_item = wrapped_get_item
        cls.release_item = wrapped_release_item
        return cls
    return decorator

ApiKeyT = namedtuple('ApiKeyT', ['key', 'citations'])

@manage_references(
    lambda self, items: (ApiKeyT(items[min(items.keys())][0], min(items.keys()) + 1), min(items.keys())) if items else (None, None),
    lambda curr, item: curr.key == item.key,
    'api_keys'
)
class OpenAIKeyManager:
    def __init__(self):
        self.api_keys = defaultdict(deque)
    
    def add_api_key(self, key):
        self.api_keys[0].append(key)

def load_keys_from_yaml(manager, filename):
    """从 YAML 文件加载 OpenAI 密钥，并将它们加载到 OpenAIKeyManager 类中"""
    # 检查文件是否存在
    if not os.path.exists(filename):
        # 文件不存在，创建文件并写入提示信息
        with open(filename, 'w') as file:
            yaml.dump(['your-openai-key-here'], file)
        print(f"YAML file created at '{filename}'. Please add your OpenAI keys to this file.")
        return

    # 读取 YAML 文件
    with open(filename, 'r') as file:
        keys = yaml.safe_load(file)
        if not keys:
            print(f"No keys found in '{filename}'.")
            return

    # 将读取的密钥加载到 OpenAIKeyManager 中
    for key in keys:
        manager.add_api_key(key)
        print(f"Loaded key: {mask_key(key)}")

def mask_key(key, visible_chars=10):
    """对密钥进行打码，只显示前几个字符"""
    masked = key[:visible_chars] + '*' * (len(key) - visible_chars)
    return masked

# ------------------------------------------------------------------------ probes
import yaml
from collections import defaultdict
from datetime import datetime, timedelta
import os

class ResourceProbes:
    def __init__(self, filename='resource_usage.yaml'):
        self.filename = filename
        self.resource_usage = defaultdict(lambda: defaultdict(int))
        self.load_data()

    def increment_usage(self, resource_name):
        """增加指定资源的调用次数，并在有先前记录时检查时间连续性"""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        
        # 判断是否已有该资源的记录
        if resource_name in self.resource_usage and self.resource_usage[resource_name]:
            # 获取最后一次记录的时间
            last_hour_str = max(self.resource_usage[resource_name].keys())
            last_hour = datetime.strptime(last_hour_str, "%Y-%m-%d-%H")
            print(last_hour)
            # 填充从最后一次记录到当前时间之间的缺失小时
            next_hour = last_hour + timedelta(hours=1)
            while next_hour < current_hour:
                next_hour_str = next_hour.strftime("%Y-%m-%d-%H")
                self.resource_usage[resource_name][next_hour_str] = 0
                next_hour += timedelta(hours=1)

        # 记录当前小时的调用次数
        current_hour_str = current_hour.strftime("%Y-%m-%d-%H")
        self.resource_usage[resource_name][current_hour_str] += 1
        self.save_data()

    def save_data(self):
        """将数据保存到 YAML 文件"""
        # 转换 defaultdict 为普通字典以保存
        data_to_save = {resource: dict(hours) for resource, hours in self.resource_usage.items()}
        with open(self.filename, 'w') as file:
            yaml.dump(data_to_save, file, default_flow_style=False)

    def load_data(self):
        """从 YAML 文件加载数据"""
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as file:
                loaded_data = yaml.safe_load(file) or {}
                # 转换加载的数据为 defaultdict 结构
                for resource, hours in loaded_data.items():
                    self.resource_usage[resource] = defaultdict(int, hours)
        else:
            self.resource_usage = defaultdict(lambda: defaultdict(int))

    def report_hourly_usage(self):
        """生成每小时资源使用情况的数据，用于图形化展示"""
        usage_data = {}
        for resource, hours in self.resource_usage.items():
            if hours:
                hours_sorted = sorted(hours.items())
                times = [time for time, _ in hours_sorted]
                counts = [count for _, count in hours_sorted]
                usage_data[resource] = {"times": times, "counts": counts}
            else:
                usage_data[resource] = {"times": [], "counts": []}
        return usage_data

    def openai_inc(self):
        """openai资源快捷增长函数"""
        self.increment_usage("OpenAI Request")
    
    def web_inc(self):
        """web资源快捷增长函数"""
        self.increment_usage("Web Request")

class ProgressProbes:
    def __init__(self, total_steps=0):
        self.total_steps = total_steps
        self.current_step = 0

    def update_progress(self, steps=1):
        """更新进度状态"""
        self.current_step += steps
        self.current_step = min(self.current_step, self.total_steps)

    def increase_total_steps(self, additional_steps):
        """增加总步数"""
        if additional_steps > 0:
            self.total_steps += additional_steps

    def get_progress(self):
        """获取当前进度百分比"""
        return (self.current_step / self.total_steps) * 100 if self.total_steps > 0 else 100

    def report_progress(self):
        """打印进度报告"""
        progress = self.get_progress()
        print(f"Progress: {progress:.2f}% ({self.current_step}/{self.total_steps})")

# --------------------------------------------------------------------- schemas

class SchemaManager:
    def __init__(self, filename='schemas.yaml'):
        self.schemas = {}
        self.filename = filename
        self.load_schemas()

    def add_schema(self, url, schema):
        """增加一个新的数据提取字典"""
        if self.validate_schema(schema):
            self.schemas[url] = schema
        else:
            print("Schemas Error")

    def remove_schema(self, url):
        """根据网址删除对应的数据提取字典"""
        if url in self.schemas:
            del self.schemas[url]

    def get_schema(self, url):
        """根据网址查询对应的数据提取字典"""
        return self.schemas.get(url, "No schema available for this URL.")

    def save_schemas(self):
        """将所有数据提取字典存储到 YAML 文件"""
        with open(self.filename, 'w') as file:
            yaml.dump(self.schemas, file, default_flow_style=False)

    def load_schemas(self):
        """从 YAML 文件加载数据提取字典"""
        try:
            with open(self.filename, 'r') as file:
                self.schemas = yaml.safe_load(file) or {}
        except FileNotFoundError:
            self.schemas = {}
    
    def validate_schema(self, schema):
        """检查 schema 是否合法"""
        # 检查 "properties" 和 "required" 是否存在
        if "properties" not in schema or "required" not in schema:
            return False

        # 检查 "properties" 是否非空且格式正确
        if not schema["properties"] or not all(isinstance(prop, dict) and 'type' in prop for prop in schema["properties"].values()):
            return False

        # 检查 "required" 下的元素是否是 "properties" 下的元素的子集
        if not set(schema["required"]).issubset(schema["properties"]):
            return False

        return True
    
def check_urls_in_schema_manager(file_path, schema_manager):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        create_file = input(f"The file '{file_path}' does not exist. Do you want to create it? (y/n): ")
        if create_file.lower() == 'y':
            with open(file_path, 'w') as file:
                yaml.dump([], file)
            print(f"File created at '{file_path}'. Please add your URLs to this file.")
        else:
            print("Operation cancelled.")
        return [], []

    # 读取 YAML 文件
    with open(file_path, 'r') as file:
        urls = yaml.safe_load(file) or []

    # 检查哪些 URL 在 SchemaManager 中不存在 schema
    existing_urls, missing_urls = [], []
    for url in urls:
        if schema_manager.get_schema(url):
            existing_urls.append(url)
        else:
            missing_urls.append(url)

    return existing_urls, missing_urls


# --------------------------------------------------------------------- LLM
import time
import pprint
import openai
from typing import Callable
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import create_extraction_chain
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import BeautifulSoupTransformer
from langchain.schema import StrOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI

openai.api_base = "https://api.chatanywhere.tech/v1"

def extract(content: str, schema: dict):
    api_keyT = API_Manager.get_item()
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo",api_key=api_keyT.key)
    data = create_extraction_chain(schema=schema, llm=llm).run(content)
    Global_Res_Probes.openai_inc()
    API_Manager.release_item(api_keyT)
    return data

def scrape_with_playwright(urls, schema):
    loader = AsyncChromiumLoader(urls)
    Global_Res_Probes.web_inc()
    docs = loader.load()
    # pprint.pprint(docs)
    bs_transformer = BeautifulSoupTransformer()
    docs_transformed = bs_transformer.transform_documents(
        docs
    )
    # pprint.pprint(docs_transformed)
    print("Extracting content with LLM")
    # Grab the first 1000 tokens of the site
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=5000, chunk_overlap=0
    )
    splits = splitter.split_documents(docs_transformed)
    
    extracted_content = []

    Global_Progress_Probes.increase_total_steps(len(splits))
    print("===========>",len(splits))

    for i in splits:
        print("------------------------------")
        print(i.page_content)
        extracted_content.append(extract(schema=schema, content=i.page_content))
        Global_Progress_Probes.update_progress()
        Global_Progress_Probes.report_progress()
    pprint.pprint(extracted_content)
    return extracted_content
    # return

def data_sql(data):
    template = '''
    The most important thing: I am a skilled engineer, please don't explain the code to me or prompt me anything, I just need the code, thank you. No need to provide test data.
    Writing a python function name is save_data has the following requirements:
    1. Receives a parameter with the data provided below
    2. Use sqlite3
    3. The database is named spider.db
    4. Based on the structured data below, write python code to execute sql statements to save the following data
    5. Check if there is a table for this data, if not, 6.
    6. Please design tables, build tables, and insert data according to the characteristics of the data. 
    7. Please note that the naming of the table should reflect the ownership of the data. For example, tables used for data storage such as LIANHE_ZAOBAO_News_article_title cannot use the name news, but should use LIANHEZAOBAO_news, because news cannot reflect the ownership of the data.
    '''

    human_text = '''
    {Data}
    '''

    prompt = ChatPromptTemplate.from_messages([
        ("system",template),
        ("human",human_text),
    ])

    api_keyT = API_Manager.get_item()
    model = ChatOpenAI(api_key=api_keyT.key)
    output_parser = StrOutputParser()

    chain = prompt | model | output_parser

    outdata = chain.invoke({"Data":data})
    Global_Res_Probes.openai_inc()
    API_Manager.release_item(api_keyT)      
    return outdata

def extract_code(text):
    # 定义代码块的开始和结束标记
    start_marker = "```python"
    end_marker = "```"

    # 查找开始和结束标记的位置
    start = text.find(start_marker)
    end = text.find(end_marker, start + len(start_marker))

    # 提取并返回代码块
    if start != -1 and end != -1:
        return text[start + len(start_marker):end].strip()
    else:
        return text
    
save_data: Callable = lambda x: None
def spider(urls, schema):
    url = [urls]
    Global_Progress_Probes.increase_total_steps(1)
    extracted_content = scrape_with_playwright(url, schema)
    for extracted in extracted_content:
        time.sleep(2)
        origin_data_sql_exec = data_sql(extracted)
        print("---------------------")
        print(origin_data_sql_exec)

        data_sql_exec = extract_code(origin_data_sql_exec)
        print("---------------------")
        print(data_sql_exec)

        # 在全局作用域中执行代码
        exec(data_sql_exec, globals())

        # 检查 save_data 是否在全局作用域中定义
        try:
            save_data(extracted)
        except:
            print("Error")
    Global_Progress_Probes.update_progress()
    Global_Progress_Probes.report_progress()
    print("++++++++++++OVER++++++++++++++++")

# --------------------------------------------------------------------- thread
import threading
def thread_manager(urls):
    threads = []

    # 为每个 URL 创建并启动一个线程
    for url in urls:
        schema = Schema_Manager.get_schema(url)
        if schema:
            thread = threading.Thread(target=spider, args=(url, schema))
            threads.append(thread)
            thread.start()
        else:
            print(f"No schema available for URL: {url}")
    # 等待所有线程完成
    

# --------------------------------------------------------------------- Graphical
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

# 创建和更新图形界面的函数
def visualize_resource_and_progress(resource_probes, progress_probes):
    def update_charts():
        nonlocal close_counter
        # 更新资源使用情况的柱形图
        usage_data = resource_probes.report_hourly_usage()
        for i, (resource, data) in enumerate(usage_data.items()):
            axs[i].clear()
            axs[i].bar(data['times'], data['counts'])
            axs[i].set_title(resource)
            axs[i].set_xlabel('Time')
            axs[i].set_ylabel('Usage Count')
            for label in axs[i].get_xticklabels():
                label.set_rotation(45)  # 旋转 X 轴标签以减少重叠

        # 更新进度条
        progress = progress_probes.get_progress()
        progress_ax.clear()
        progress_ax.barh([0], [progress/100], height=0.1)
        progress_ax.set_xlim(0, 1)
        progress_ax.set_title('Progress')
        progress_ax.get_yaxis().set_visible(False)

        canvas.draw()

        # 检查进度是否已满
        if progress >= 100:
            close_counter -= 1
            if close_counter <= 0:
                root.destroy()
                return

        root.after(1000, update_charts)  # 每秒更新一次

    num_resources = len(resource_probes.resource_usage)
    close_counter = 10  # 10 秒计时器

    root = tk.Tk()
    root.title("Resource and Progress Visualization")

    # 创建更大的图形和子图
    fig, axs = plt.subplots(num_resources + 1, 1, figsize=(3, 1 + 2 * num_resources))
    axs = axs.ravel()  # 将 axs 转换为一维数组，以便于索引
    progress_ax = axs[-1]

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    plt.rcParams.update({'font.size': 5}) 
    # 调整子图间的间距
    plt.subplots_adjust(hspace=2)

    update_charts()  # 初始绘制
    root.mainloop()





# --------------------------------------------------------------------- Main

Global_Res_Probes = ResourceProbes()
Global_Progress_Probes = ProgressProbes() 

API_Manager = OpenAIKeyManager()
load_keys_from_yaml(API_Manager, 'openai_keys.yaml')

Schema_Manager = SchemaManager()

urlsfile = "urls.yaml"
existing_urls, missing_urls = check_urls_in_schema_manager(urlsfile,Schema_Manager)
if missing_urls:
    print("Missing URLs:", missing_urls)
else:
    print("All URLs have existing schemas:", existing_urls)


thread_manager(existing_urls)

visualize_resource_and_progress(Global_Res_Probes, Global_Progress_Probes)
# url = existing_urls[0]
# sche = Schema_Manager.get_schema(url)
# print(url)
# print(sche)
# spider(url,sche)