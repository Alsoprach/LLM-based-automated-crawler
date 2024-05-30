import yaml
import os
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

manager = OpenAIKeyManager()
load_keys_from_yaml(manager, 'openai_keys.yaml')
