# from modelscope.msdatasets import MsDataset
# ds =  MsDataset.load('modelscope/truthful_qa', subset_name='generation', split='validation')
# # # 保存数据集到本地目录
# save_directory = '/tmp/pycharm_project_506/honest_llama-master/truthfulqa_cache/generation'  # 你可以选择任意路径
# ds.save_to_disk(save_directory)

from datasets import *

# 指定之前保存数据集的目录

# 从本地加载数据集
# ds_loaded = load_from_disk('/tmp/pycharm_project_506/honest_llama-master/truthfulqa_cache/generation')
dataset = load_dataset("/tmp/pycharm_project_506/honest_llama-master/truthfulqa_cache/generation", 'default')['validation']
print(dataset)

# 确认数据集正确加载
# print(ds_loaded)
