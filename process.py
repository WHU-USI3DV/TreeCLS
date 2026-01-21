import json

from tqdm import tqdm

test_json_path = '/root/data1/remote_data/tree_fineR_classification/TreeCLS/data/whu_test_attributes_reproduced.json'
val_json_path = '/root/data1/remote_data/tree_fineR_classification/TreeCLS/data/whu_val_attributes_reproduced.json'
train_json_path = '/root/data1/remote_data/tree_fineR_classification/TreeCLS/data/whu_train_attributes_reproduced.json'

f = open(test_json_path, 'r')
test_data = json.load(f)
f.close()

f = open(val_json_path, 'r')
val_data = json.load(f)
f.close()

f = open(train_json_path, 'r')
train_data = json.load(f)
f.close()


new_dict = {'train': [], 'val': [], 'test': []}

for d in train_data:
    new_dict['train'].append(
        {'img_path': d['img_path'], 'label': d['label']}
    )

for d in val_data:
    new_dict['val'].append(
        {'img_path': d['img_path'], 'label': d['label']}
    )

for d in test_data:
    new_dict['test'].append(
        {'img_path': d['img_path'], 'label': d['label']}
    )

with open('WHU.json', 'w', encoding='utf-8') as f:
    json.dump(new_dict, f, ensure_ascii=False)