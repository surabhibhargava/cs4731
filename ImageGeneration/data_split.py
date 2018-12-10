import os, sys
import random
from shutil import copyfile

data_path_sketchy = '/Users/surabhibhargava/acads/cs4731/project/code/sketchy_concat'

classes_sketchy = [item for item in os.listdir(data_path_sketchy) if
                   os.path.isdir(os.path.join(data_path_sketchy, item))]

classes = [cls for cls in classes_sketchy]
print(classes)

for cls in classes:
    cls_path = os.path.join(data_path_sketchy, cls)
    all_files = [f for f in os.listdir(cls_path)]
    random.shuffle(all_files)
    length = len(all_files)
    train = all_files[:400]
    val = all_files[400:450]
    test = all_files[450:500]
    if not os.path.exists('/Users/surabhibhargava/acads/cs4731/project/code/sketchy_splits/' + cls):
        os.makedirs('/Users/surabhibhargava/acads/cs4731/project/code/sketchy_splits/' + cls)
        os.makedirs('/Users/surabhibhargava/acads/cs4731/project/code/sketchy_splits/' + cls + '/train')
        os.makedirs('/Users/surabhibhargava/acads/cs4731/project/code/sketchy_splits/' + cls + '/val')
        os.makedirs('/Users/surabhibhargava/acads/cs4731/project/code/sketchy_splits/' + cls + '/test')
    for f in train:
        copyfile(os.path.join(data_path_sketchy, cls, f),os.path.join('/Users/surabhibhargava/acads/cs4731/project/code/sketchy_splits/'+cls+'/train', f))
    for f in val:
        copyfile(os.path.join(data_path_sketchy, cls, f),
                 os.path.join('/Users/surabhibhargava/acads/cs4731/project/code/sketchy_splits/'+cls+'/val', f))
    for f in test:
        copyfile(os.path.join(data_path_sketchy, cls, f),
                 os.path.join('/Users/surabhibhargava/acads/cs4731/project/code/sketchy_splits/'+cls+'/test', f))

# print(all_files)
