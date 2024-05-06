import os
from PIL import Image
from datasets import load_dataset

# 데이터셋 다운로드
dataset_splits = load_dataset('imagenet-1k', split=['validation', 'test'])

# 저장할 디렉토리 생성
validation_dir = 'data/Imagenet/validation'
test_dir = 'data/Imagenet/test'
os.makedirs(validation_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# 클래스 목록 추출 (assumes class names are stored in a dictionary or list accessible in this way)
class_names = dataset_splits[0].features['label'].names  # Adjust this if needed based on actual dataset structure

# validation 데이터 저장
for example in dataset_splits[0]:
    image = example['image']
    label = example['label']
    class_name = class_names[label]  # Get the class name directly using the label as index
    class_dir = os.path.join(validation_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    image_path = os.path.join(class_dir, f'{label}.jpeg')
    image.save(image_path)

# test 데이터 저장
for example in dataset_splits[1]:
    image = example['image']
    label = example['label']
    class_name = class_names[label]  # Same as above
    class_dir = os.path.join(test_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    image_path = os.path.join(class_dir, f'{label}.jpeg')
    image.save(image_path)
