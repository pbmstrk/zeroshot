from pathlib import Path
from collections import OrderedDict

from ..utils import download_extract, parse_from_txt_file
from .base import TextDataset


def ZeroShotTopicClassificationDataset(return_subset='v0'):

    URL = "https://drive.google.com/u/0/uc?id=1qGmyEVD19ruvLLz9J0QGV7rsZPFEz2Az&export=download"
    root = Path(".data")
    name = Path("zeroshot_data")
    filename = Path("topic.tar.gz")
    foldername = Path("BenchmarkingZeroShot")

    download_extract(URL, root=root, name=name, filename=filename)

    classes_file = Path("topic/classes.txt")
    category_dict = OrderedDict()
    with open(root / name / foldername / classes_file) as data:
        for i, line in enumerate(data):
            category_dict[line.strip()] = i

    if return_subset == 'v0':
        train_file = Path("topic/train_pu_half_v0.txt")
    elif return_subset == 'v1':
        train_file = Path("topic/train_pu_half_v1.txt")
    dev_file = Path("topic/dev.txt")
    test_file = Path("topic/test.txt")

    train_data = parse_from_txt_file(root / name / foldername / train_file)
    dev_data = parse_from_txt_file(root / name / foldername / dev_file)
    test_data = parse_from_txt_file(root / name / foldername / test_file)

    return (
        TextDataset(train_data),
        TextDataset(dev_data),
        TextDataset(test_data),
        category_dict
    )
