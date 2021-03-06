from collections import OrderedDict
from pathlib import Path

from ..utils import download_extract, parse_from_txt_file
from .base import TextDataset


def ZeroShotTopicClassificationDataset(return_subset="v0"):

    r"""
    Zero-Shot Topic Classification Dataset.

    A version of the Yahoo Answers dataset. Returns train, dev, test set as well as target labels.

    Reference: `Yin et al. (2019) <https://www.aclweb.org/anthology/D19-1404/>`_

    Args:
        return_subset: Which subset of the training set to return.  One of 'v0' or 'v1'.

    Example::

        >>> train, test, dev, category_dict = ZeroShotTopicClassificationDataset()
    """

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

    label_map = {str(value): key for key, value in category_dict.items()}
    if return_subset == "v0":
        train_file = Path("topic/train_pu_half_v0.txt")
    elif return_subset == "v1":
        train_file = Path("topic/train_pu_half_v1.txt")
    dev_file = Path("topic/dev.txt")
    test_file = Path("topic/test.txt")

    train_data = parse_from_txt_file(root / name / foldername / train_file, label_map)
    dev_data = parse_from_txt_file(root / name / foldername / dev_file, label_map)
    test_data = parse_from_txt_file(root / name / foldername / test_file, label_map)

    return (
        TextDataset(train_data),
        TextDataset(dev_data),
        TextDataset(test_data),
        category_dict,
    )
