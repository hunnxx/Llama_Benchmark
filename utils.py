import json

from operator import itemgetter
from PIL import Image

from torch.utils.data import Dataset

class DocVQADataset(Dataset):
    def __init__(self, file_path):
        super().__init__()
        
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        __data = self.data[index]
        __img_path, __q, __q_id = itemgetter('image', 'question', 'question_id')(__data)
        
        return __img_path, __q, __q_id