from torch.utils.data import Dataset

class Toy_dataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        print("\tCalling __len__")
        return len(self.data_list)

    def __getitem__(self, idx):
        print("\tCalling __getitem__ for index:", idx)
        return self.data_list[idx]
    
class Img_dataset(Dataset):
    def __init__(self, img_data_list):
        self.img_data_list = img_data_list

    def __len__(self):
        return len(self.img_data_list)

    def __getitem__(self, idx):
        return self.img_data_list[idx]