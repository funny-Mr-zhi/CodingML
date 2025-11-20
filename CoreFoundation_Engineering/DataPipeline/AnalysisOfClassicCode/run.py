from customer_dataset import *
from customer_dataloader import *
from utils import *

def run_dataset_exp():
    dataset_list = []
    # 创建玩具数据集
    print("-"*20, "\n", "Toy_dataset")
    data_list = [i for i in range(10)]
    toy_dataset = Toy_dataset(data_list)
    print("Dataset length:", len(toy_dataset))
    print("First item dataset[0]:", toy_dataset[0])
    print("Tenth item dataset[9]:", toy_dataset[9])
    dataset_list.append(toy_dataset)

    # 创建图像数据集
    print("-"*20, "\n", "Image Dataset")
    img_data_list = fake_data("img", 10)
    img_dataset = Img_dataset(img_data_list)
    print("Dataset length:", len(img_dataset))
    print("First item shape img_dataset[0]:", img_dataset[0].shape)
    print("Tenth item shape img_dataset[9]:", img_dataset[9].shape)
    dataset_list.append(img_dataset)

    # 创建图数据集
    # print("-"*20, "\n", "Graph Dataset")
    # graph_data_list = fake_data("graph", 10)

    return dataset_list
    

def run_dataloader_exp(dataset_list):
    toy_dataset, img_dataset = dataset_list[0], dataset_list[1]

    # Toy数据集，追踪函数调用
    print("-"*20, "\n", "Toy dataloader")
    print(f"total batches: {len(toy_dataset) // 3}, total len: {len(toy_dataset)}")
    print("using default collate_fn, batch_size=3, shuffle=True, drop_last=True")
    toy_dataloader = DataLoader(toy_dataset, batch_size= 3, shuffle=True, drop_last=True)
    for i, data in enumerate(toy_dataloader):
        print(f"Batch {i}: {data}")

    # img数据集，观察默认拼接函数
    print("-"*20, "\n", "Img dataloader")
    print(f"total batches: {len(img_dataset) // 4}, total len: {len(img_dataset)}")
    print("using default collate_fn, batch_size=4, shuffle=False, drop_last=False")
    print(f"single data shape {img_dataset[0].shape}")
    img_dataloader = DataLoader(img_dataset, batch_size= 4, shuffle=False, drop_last=False)
    for i, data in enumerate(img_dataloader):
        print(f"Batch {i}: {data.shape}")
    

def main():
    # 运行三种类型的数据集实验：toy, img, graph
    dataset_list = run_dataset_exp()

    run_dataloader_exp(dataset_list)

if __name__ == "__main__":
    main()