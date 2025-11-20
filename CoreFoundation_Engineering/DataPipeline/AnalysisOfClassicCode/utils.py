import torch

def fake_data(data_type, size):
    """
    根据不同数据类型，生成对应的假数据
    参数：
    - data_type: 数据类型，支持"img"、"text"、"tabular"、"graph"
    - size: 生成数据的数量
    返回：
    - data_list: 生成的数据列表
    """
    if data_type == "img":
        img_list = []
        for i in range(size):
            img = torch.randn(3, 224, 224)  # 假设图像大小为3x224x224
            img_list.append(img)
        return img_list
    elif data_type == "text":
        text_list = []
        for i in range(size):
            text = "This is a sample text number {}".format(i)
            text_list.append(text)
        return text_list
    elif data_type == "tabular":
        tabular_list = []
        for i in range(size):
            row = torch.randn(10)  # 假设每行有10个特征
            tabular_list.append(row)
        return tabular_list
    elif data_type == "graph":
        from torch_geometric.data import Data
        graph_list = []
        for i in range(size):
            x = torch.randn(5, 16)  # 5个节点，每个节点16维特征
            edge_index = torch.tensor([[0, 1, 2, 3],
                                       [1, 0, 3, 2]], dtype=torch.long)  # 简单的边连接
            graph = Data(x=x, edge_index=edge_index)
            graph_list.append(graph)
        return graph_list