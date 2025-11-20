# 

在VQGraph中找到`VQ`模块，实现了VQVAE，主要目的是学会使用该模块

Code from: [VQGraph paper: VQ com0ponent](https://github.com/YangLing0818/VQGraph/blob/main/vq.py) 

## 主要类

**VectorQuantize(nn.Module)**

### init

### forward

* 输入：`X`
* 输出：
    * `quantize`： 量化后的潜在表示
    * `embed_ind`：
    * `loss`
    * `dist`
    * `self._codebook.embed`

* 处理
    * 一系列变形操作
    * `x = self.project_in(x)`：如果`dim(x) != dim(code)`
    * 变换x维度，如果采用多头
    * `quantize, embed_ind, dist, embed = self._codebook(x)`
    * if self.training
        * `quantize = x + (quantize - x).detach()`
        * `loss += commit_loss * commitment_weight` 防止code主动向量化前的z靠近
        * `loss += orthogonal_reg_loss * orthogonal_reg_weight` 鼓励码本向量正交化
        * `loss += `
    * `quantize = self.project_out(quantize)`：如果`dim(x) != dim(code)`
    * 一系列变形操作

## 码表类

**EuclideanCodebook**和**CosineSimCodebook**类

这里主要看`cosine`这种

### init

### forward

* 输入： `x`
* 输出：
    * `quantize`：与原始输入X维度相同，代表离散的code
    * `embed_ind`：选择索引enbed_ind
    * `dist`： 维度为`h, i, j`， 对应head中第i个输入向量与第j个code的相似度
    * `self.embed`：码表`h, codebook_size, dim`