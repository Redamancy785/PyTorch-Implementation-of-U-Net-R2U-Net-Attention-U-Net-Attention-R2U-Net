<a name="JwUPt"></a>
## 该项目使用PyTorch实现了U-Net、R2U-Net、Attention U-Net以及Attention R2U-Net模型的训练。同时，对这四个模型的关键参数进行了详细的分析和比较，旨在更全面地评估各个模型的优缺点。
注1：为了防止代码运行中出现路径检索错误，请将项目下载至新建的**ISIC**文件目录之下<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/21820237/1706884484205-ea376c7b-8bab-4ca4-891d-96fc7500d427.png#averageHue=%23fbf9f8&clientId=uffba3bf4-4186-4&from=paste&height=375&id=ud933ff88&originHeight=540&originWidth=818&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=61230&status=done&style=none&taskId=uadb96c6c-0387-47f9-82f8-be3082e2b07&title=&width=567.4000244140625)<br />注2：自行创建**dataset**、**models**、**result**文件夹<br />注3：**训练结果**文件夹为本人实验所得数据，仅供参考<br />注4：运行环境**python3.9**<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/21820237/1706927257238-c0d5a2b6-b96f-4af3-80bd-e934523c1f92.png#averageHue=%23dbe2db&clientId=u25d34394-3259-4&from=paste&height=397&id=ub8baff11&originHeight=496&originWidth=1095&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=55770&status=done&style=none&taskId=u6d649412-01cd-498b-8c8f-925368299ef&title=&width=876)

<a name="mufJU"></a>
# 数据集下载
数据集使用**ISIC-2018**数据集：[https://challenge.isic-archive.com/data/#2018](https://challenge.isic-archive.com/data/#2018)<br />数据集被分为三个子集，分别为训练集、验证集和测试集，其比例分别占整个数据集的70%、10%和20%。整个数据集包含 2594 张图像，其中 1815 张图像用于训练，259 张用于验证，520 张用于测试模型。<br />**Step1**：下载如下框选文件<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/21820237/1706882998279-b095698c-6fad-4141-8872-ec8581e30fc2.png#averageHue=%23fbfaf9&clientId=uffba3bf4-4186-4&from=paste&height=205&id=u74c32ede&originHeight=537&originWidth=1620&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=113273&status=done&style=none&taskId=ua4ce2aa6-9acd-4e4a-9d26-198116592b1&title=&width=617.4000244140625)<br />**Step2**：解压缩后文件名如下所示，并将两个文件夹存放至dataset文件目录之下，无需其他操作<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/21820237/1706883240550-808fd460-9b0f-4044-9646-776c6acc7336.png#averageHue=%23fbf9f7&clientId=uffba3bf4-4186-4&from=paste&height=56&id=u84a35588&originHeight=70&originWidth=843&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=8135&status=done&style=none&taskId=u8c38396c-ef35-4408-88ee-71933b9dcd2&title=&width=674.4)

<a name="RNN0Y"></a>
# 代码运行步骤
**Step1**：单独运行**dataset.py**文件，对数据集进行处理<br />**Step2**：配置**main.py**文件<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/21820237/1706885131681-b7b21e39-c310-4025-a19a-f422b40afac5.png#averageHue=%23fcf8f7&clientId=uffba3bf4-4186-4&from=paste&height=427&id=u62ea1e1e&originHeight=534&originWidth=1078&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=73617&status=done&style=none&taskId=uf597efb0-e5d1-4f1d-ad93-b1a258cf65a&title=&width=862.4)<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/21820237/1706885261905-27164510-7d16-46ff-85c1-8e240d3d000c.png#averageHue=%23f9f6f5&clientId=uffba3bf4-4186-4&from=paste&height=383&id=u4feee0e5&originHeight=479&originWidth=1146&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=88746&status=done&style=none&taskId=u89edc979-e4ee-460d-9252-d87374fbcad&title=&width=916.8)<br />**Step3**：每次运行完毕检查输出结果<br />![image.png](https://cdn.nlark.com/yuque/0/2024/png/21820237/1706885337722-733e8880-fd63-4070-b5b3-c21b5d15886c.png#averageHue=%23fbf4f3&clientId=uffba3bf4-4186-4&from=paste&height=418&id=Fl6EB&originHeight=522&originWidth=794&originalType=binary&ratio=1.25&rotation=0&showTitle=false&size=58280&status=done&style=none&taskId=uecc18f01-0aa3-4a1c-8b1a-c405c99b20f&title=&width=635.2)
<a name="JZFTf"></a>
# 模型结构介绍
<a name="ZEz8B"></a>
## U-Net
[![](https://github.com/LeeJunHyun/Image_Segmentation/raw/master/img/U-Net.png#from=url&id=iZhyF&originHeight=278&originWidth=418&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=)](https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/U-Net.png)
<a name="x47WV"></a>
## R2U-Net
[![](https://github.com/LeeJunHyun/Image_Segmentation/raw/master/img/R2U-Net.png#from=url&id=dbA4K&originHeight=335&originWidth=960&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=)](https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/R2U-Net.png)
<a name="A9ipX"></a>
## Attention U-Net
[![](https://github.com/LeeJunHyun/Image_Segmentation/raw/master/img/AttU-Net.png#from=url&id=nUv0K&originHeight=822&originWidth=1272&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=)](https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/AttU-Net.png)
<a name="FfbkN"></a>
## Attention R2U-Net
[![](https://github.com/LeeJunHyun/Image_Segmentation/raw/master/img/AttR2U-Net.png#from=url&id=lqcCP&originHeight=522&originWidth=1500&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=)](https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/AttR2U-Net.png)
<a name="f1yW3"></a>
# 模型评估结果
[![](https://github.com/LeeJunHyun/Image_Segmentation/raw/master/img/Evaluation.png#from=url&id=toXKM&originHeight=673&originWidth=1670&originalType=binary&ratio=1.25&rotation=0&showTitle=false&status=done&style=none&title=)](https://github.com/LeeJunHyun/Image_Segmentation/blob/master/img/Evaluation.png)
