使用bge模型预测MCQ多项选择题中的错误选项与与误解的亲和力
具体实现步骤：
1.使用meta-llama/Llama-3-8b-instruct生成错误选项背后的误解机制
2.使用生成的误解和误解表Misconception进行向量相似度计算，返回前25个，构造bge微调的正负样本
3.bge微调
4.使用微调的bge模型进行推理，我上传到kaggle上面
下载路径为：

终端运行：
!python bge_data_prepare.py --config_path conf_bge/conf_bge.yaml
终端运行：
!python bge_train.py --config_path conf_bge/conf_bge.yaml
