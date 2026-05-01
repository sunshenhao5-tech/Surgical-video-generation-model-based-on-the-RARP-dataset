# logs 目录说明

本目录 `logs/` 存放训练过程中的输出文件和日志。

## 目录结构

```
logs/
|-- train_stage1_rarp/
|   |-- checkpoint-12000/
|   |-- checkpoint-15000/
|   |-- checkpoint-18000/
|   |-- seg_encoder.pth
|   |-- validation_images/
|   `-- (其他训练输出)
`-- train_stage2_rarp/
    |-- seg_encoder.pth
    `-- (其他训练输出)
```

## 主要内容

- **`seg_encoder.pth`**：第一阶段训练生成的分割编码器权重，用于第二阶段训练。
- **`checkpoint-*`**：训练检查点，包含模型状态，可用于恢复训练。
- **`validation_images/`**：训练过程中的验证图片，用于评估模型效果。

## 生成方式

这些文件通过运行训练脚本生成：

1. 第一阶段训练：
   ```bash
   python Training/train_stage1.py
   ```

2. 第二阶段训练：
   ```bash
   python Training/train_stage2.py
   ```

## 注意事项

- **不上传到 GitHub**：这些文件体积大，且是训练结果。项目仓库中已通过 `.gitignore` 忽略此目录。
- **本地保存**：如果需要继续训练或分享模型，请备份 `seg_encoder.pth` 和重要检查点。
- **清理**：训练完成后，可删除不必要的日志文件以节省空间。

如果您需要使用这些权重，请先运行相应训练脚本生成。