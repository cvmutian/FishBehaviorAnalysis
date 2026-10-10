# 鱼体行为分析

鱼体行为分析，并统计鱼群行为占比。

## 目录

```text
config/config.yaml
data/
features/
models/
train.py
test.py
predictor.py
online_analyzer.py
demo.py
```

## 运行

```bash
pip install -r requirements.txt
python -m data.build_samples --input <序列目录> --config config/config.yaml
python train.py --config config/config.yaml
python test.py --config config/config.yaml
```
