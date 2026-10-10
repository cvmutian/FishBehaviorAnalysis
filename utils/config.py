from pathlib import Path

import torch
import yaml


MODEL_NAME = "multistream_fbpem"
REQUIRED_BEHAVIORS = ["低活动", "正常游动", "快速游动", "急转避让"]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.is_file():
        alt = project_root() / path
        if alt.is_file():
            config_path = alt
        else:
            raise FileNotFoundError(f"找不到配置文件: {path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    validate_config(config)
    return config


def resolve_path(root: Path, path: str) -> Path:
    target = Path(path)
    if target.is_absolute():
        return target
    return root / target


def select_device(name: str) -> torch.device:
    if name is None or name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def validate_config(config: dict) -> None:
    names = list(config["behavior_names"])
    if names != REQUIRED_BEHAVIORS or int(config["num_classes"]) != 4:
        raise ValueError("行为类别必须依次为低活动、正常游动、快速游动、急转避让")
    if int(config["num_keypoints"]) != 11:
        raise ValueError("关键点数量必须为 11")
    if int(config["window_size"]) <= 1 or int(config["stride"]) <= 0:
        raise ValueError("window_size 必须大于 1，stride 必须为正整数")
    if float(config["fps"]) <= 0:
        raise ValueError("fps 必须为正数")

    keypoints = config["keypoints"]
    if list(keypoints["names"]) != [
        "嘴巴", "眼睛", "鳃盖后缘上部", "背鳍前部", "尾鳍前端上部",
        "尾鳍顶部", "尾鳍中部", "尾鳍底部", "尾鳍前端下部", "臀鳍前部", "鳃盖后缘下部",
    ]:
        raise ValueError("关键点名称或顺序与 FishPose 定义不一致")
    if int(keypoints["mouth_index"]) != 0 or int(keypoints["tail_index"]) != 6:
        raise ValueError("鱼嘴编号必须为 0，尾鳍中部编号必须为 6")
    if [int(item) for item in keypoints["gill_indices"]] != [2, 10]:
        raise ValueError("鳃盖关键点必须为 2 和 10")
    if [int(item) for item in keypoints["peduncle_indices"]] != [4, 8]:
        raise ValueError("尾柄关键点必须为 4 和 8")
    parts = keypoints["parts"]
    for index in (2, 10):
        if index not in parts["head"] or index not in parts["trunk"]:
            raise ValueError("鳃盖关键点需要同时参与头部和躯干")
    for index in (4, 8):
        if index not in parts["trunk"] or index not in parts["tail"]:
            raise ValueError("尾柄关键点需要同时参与躯干和尾部")
    for edge in keypoints["skeleton"]:
        if len(edge) != 2 or not (0 <= int(edge[0]) < 11 and 0 <= int(edge[1]) < 11):
            raise ValueError("骨架连线存在无效关键点编号")

    model = config["model"]
    embed_sum = int(model["coord_embed"]) + int(model["motion_embed"]) + int(model["geometry_embed"])
    if embed_sum != int(model["fusion_channels"]):
        raise ValueError("融合通道必须等于三流嵌入通道之和")
    fbpem_cfg = model["fbpem"]
    if fbpem_cfg["pooling"] != "mean" or int(fbpem_cfg["mapping_layers"]) != 1:
        raise ValueError("FBPEM 各部位必须平均池化，并各自使用 1 层特征映射")
    if "part_channels" in fbpem_cfg and int(fbpem_cfg["part_channels"]) <= 0:
        raise ValueError("FBPEM 部位映射通道必须为正整数")
    if int(model["pqtgm"]["out_channels"]) != int(model["fusion_channels"]):
        raise ValueError("PQTGM 输出通道必须等于融合通道")
    dilations = [int(item) for item in model["mdtcm"]["dilations"]]
    if int(model["mdtcm"]["layers"]) != 2 or int(model["mdtcm"]["branches"]) != len(dilations):
        raise ValueError("MDTCM 层数或分支数与配置不一致")
    if [int(item) for item in model["classifier_channels"]] != [int(model["fusion_channels"]), 64, int(config["num_classes"])]:
        raise ValueError("分类头必须为 [融合通道, 64, 类别数]")
    if [int(item) for item in model["regressor_channels"]] != [int(model["fusion_channels"]), 64, 5]:
        raise ValueError("辅助回归头必须为 [融合通道, 64, 5]")
    for key, value in config["aux_norm"].items():
        if float(value) <= 0:
            raise ValueError(f"aux_norm.{key} 必须为正数")

