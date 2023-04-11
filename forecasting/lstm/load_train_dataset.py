from argparse import ArgumentParser

from data.utils import *
from data.paths import PATHS

argparser = ArgumentParser()
argparser.add_argument("--dataset", default="av2", choices=["av2", "nuscenes"])
config = argparser.parse_args()

if config.dataset == "av2":
    from data.av2_adapter import *

    classes = AV2_CLASS_NAMES
    infos_path = PATHS["av2"]["train"]["infos"]
    dataset_dir = PATHS["av2"]["train"]["dataset_dir"]
    raw_labels = load(infos_path)
    label_list = unpack_labels(raw_labels)
    labels = group_frames(label_list)
    city_SE3_by_seq_id = read_city_SE3_ego_by_seq_id(dataset_dir, seq_ids=labels.keys())
    labels = transform_to_global_reference(labels, city_SE3_by_seq_id)
    save(labels, "dataset/av2-train/labels.pkl")
elif config.dataset == "nuscenes":
    from nuscenes import NuScenes

    from data.nuscenes_adapter import *

    classes = NUSCENES_CLASS_NAMES
    infos_path = PATHS["nuscenes"]["train"]["infos"]
    dataset_dir = PATHS["nuscenes"]["train"]["dataset_dir"]
    raw_labels_dict = load(infos_path)
    raw_labels, nusc_version = (
        raw_labels_dict["infos"],
        raw_labels_dict["metadata"]["version"],
    )
    nusc = NuScenes(nusc_version, dataroot=PATHS["nuscenes"]["train"]["dataset_dir"])
    raw_labels = list(sorted(raw_labels, key=lambda e: e["timestamp"]))

    label_list = unpack_and_annotate_labels(raw_labels, nusc, classes)
    labels = group_frames(label_list)
    labels = transform_to_global_reference(labels)
    save(labels, "dataset/nuscenes-train/labels.pkl")
