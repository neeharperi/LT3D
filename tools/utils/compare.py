import pandas as pd 
from matplotlib import pyplot as plt 
import seaborn as sns 
import pdb 

class_names = [
    'REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG', 'AVERAGE_METRICS']

'''
for max_range in [50, 100, 150]:
    base = "/home/nperi/Workspace/LT3D/mmdetection3d-lt3d/results/torchbox3d_{}m.csv".format(max_range)
    wide = "/home/nperi/Workspace/LT3D/mmdetection3d-lt3d/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_wide_tta_20e_av2/results_{}m.csv".format(max_range)
    hierarchy = "/home/nperi/Workspace/LT3D/mmdetection3d-lt3d/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_wide_hierarchy_tta_20e_av2/results_{}m.csv".format(max_range)
    rgb = "/home/nperi/Workspace/LT3D/mmdetection3d-lt3d/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_wide_hierarchy_tta_20e_av2/results_filter_{}m.csv".format(max_range)

    base = pd.read_csv(base)
    wide = pd.read_csv(wide)
    hierarchy = pd.read_csv(hierarchy)
    rgb = pd.read_csv(rgb)

    dataFrame = {"Category" : [],
                 "Method" : [],
                 "AP" : []}

    for i in range(len(class_names)):
        dataFrame["Method"].append("Baseline")
        dataFrame["Category"].append(base.iloc[i][0])
        dataFrame["AP"].append(base.iloc[i][1])

        dataFrame["Method"].append("26-Class")
        dataFrame["Category"].append(wide.iloc[i][0])
        dataFrame["AP"].append(wide.iloc[i][1])

        dataFrame["Method"].append("30-Class")
        dataFrame["Category"].append(hierarchy.iloc[i][0])
        dataFrame["AP"].append(hierarchy.iloc[i][1])

        dataFrame["Method"].append("30-Class + RGB")
        dataFrame["Category"].append(rgb.iloc[i][0])
        dataFrame["AP"].append(rgb.iloc[i][1])

    dataFrame = pd.DataFrame.from_dict(dataFrame)
    ax = sns.barplot(data=dataFrame, x="AP", y="Category", hue="Method")
    plt.title("AV2 CenterPoint Trained @ 50m, Evaluated @ {}m (AP)".format(max_range))

    plt.savefig("evaluate_{}m.png".format(max_range), bbox_inches="tight")
    plt.clf()
'''

m50 = "/home/nperi/Workspace/LT3D/mmdetection3d-lt3d/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_wide_tta_20e_av2/results_{}m.csv".format(50)
m100 = "/home/nperi/Workspace/LT3D/mmdetection3d-lt3d/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_wide_tta_20e_av2/results_{}m.csv".format(100)
m150 = "/home/nperi/Workspace/LT3D/mmdetection3d-lt3d/work_dirs/centerpoint_0075voxel_second_secfpn_dcn_4x8_cyclic_wide_tta_20e_av2/results_{}m.csv".format(150)

m50 = pd.read_csv(m50)
m100 = pd.read_csv(m100)
m150 = pd.read_csv(m150)

dataFrame = {"Category" : [],
            "Range" : [],
            "AP" : []}

for i in range(len(class_names)):
    dataFrame["Range"].append("50m")
    dataFrame["Category"].append(m50.iloc[i][0])
    dataFrame["AP"].append(m50.iloc[i][1])

    dataFrame["Range"].append("100m")
    dataFrame["Category"].append(m100.iloc[i][0])
    dataFrame["AP"].append(m100.iloc[i][1])

    dataFrame["Range"].append("150m")
    dataFrame["Category"].append(m150.iloc[i][0])
    dataFrame["AP"].append(m150.iloc[i][1])

dataFrame = pd.DataFrame.from_dict(dataFrame)
ax = sns.barplot(data=dataFrame, x="AP", y="Category", hue="Range")
plt.title("26-Class AV2 CenterPoint Trained @ 50m")

plt.savefig("evaluate_range.png", bbox_inches="tight")
plt.clf()
