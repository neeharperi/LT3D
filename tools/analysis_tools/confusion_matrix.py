import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pdb

for class_type in ["vehicle_acc", "pedestrian_acc", "movable_acc", "all_acc"]:
    if class_type == "vehicle_acc":
        ticks = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)"]

    elif class_type == "pedestrian_acc":
        ticks = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)"]

    elif class_type == "movable_acc":
        ticks = ["(a)", "(b)", "(c)", "(d)"]

    elif class_type == "all_acc":
        ticks = ["(a)", "(b)", "(c)", "(d)", "(e)", "(f)", "(g)", "(h)", "(i)", "(j)", "(k)", "(l)", "(m)", "(n)", "(o)", "(p)", "(q)", "(r)", "(s)"]

    filename = "/home/nperi/Workspace/DAMO/models/CenterPoint/nusc_centerpoint_finegrain_wc_detection/{}.csv".format(class_type)
    dataFrame = pd.read_csv(filename, index_col=0).round(2)
    sns.set(font_scale=0.25)
    sns.heatmap(dataFrame, annot=True, xticklabels = ticks, yticklabels = ticks)

    plt.savefig("Figures/{}.pdf".format(class_type), bbox_inches='tight')
    plt.clf()