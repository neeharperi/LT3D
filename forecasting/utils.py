import numpy as np
num_timesteps = 6
time_delta = 0.5 
nelem = 101
n_jobs = 8

nus_classes = [
    'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle', 'motorcycle', 'emergency_vehicle',
    'adult', 'child', 'police_officer', 'construction_worker', 'stroller', 'personal_mobility', 
    'pushable_pullable', 'debris', 'traffic_cone', 'barrier'
]

av2_classes = [
    'REGULAR_VEHICLE', 'PEDESTRIAN', 'BICYCLIST', 'MOTORCYCLIST', 'WHEELED_RIDER',
    'BOLLARD', 'CONSTRUCTION_CONE', 'SIGN', 'CONSTRUCTION_BARREL', 'STOP_SIGN', 'MOBILE_PEDESTRIAN_CROSSING_SIGN',
    'LARGE_VEHICLE', 'BUS', 'BOX_TRUCK', 'TRUCK', 'VEHICULAR_TRAILER', 'TRUCK_CAB', 'SCHOOL_BUS', 'ARTICULATED_BUS',
    'MESSAGE_BOARD_TRAILER', 'BICYCLE', 'MOTORCYCLE', 'WHEELED_DEVICE', 'WHEELCHAIR', 'STROLLER', 'DOG'
]

nus_velocity = {
    'car': 1.79,
    'truck': 1.38,
    'trailer': 0.68,
    'bus': 2.68,
    'construction_vehicle': 0.16,
    'bicycle': 0.72,
    'motorcycle': 1.88,
    'emergency_vehicle': 1.44,
    'adult': 0.89,
    'child': 0.39,
    'police_officer': 0.81,
    'construction_worker': 0.34,
    'stroller': 0.70,
    'personal_mobility': 0.04,
    'pushable_pullable': 0.10,
    'debris': 0.05,
    'traffic_cone': 0.05,
    'barrier': 0.06
 }

av2_velocity = {
    'REGULAR_VEHICLE': 2.36,
    'PEDESTRIAN': 0.80,
    'BICYCLIST': 3.61,
    'MOTORCYCLIST': 4.08,
    'WHEELED_RIDER': 2.03,
    'BOLLARD': 0.02,
    'CONSTRUCTION_CONE': 0.02,
    'SIGN': 0.05,
    'CONSTRUCTION_BARREL': 0.03,
    'STOP_SIGN': 0.09,
    'MOBILE_PEDESTRIAN_CROSSING_SIGN': 0.03,
    'LARGE_VEHICLE': 1.56,
    'BUS': 3.10,
    'BOX_TRUCK': 2.59,
    'TRUCK': 2.76,
    'VEHICULAR_TRAILER': 1.72,
    'TRUCK_CAB': 2.36,
    'SCHOOL_BUS': 4.44,
    'ARTICULATED_BUS': 4.58,
    'MESSAGE_BOARD_TRAILER': 0.41,
    'BICYCLE': 0.97,
    'MOTORCYCLE': 1.58,
    'WHEELED_DEVICE': 0.37,
    'WHEELCHAIR': 1.50,
    'STROLLER': 0.91,
    'DOG': 0.72
 }

forecast_scalar = np.linspace(0, 1, num_timesteps + 1)
dist_th = [0.5, 1, 2, 4]
velocity_profile = ["static", "linear", "non-linear"]

def agent_velocity(agent):
    if "future_translation" in agent: #ground_truth
        return (agent['future_translation'][0][:2] - agent['current_translation'][:2]) / time_delta

    else: #predictions
        res = []
        for i in range(agent["prediction"].shape[0]):
            res.append((agent['prediction'][i][0][:2] - agent['current_translation'][:2]) / time_delta)
        
        return res 
    
def trajectory_type(agent, class_velocity):
    if "future_translation" in agent: #ground_truth
        time = agent['future_translation'].shape[0] * time_delta
        static_target = agent['current_translation'][:2]
        linear_target = agent['current_translation'][:2] + time * agent['velocity'][:2]
        
        final_position = agent['future_translation'][-1][:2]
        
        threshold = 1 + forecast_scalar[len(agent['future_translation'])] * class_velocity.get(agent["name"], 0)
        if np.linalg.norm(final_position - static_target) < threshold:
            return "static"
        elif np.linalg.norm(final_position - linear_target) < threshold:
            return "linear"
        else:  
            return "non-linear" 
        
    else: #predictions
        res = []
        time = agent['prediction'].shape[1] * time_delta

        threshold = 1 + forecast_scalar[len(agent['prediction'])] * class_velocity.get(agent["name"], 0)
        for i in range(agent["prediction"].shape[0]):
            static_target = agent['current_translation'][:2]
            linear_target = agent['current_translation'][:2] + time * agent['velocity'][i][:2]
            
            final_position = agent['prediction'][i][-1][:2]
            
            if np.linalg.norm(final_position - static_target) < threshold:
                res.append("static")
            elif np.linalg.norm(final_position - linear_target) < threshold:
                res.append("linear")
            else:  
                res.append("non-linear") 
        
        return res

def center_distance(pred_box, gt_box):
    return np.linalg.norm(pred_box - gt_box)