"""
Loads external datasets and converts into the following format:
{
    sequence_id: [
        {
            "timestamp_ns": int, # nano seconds
            "seq_id": str|int,
            "translation": np.ndarray,
            "size": np.ndarray,
            "yaw": np.ndarray,
            "velocity": np.ndarray,
            "label": np.ndarray,
            "score": np.ndarray,
            "name": np.ndarray,
            ...
        }
    ]
}
"""
