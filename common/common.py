import time
import numpy as np
import logging
import common

log_level = logging.DEBUG


def timefn2(fn):
    # @wraps(fn)
    def measure_time(*args, **kwargs):
        t1 = time.time()
        result = fn(*args, **kwargs)
        t2 = time.time()
        # logger = logging.getLogger(LOGGER_NAME)
        if log_level in [logging.DEBUG]:
            logger = common.get_runtime_logger()
            time_str = "@timefn: {} took {} secons".format(fn.__name__, t2 - t1)
            if logger is not None:
                logger.info(time_str)
            else:
                print(time_str)
        return result
    return measure_time


def apply_mask(image, mask, color, alpha=0.5):
    """
    Apply the given mask to the image.
    """
    for c in range(3):
        image[:, :, c] = np.where(mask > 0,
                                  image[:, :, c] *
                                  (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image

@timefn2
def load_camera_config(calib_path):
    import pickle
    with open(calib_path, 'r') as f:
        camera = pickle.load(f)

    return camera