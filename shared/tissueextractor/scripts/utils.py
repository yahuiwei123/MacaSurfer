from pynvml import *

def detect_available_device(mem_size):
    nvmlInit()
    deviceCount = nvmlDeviceGetCount()

    # check gpu with mem_size GB memory available
    available_device = {}

    for i in range(deviceCount):
        handle = nvmlDeviceGetHandleByIndex(i)
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free / 1024 ** 3 > mem_size:
            available_device[i] = info.free / 1024 ** 3

    # sort by available memory
    available_device = sorted(available_device.items(),  key=lambda d: d[1], reverse=True)
    available_device = [str(item[0]) for item in available_device]
    
    if len(available_device) < 1:
        raise ValueError(f"No device with {mem_size} GB memory available!")
    return available_device
