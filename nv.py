from pynvml import *
nvmlInit()

#gpuObj = nvmlDeviceGetHandleByIndex(0)

handle = nvmlDeviceGetHandleByIndex(0)

totalMemory = nvmlDeviceGetMemoryInfo(handle)
print totalMemory.total / 1024. / 1024.
print totalMemory.free / 1024. / 1024.
print totalMemory.used / 1024. / 1024.
