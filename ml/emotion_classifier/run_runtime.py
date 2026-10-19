import traceback
from model_runtime import load_runtime
r = load_runtime()
print('runtime.error:', r.error)
print('model_version:', r.model_version)
print('ready:', r.ready)
if r.error:
    traceback.print_exc()
