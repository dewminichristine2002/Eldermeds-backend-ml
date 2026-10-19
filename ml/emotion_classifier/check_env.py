import sys, traceback
print('PYTHON_EXECUTABLE:', sys.executable)
print('PYTHON_VERSION:', sys.version.replace('\n',' '))

# Torch check
try:
    import torch
    print('TORCH_INSTALLED: True')
    try:
        print('TORCH_VERSION:', torch.__version__)
        print('TORCH_FILE:', getattr(torch, '__file__', 'n/a'))
        try:
            print('CUDA_AVAILABLE:', torch.cuda.is_available())
        except Exception as e:
            print('CUDA_CHECK_ERROR:', type(e).__name__, e)
    except Exception as e:
        print('TORCH_IMPORT_ERROR:', type(e).__name__, e)
        traceback.print_exc()
except ModuleNotFoundError:
    print('TORCH_INSTALLED: False')
except Exception as e:
    print('TORCH_IMPORT_ERROR:', type(e).__name__, e)
    traceback.print_exc()

# sentence-transformers check
try:
    import sentence_transformers as st
    print('SENTENCE_TRANSFORMERS_VERSION:', getattr(st, '__version__', 'n/a'))
except ModuleNotFoundError:
    print('SENTENCE_TRANSFORMERS_INSTALLED: False')
except Exception as e:
    print('SENTENCE_TRANSFORMERS_IMPORT_ERROR:', type(e).__name__, e)
    traceback.print_exc()
