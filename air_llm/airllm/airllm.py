

import time
from .airllm_base import AirLLMBaseModel
from .profiler import LayeredProfiler
from .utils import clean_memory



class AirLLMLlama2(AirLLMBaseModel):
    def __init__(self, *args, **kwargs):
        super(AirLLMLlama2, self).__init__(*args, **kwargs)
        self.profiler = LayeredProfiler(print_memory=True)

    def inference(self, *args, **kwargs):
        self.profiler.add_profiling_time('start_inference', time.time())
        result = super(AirLLMLlama2, self).inference(*args, **kwargs)
        self.profiler.add_profiling_time('end_inference', time.time())
        self.profiler.print_profiling_time()
        clean_memory()
        return result

