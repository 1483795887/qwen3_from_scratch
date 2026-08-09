import multiprocessing

from qwen3_from_scratch.factory import BatchConfig
from qwen3_from_scratch.inference.model_manager import ModelManager
from qwen3_from_scratch.inference.kv_cache.paged_cache import PagedKVCache

from qwen3_from_scratch.inference.logger import get_logger

logger = get_logger(__name__)


class ModelWorker:
    def __init__(self, config: BatchConfig, model_name:str):
        self.model, self.kv_cache = self._init_model(config, model_name)

    def _init_model(self, config : BatchConfig, model_name:str):
        if model_name not in config.list_model_names():
            raise KeyError(f"模型 {model_name} 不可用")
        model_manager = ModelManager(config)
        model = model_manager.load_model(model_name)
        model_info = config.get_model(model_name)
        ava_mem = PagedKVCache.get_available_mem()
        alloc_mem = int(config.scheduler.gpu_utilization * ava_mem)
        model_config = model.config
        blocks = PagedKVCache.get_block_num(alloc_mem, model_config.num_hidden_layers, model_config.num_key_value_heads,
                                            model_config.hidden_size)
        kv_cache = PagedKVCache(blocks, model_config.num_hidden_layers, model_config.num_key_value_heads,
                                model_config.hidden_size, device=model_info.device,
                                dtype=config.kv_cache_dtype)

        return model, kv_cache

    @staticmethod
    def run(config: BatchConfig, model_name: str, request_mp: multiprocessing.Queue, result_mp: multiprocessing.Queue,
            get_blocks_mp: multiprocessing.Queue):
        worker = ModelWorker(config, model_name)
        get_blocks_mp.put(worker.kv_cache.num_pages)

        while True:
            reqs = request_mp.get()

            result_mp.put([1] * len(reqs))
