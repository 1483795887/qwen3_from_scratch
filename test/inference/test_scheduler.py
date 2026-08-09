from qwen3_from_scratch.inference.scheduler import SchedulerConfig, Scheduler


def make_scheduler(num_pages: int = 100, block_size: int = 16, max_num_seqs: int = 8, max_num_tokens=100) -> Scheduler:
    config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_tokens=max_num_tokens,
        block_size=block_size,
        max_blocks=num_pages
    )
    return Scheduler(config)


class TestAddRequest:
    pass


class TestSchedule:
    def test_return_no_seq_when_init(self):
        scheduler = make_scheduler()
        seqs = scheduler.schedule()
        assert len(seqs) == 0

class TestPostProcess:
    pass
