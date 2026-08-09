from qwen3_from_scratch.inference.scheduler import SchedulerConfig, Scheduler
from qwen3_from_scratch.inference.sequence import Sequence, SequenceStatus


def make_scheduler(num_pages: int = 100, block_size: int = 16, max_num_seqs: int = 8, max_num_tokens=100) -> Scheduler:
    config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_tokens=max_num_tokens,
        block_size=block_size,
        max_blocks=num_pages
    )
    return Scheduler(config)


class TestAddRequest:
    def test_empty_queue_when_init(self):
        scheduler = make_scheduler()
        assert len(scheduler.active) == 0
        assert len(scheduler.waiting) == 0

    def test_return_one_seq_when_add(self):
        scheduler = make_scheduler()
        seq = Sequence([0] * 32)
        scheduler.add_request(seq)
        assert len(scheduler.active) == 0
        assert len(scheduler.waiting) == 1

    def test_return_false_when_prompts_more_than_max_num_tokens(self):
        scheduler = make_scheduler(max_num_tokens=100)
        seq = Sequence([0]* 101)
        assert not scheduler.add_request(seq)

class TestSchedule:
    def test_return_no_seq_when_init(self):
        scheduler = make_scheduler()
        seqs = scheduler.schedule()
        assert len(seqs) == 0

    def test_return_added_seq(self):
        scheduler = make_scheduler()
        seq = Sequence([0]* 32)
        scheduler.add_request(seq)
        scheduled_seqs = scheduler.schedule()
        assert len(scheduled_seqs) == 1
        scheduled_seq = scheduled_seqs[0]
        assert scheduled_seq.req_id == seq.req_id

    def test_fill_seq_when_schedule_prefill(self):
        scheduler = make_scheduler()
        seq = Sequence([0] * 32)
        assert seq.status == SequenceStatus.WAITING
        scheduler.add_request(seq)
        scheduled_seqs = scheduler.schedule()
        assert len(scheduled_seqs) == 1
        scheduled_seq = scheduled_seqs[0]
        assert scheduled_seq.req_id == seq.req_id
        assert scheduled_seq.block_tables is not None
        assert scheduled_seq.is_prefill
        assert scheduled_seq.status == SequenceStatus.RUNNING

    def test_restricted_by_max_seq_num(self):
        scheduler = make_scheduler(max_num_tokens=1000, max_num_seqs=2)
        for _ in range(100):
            scheduler.add_request(Sequence([0]*2))
        reqs = scheduler.schedule()
        assert len(reqs) == 2
        
    def test_restricted_by_max_seq_num_twice(self):
        scheduler = make_scheduler(max_num_tokens=1000, max_num_seqs=2)
        for _ in range(100):
            scheduler.add_request(Sequence([0]*2))
        reqs = scheduler.schedule()
        assert len(reqs) == 2
        reqs = scheduler.schedule()
        assert len(reqs) == 2

    def test_restricted_by_max_token_num(self):
        scheduler = make_scheduler(max_num_tokens=5, max_num_seqs=1000)
        for _ in range(100):
            scheduler.add_request(Sequence([0] * 2))
        reqs = scheduler.schedule()
        assert len(reqs) == 2

    def test_restricted_by_max_token_num_multiple_step(self):
        scheduler = make_scheduler(max_num_tokens=5, max_num_seqs=1000)
        for _ in range(3):
            scheduler.add_request(Sequence([0] * 2))
        reqs = scheduler.schedule()
        assert len(reqs) == 2
        reqs = scheduler.schedule()
        assert len(reqs) == 1

    def test_restricted_by_block_num(self):
        scheduler = make_scheduler(num_pages=2)
        scheduler.add_request(Sequence([0]* 32))
        scheduler.add_request(Sequence([0]* 32))
        reqs = scheduler.schedule()
        assert len(reqs) == 1

    def test_restricted_by_block_num_multiple_step(self):
        scheduler = make_scheduler(num_pages=2)
        scheduler.add_request(Sequence([0]* 32))
        scheduler.add_request(Sequence([0]* 32))
        reqs = scheduler.schedule()
        assert len(reqs) == 1
        reqs = scheduler.schedule()
        assert len(reqs) == 0

class TestPostProcess:
    pass
