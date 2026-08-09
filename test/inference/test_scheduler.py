from qwen3_from_scratch.inference.scheduler import SchedulerConfig, Scheduler
from qwen3_from_scratch.inference.sequence import Sequence, SequenceStatus
from typing import Callable


def make_scheduler(num_pages: int = 100, block_size: int = 16, max_num_seqs: int = 8, max_num_tokens=100, check_seq_finish_func: Callable[[Sequence], bool] = lambda seq: False) -> Scheduler:
    config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_tokens=max_num_tokens,
        block_size=block_size,
        max_blocks=num_pages
    )
    return Scheduler(config, check_seq_finish_func)


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
        assert not scheduled_seq.is_prefill
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

    def test_schedule_prefill_first(self):
        seqs = [Sequence([0]* 32) for _ in range(2)]
        check_seq_finish_func = lambda seq: seq.req_id == seqs[0].req_id
        scheduler = make_scheduler(check_seq_finish_func=check_seq_finish_func)
        scheduler.add_request(seqs[0])
        scheduler.add_request(seqs[1])
        reqs = scheduler.schedule()
        assert len(reqs) == 2
        scheduler.post_process(reqs, [0,0])
        # 此时第一个序列完成，第二个变成 decode
        seq3 = Sequence([0]*32)
        scheduler.add_request(seq3)
        # 混合都有的情况下，优先调度 prefill 序列
        reqs = scheduler.schedule()
        assert len(reqs) == 1
        assert reqs[0].req_id == seq3.req_id

    def test_decode_restricted_by_max_token_num(self):
        scheduler = make_scheduler(max_num_tokens=7, max_num_seqs=1000)
        for _ in range(9):
            scheduler.add_request(Sequence([0] * 2))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0]*len(reqs))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0]*len(reqs))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0]*len(reqs))
        # 此时有9个解码，但受限于长度只能调度7个
        reqs = scheduler.schedule()
        assert len(reqs) == 7

    def test_decode_restricted_by_max_seq_num(self):
        scheduler = make_scheduler(max_num_tokens=1000, max_num_seqs=5)
        for _ in range(9):
            scheduler.add_request(Sequence([0] * 2))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0]*len(reqs))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0]*len(reqs))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0]*len(reqs))
        # 此时有9个解码，但受限于长度只能调度5个
        reqs = scheduler.schedule()
        assert len(reqs) == 5


class TestPostProcess:
    def test_append_token_id(self):
        scheduler = make_scheduler()
        seqs = [Sequence([0]* 32) for _ in range(100)]
        scheduler.post_process(seqs, [0]*len(seqs))
        assert len(seqs[0].token_ids) == 33

    def test_check_seq_finish_func(self):
        seqs = [Sequence([0]* 32) for _ in range(100)]
        check_seq_finish_func = lambda seq: seq.req_id == seqs[0].req_id
        scheduler = make_scheduler(check_seq_finish_func=check_seq_finish_func)
        scheduler.post_process(seqs, [0]*len(seqs))
        assert seqs[0].status == SequenceStatus.FINISHED

    def test_check_can_schedule_more_when_finish(self):
        seqs = [Sequence([0]* 32) for _ in range(2)]
        check_seq_finish_func = lambda seq: seq.req_id == seqs[0].req_id
        scheduler = make_scheduler(check_seq_finish_func=check_seq_finish_func, num_pages=2)
        scheduler.add_request(seqs[0])
        scheduler.add_request(seqs[1])
        reqs = scheduler.schedule()
        assert len(reqs) == 1
        scheduler.post_process(reqs[:1],[0])
        reqs = scheduler.schedule()
        assert len(reqs) == 1