from typing import Callable

from qwen3_from_scratch.inference.scheduler import Scheduler, SchedulerConfig
from qwen3_from_scratch.inference.sequence import Sequence, SequenceStatus


def basic_check_seq_finish(seq: Sequence) -> bool:
    return seq.generated_lens >= seq.max_new_tokens


def make_scheduler(
    num_pages: int = 100,
    block_size: int = 16,
    max_num_seqs: int = 8,
    max_num_tokens=100,
    check_seq_finish_func: Callable[[Sequence], bool] = lambda seq: False,
) -> Scheduler:
    config = SchedulerConfig(
        max_num_seqs=max_num_seqs,
        max_num_tokens=max_num_tokens,
        block_size=block_size,
        max_blocks=num_pages,
    )
    return Scheduler(config, check_seq_finish_func)


def make_sequence(num_tokens: int, max_new_tokens: int = 10):
    return Sequence([0] * num_tokens, max_new_tokens=max_new_tokens)


class TestAddRequest:
    def test_empty_queue_when_init(self):
        scheduler = make_scheduler()
        assert len(scheduler.active) == 0
        assert len(scheduler.waiting) == 0

    def test_return_one_seq_when_add(self):
        scheduler = make_scheduler()
        seq = make_sequence(32)
        scheduler.add_request(seq)
        assert len(scheduler.active) == 0
        assert len(scheduler.waiting) == 1

    def test_return_false_when_prompts_more_than_max_num_tokens(self):
        scheduler = make_scheduler(max_num_tokens=100)
        seq = make_sequence(101)
        assert not scheduler.add_request(seq)

    def test_return_false_when_prompts_and_max_new_tokens_more_than_max_tokens(
        self,
    ):
        # 最多160个
        scheduler = make_scheduler(num_pages=10, max_num_tokens=1000)
        seq = make_sequence(100, max_new_tokens=61)
        assert not scheduler.add_request(seq)


class TestSchedule:
    def test_return_no_seq_when_init(self):
        scheduler = make_scheduler()
        seqs = scheduler.schedule()
        assert len(seqs) == 0

    def test_return_added_seq(self):
        scheduler = make_scheduler()
        seq = make_sequence(32)
        scheduler.add_request(seq)
        scheduled_seqs = scheduler.schedule()
        assert len(scheduled_seqs) == 1
        scheduled_seq = scheduled_seqs[0]
        assert scheduled_seq.req_id == seq.req_id

    def test_fill_seq_when_schedule_prefill(self):
        scheduler = make_scheduler()
        seq = make_sequence(32)
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
            scheduler.add_request(make_sequence(2))
        reqs = scheduler.schedule()
        assert len(reqs) == 2

    def test_restricted_by_max_seq_num_twice(self):
        scheduler = make_scheduler(max_num_tokens=1000, max_num_seqs=2)
        for _ in range(100):
            scheduler.add_request(make_sequence(2))
        reqs = scheduler.schedule()
        assert len(reqs) == 2
        reqs = scheduler.schedule()
        assert len(reqs) == 2

    def test_restricted_by_max_token_num(self):
        scheduler = make_scheduler(max_num_tokens=5, max_num_seqs=1000)
        for _ in range(100):
            scheduler.add_request(make_sequence(2))
        reqs = scheduler.schedule()
        assert len(reqs) == 2

    def test_restricted_by_max_token_num_multiple_step(self):
        scheduler = make_scheduler(max_num_tokens=5, max_num_seqs=1000)
        for _ in range(3):
            scheduler.add_request(make_sequence(2))
        reqs = scheduler.schedule()
        assert len(reqs) == 2
        reqs = scheduler.schedule()
        assert len(reqs) == 1

    def test_schedule_prefill_first(self):
        seqs = [make_sequence(32) for _ in range(2)]

        def check_seq_finish_func(seq):
            return seq.req_id == seqs[0].req_id

        scheduler = make_scheduler(check_seq_finish_func=check_seq_finish_func)
        scheduler.add_request(seqs[0])
        scheduler.add_request(seqs[1])
        reqs = scheduler.schedule()
        assert len(reqs) == 2
        scheduler.post_process(reqs, [0, 0])
        # 此时第一个序列完成，第二个变成 decode
        seq3 = make_sequence(32)
        scheduler.add_request(seq3)
        # 混合都有的情况下，优先调度 prefill 序列
        reqs = scheduler.schedule()
        assert len(reqs) == 1
        assert reqs[0].req_id == seq3.req_id

    def test_decode_restricted_by_max_token_num(self):
        scheduler = make_scheduler(max_num_tokens=7, max_num_seqs=1000)
        for _ in range(9):
            scheduler.add_request(make_sequence(2))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0] * len(reqs))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0] * len(reqs))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0] * len(reqs))
        # 此时有9个解码，但受限于长度只能调度7个
        reqs = scheduler.schedule()
        assert len(reqs) == 7

    def test_decode_restricted_by_max_seq_num(self):
        scheduler = make_scheduler(max_num_tokens=1000, max_num_seqs=5)
        for _ in range(9):
            scheduler.add_request(make_sequence(2))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0] * len(reqs))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0] * len(reqs))
        reqs = scheduler.schedule()
        scheduler.post_process(reqs, [0] * len(reqs))
        # 此时有9个解码，但受限于长度只能调度5个
        reqs = scheduler.schedule()
        assert len(reqs) == 5


class TestPostProcess:
    def test_append_token_id(self):
        scheduler = make_scheduler()
        seqs = [make_sequence(32) for _ in range(100)]
        scheduler.post_process(seqs, [0] * len(seqs))
        assert len(seqs[0].token_ids) == 33

    def test_check_seq_finish_func(self):
        seqs = [make_sequence(32) for _ in range(100)]

        def check_seq_finish_func(seq):
            return seq.req_id == seqs[0].req_id

        scheduler = make_scheduler(check_seq_finish_func=check_seq_finish_func)
        scheduler.post_process(seqs, [0] * len(seqs))
        assert seqs[0].status == SequenceStatus.FINISHED

    def test_check_can_schedule_more_when_finish(self):
        seqs = [make_sequence(32, max_new_tokens=16) for _ in range(2)]

        def check_seq_finish_func(seq):
            return seq.req_id == seqs[0].req_id

        scheduler = make_scheduler(
            check_seq_finish_func=check_seq_finish_func, num_pages=3
        )
        scheduler.add_request(seqs[0])
        scheduler.add_request(seqs[1])
        reqs = scheduler.schedule()
        assert len(reqs) == 1
        scheduler.post_process(reqs[:1], [0])
        reqs = scheduler.schedule()
        assert len(reqs) == 1


class TestPreempt:
    def test_decode_preempt_when_block_num_not_enough(self):
        # 4个页面，两个请求，每个预填充占2，占满，下一轮调度变成解码时不足，最新的被强占，第一个继续解码
        scheduler = make_scheduler(num_pages=4)
        seq = make_sequence(32, max_new_tokens=16)
        seq2 = make_sequence(32, max_new_tokens=16)
        scheduler.add_request(seq)
        scheduler.add_request(seq2)
        seqs = scheduler.schedule()
        scheduler.post_process(seqs, [0] * len(seqs))
        seqs = scheduler.schedule()
        assert len(seqs) == 1
        assert seqs[0].req_id == seq.req_id
        assert len(scheduler.waiting) == 1
        assert scheduler.waiting[0].cached_len == 0
        preempted_seq = scheduler.waiting[0]
        assert preempted_seq.cached_len == 0
        assert preempted_seq.status == SequenceStatus.WAITING
        assert preempted_seq.generated_lens == 0
        assert len(preempted_seq.block_tables) == 0

    def test_decode_preempt_finish_and_prefill(self):
        """
        1. 4个页，请求1预填充2，请求2预填充2
        2. 解码时，1不够追加，把2抢占了，2进入预填充
        3. 下一轮调度，收集预填充时根本不够任何，直接继续解码
        4. 2次解码后1结束
        5. 下一次调度能调度2
        """
        scheduler = make_scheduler(
            num_pages=4, check_seq_finish_func=basic_check_seq_finish
        )
        seq = make_sequence(32, max_new_tokens=2)
        seq2 = make_sequence(32, max_new_tokens=16)
        scheduler.add_request(seq)
        scheduler.add_request(seq2)
        seqs = scheduler.schedule()
        # 第一轮调度，两个请求都选出
        assert len(seqs) == 2
        scheduler.post_process(seqs, [0] * len(seqs))
        seqs = scheduler.schedule()
        # 第二轮调度，只有请求1选出，进入解码，2被强占回到等待列表
        assert len(seqs) == 1
        assert seqs[0].req_id == seq.req_id
        assert not seqs[0].is_prefill
        # 1解码完成释放
        scheduler.post_process(seqs, [0] * len(seqs))
        # 第三轮调度，可以得到2的预填充
        seqs = scheduler.schedule()
        assert len(seqs) == 1
        assert seqs[0].req_id == seq2.req_id
        assert seqs[0].is_prefill
