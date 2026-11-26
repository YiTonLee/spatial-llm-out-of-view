"""Cost accounting (E0): time every stage of answering one question, and find the accuracy-cost frontier.

Plain: to say a note is "cheaper" than video we must count everything the note needs, not just the words the model
reads: loading the frames, turning them into picture tokens, running the tools, reading the prompt and writing the
answer. E0 showed why: turning 8 frames into picture tokens (86 ms) is about half of the whole answer time, and the
tools need that step too, so a note 3 times shorter saved only 12% of the time.

Technical: `tsync` is a CUDA-synchronised median timer. A condition's cost is the sum of its stage medians, in two
settings: 'A' one-shot (all frame work paid at question time) and 'B' streaming (frame work already paid as frames
arrived). `frontier` returns the conditions no other condition beats on both cost and accuracy.
"""
import statistics, time


def tsync(fn, reps=5, warm=2):
    """Median wall time of fn() in milliseconds, with CUDA synchronised before and after each call."""
    import torch
    for _ in range(warm):
        fn()
    ts = []
    for _ in range(reps):
        torch.cuda.synchronize(); t0 = time.perf_counter(); fn(); torch.cuda.synchronize()
        ts.append(1000 * (time.perf_counter() - t0))
    return statistics.median(ts)


def total(stages, med, setting='A', frame_stages=('decode', 'vit8', 'est')):
    """Cost of one condition: sum of the stage medians it uses; setting 'B' drops the per-frame stages."""
    return sum(med[s] for s in stages if setting == 'A' or s not in frame_stages)


def frontier(cost, acc):
    """Keys of `cost` not dominated by any other key: no other is at most as costly and at least as accurate,
    and strictly better on one of the two."""
    return [k for k in cost if not any(cost[j] <= cost[k] and acc[j] >= acc[k] and (cost[j] < cost[k] or acc[j] > acc[k])
                                       for j in cost)]
