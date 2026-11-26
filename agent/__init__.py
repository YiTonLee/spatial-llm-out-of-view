"""The evidence-on-demand agent (branch `evidence-pipeline`).

Plain: the parts that decide what evidence to hand the small model and in what form, and that count what it costs.
Technical: packets.py renders a tool's result as a packet in one of the formats compared in E1; cost.py times each
stage of answering one question and finds the accuracy-cost frontier (E0); data/ downloads the public datasets.
The experiment scripts that use these live in src/ (e0_ledger.py, e1_d1.py) and run through run.py like the rest.
"""
