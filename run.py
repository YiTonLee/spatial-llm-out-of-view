"""Main entry point: run the whole project, or one stage of it, and compare the numbers with ours.

  python run.py list                      # the stages and their steps, with rough times and what is already done
  python run.py all                       # everything, in order (about 8-12 GPU hours + 1 CPU hour; resumable)
  python run.py data cache                # only these stages
  python run.py ladder --step m_r_ca_dist # only steps whose name contains this text
  python run.py all --quick --work workspace_quick   # smoke test: short training/eval, meaningless numbers
  python run.py report                    # your numbers next to ours (docs/04_RESULTS.md explains them)
  python run.py e0 e1_d1 e1_d2 e6a        # evidence-pipeline experiments (docs/07_EVIDENCE_PIPELINE.md)

Options
  --work PATH   workspace folder (default ./workspace, or $SPATIAL_LLM_WORK)
  --quick       shorten every training/evaluation (checks that the code runs; needs the full data and cache)
  --force       re-run steps even if their outputs exist
  --dry         print what would run, run nothing

Stages run in this order: data, cache, model_w, estimator, record_lora, rules, ladder, versions, look_choice, report.
`all` runs these ten. The evidence-pipeline stages (e0, e1_d1, e1_d2, e6a) run only when named; e0 and e1_d1 need model_w and
estimator, e1_d2 and e6a need the public data (python -m agent.data.download --only e1_d2 / --only e6a).
Public datasets for later experiments: python -m agent.data.download (docs/07_EVIDENCE_PIPELINE.md).
Each step is skipped when its output files already exist, so an interrupted run continues where it stopped.
"""
import argparse, sys, time

from pipeline import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('stages', nargs='+', help="'list', 'all', or stage names")
    ap.add_argument('--work', default=None)
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--dry', action='store_true')
    ap.add_argument('--step', default=None, help='only steps whose name contains this text')
    a = ap.parse_args()
    if a.work:
        config.set_work(a.work)
    from pipeline.stages import STAGES, CORE
    from pipeline.runner import run_step, done

    if a.stages == ['list']:
        print(f'workspace: {config.WORK}\n')
        for i, (name, make) in enumerate(STAGES.items(), 1):
            print(f'{i:2d} {name}')
            for s in make(a.quick):
                flag = 'done' if done(s) else ('GPU ' if s.gpu else '    ')
                print(f'     [{flag:4s}] {s.name:22s} {s.minutes:>16s}  {s.what}')
        return
    names = CORE if a.stages == ['all'] else a.stages
    unknown = [n for n in names if n not in STAGES]
    if unknown:
        sys.exit(f'unknown stage(s) {unknown}; choose from {list(STAGES)}')
    print(f'workspace: {config.WORK}' + ('   (QUICK mode)' if a.quick else ''))
    t0 = time.time()
    for name in names:
        for s in STAGES[name](a.quick):
            if a.step and a.step not in s.name:
                continue
            if not run_step(s, name, force=a.force, dry=a.dry):
                sys.exit(f'stopped at {name}/{s.name}. Fix the problem and re-run the same command: finished steps are skipped.')
    print(f'finished in {(time.time() - t0) / 60:.1f} min')


if __name__ == '__main__':
    main()
