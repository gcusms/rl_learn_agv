"""Train once, then check the identical physical development tasks in pairs."""
import argparse
from datetime import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback


def stamp(): return datetime.now().astimezone().isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('baseline','checkpoint','training-cases','evaluation-cases','model-output','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--steps',type=int,default=4000);parser.add_argument('--seed',type=int,default=221015)
    parser.add_argument('--cpu-cores',type=float,default=.35)
    from neural_navigation.decision_feedback import LEGACY_SCHEMA,WAIT_SCHEMA
    parser.add_argument('--feedback-schema',choices=(LEGACY_SCHEMA,WAIT_SCHEMA),default=LEGACY_SCHEMA)
    parser.add_argument('--record-rollouts',action='store_true')
    parser.add_argument('--exploration',choices=('none','sensor_temperature_v1'),default='none')
    args=parser.parse_args()
    for name in ('baseline','checkpoint','training_cases','evaluation_cases','model_output','output'):
        setattr(args,name,getattr(args,name).resolve())
    args.output.mkdir(parents=True,exist_ok=False)
    status=dict(status='preparing',started=stamp(),training_target=args.steps,
        deployment_approved=False,auto_replace=False,feedback_schema=args.feedback_schema,
        exploration_schema=args.exploration)
    def save(**values):
        status.update(values,updated=stamp())
        temporary=args.output/'status.tmp';temporary.write_text(json.dumps(status,indent=2)+'\n')
        temporary.replace(args.output/'status.json')
    def run(command,log):
        save(command=command)
        with (args.output/log).open('w') as stream:
            subprocess.run(command,check=True,stdout=stream,stderr=subprocess.STDOUT)
    try:
        for name,source in [('training_cases.json',args.training_cases),('evaluation_cases.json',args.evaluation_cases)]:
            (args.output/name).write_bytes(source.read_bytes())
        save(status='training')
        training_command=[sys.executable,'-m','experiments.train_stop_feedback','--bundle',str(args.baseline),
            '--checkpoint',str(args.checkpoint),'--training-cases',str(args.training_cases),
            '--output',str(args.model_output),'--steps',str(args.steps),'--seed',str(args.seed),
            '--cpu-cores',str(args.cpu_cores),'--feedback-schema',args.feedback_schema,
            '--exploration',args.exploration]
        if args.record_rollouts:training_command.append('--record-rollouts')
        run(training_command,'training.log')
        training=json.loads((args.model_output/'status.json').read_text())
        if training['status']!='complete' or training['additional_ppo_steps']!=args.steps:
            raise RuntimeError('Incomplete declared training budget')
        save(status='paired_evaluation',training=training)
        run([sys.executable,'-m','experiments.evaluate_docking_navigation',
            '--candidate',str(args.model_output/'portable'),'--baseline',str(args.baseline),
            '--cases',str(args.evaluation_cases),'--output',str(args.output/'paired'),
            '--cpu-cores',str(args.cpu_cores),'--candidate-name','stop_feedback_ppo',
            '--baseline-name','prior_constraints'],'evaluation.log')
        report=json.loads((args.output/'paired/RESULTS.json').read_text())
        cases=json.loads(args.evaluation_cases.read_text())
        if not report['completed'] or len(report['rows'])!=2*len(cases):
            raise RuntimeError('Incomplete paired navigation evaluation')
        before=[r for r in report['rows'] if r['variant']=='prior_constraints']
        after=[r for r in report['rows'] if r['variant']=='stop_feedback_ppo']
        from experiments.paired_navigation_gate import compare
        gate=compare(before,after)
        def stats(rows):
            return dict(tasks=len(rows),safe_success=sum(r['safe_success'] for r in rows),
                collisions=sum(r['collisions'] for r in rows),timeouts=sum(r['timeout'] for r in rows),
                false_arrivals=sum(r['false_arrival'] for r in rows),
                voluntary_wait_frames=sum(r['voluntary_wait_frames'] for r in rows),
                only_stop_frames=sum(r['only_stop_frames'] for r in rows))
        summary=dict(completed=True,deployment_approved=False,before=stats(before),after=stats(after),
            training=training,paired_gate=gate,
            waiting_note='A legal movement is not necessarily useful or future-safe; wait counts alone do not prove improvement.',
            scope='Previously used development tasks, no independent real-world acceptance.')
        (args.output/'SUMMARY.json').write_text(json.dumps(summary,indent=2)+'\n')
        save(status='complete',finished=stamp(),summary=summary)
    except BaseException as error:
        save(status='failed',error=repr(error),finished=stamp())
        (args.output/'ERROR.txt').write_text(traceback.format_exc())
        raise


if __name__=='__main__':main()
