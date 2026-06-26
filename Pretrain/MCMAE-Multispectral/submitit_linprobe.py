# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# A script to run multinode training with submitit.
# --------------------------------------------------------

import argparse
import os
import uuid
from pathlib import Path

import main_linprobe as classification
import submitit


def parse_args():
    classification_parser = classification.get_args_parser()
    parser = argparse.ArgumentParser("Submitit for MAE linear probe", parents=[classification_parser])
    parser.add_argument("--ngpus", default=8, type=int, help="Number of gpus to request on each node")
    parser.add_argument("--nodes", default=2, type=int, help="Number of nodes to request")
    parser.add_argument("--timeout", default=4320, type=int, help="Duration of the job")
    parser.add_argument("--job_dir", default="", type=str, help="Submitit log dir. Defaults to <output_dir>/submitit/%j.")
    parser.add_argument("--partition", default="", type=str, help="Optional Slurm partition. Omit to use the cluster default.")
    parser.add_argument("--constraint", default="", type=str, help="Optional Slurm constraint, for example a GPU type.")
    parser.add_argument('--comment', default="", type=str, help="Comment to pass to scheduler")
    return parser.parse_args()


def get_init_file(job_dir):
    # Init file must not exist, but it's parent dir must exist.
    job_dir = Path(job_dir)
    init_dir = job_dir.parent if "%j" in str(job_dir) else job_dir
    os.makedirs(str(init_dir), exist_ok=True)
    init_file = init_dir / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main_linprobe as classification

        self._setup_gpu_args()
        classification.main(self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(self.args.job_dir).as_uri()
        checkpoint_file = os.path.join(self.args.output_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit
        from pathlib import Path

        job_env = submitit.JobEnvironment()
        self.args.output_dir = Path(str(self.args.output_dir).replace("%j", str(job_env.job_id)))
        self.args.log_dir = self.args.output_dir
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def main():
    args = parse_args()
    if args.job_dir == "":
        args.job_dir = Path(args.output_dir or ".") / "submitit" / "%j"

    # Note that the folder will depend on the job_id, to easily track experiments
    executor = submitit.AutoExecutor(folder=args.job_dir, slurm_max_num_timeout=30)

    num_gpus_per_node = args.ngpus
    nodes = args.nodes
    timeout_min = args.timeout

    kwargs = {}
    if args.partition:
        kwargs['slurm_partition'] = args.partition
    if args.constraint:
        kwargs['slurm_constraint'] = args.constraint
    if args.comment:
        kwargs['slurm_comment'] = args.comment

    executor.update_parameters(
        mem_gb=40 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node, # one task per GPU
        cpus_per_task=10,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_signal_delay_s=120,
        **kwargs
    )

    executor.update_parameters(name="mae")

    args.dist_url = get_init_file(args.job_dir).as_uri()
    args.output_dir = args.job_dir

    trainer = Trainer(args)
    job = executor.submit(trainer)

    # print("Submitted job_id:", job.job_id)
    print(job.job_id)


if __name__ == "__main__":
    main()
