import os
import numpy as np
from dask.distributed import Client, as_completed
import time
import pprint as pp
import socket

def get_cluster_ticket(job_queue_sys,
                       num_nodes=1,
                       num_worker_per_node=1, num_threads_per_worker=1,
                       maximum_memory_per_node=None, maximum_allowable_walltime=None,
                       infiniband_flag="", par_env="", queue_sys_opt=[],
                       death_timeout=60, local_directory='./', log_directory='./'):
    """Return a dask cluster class object from the given parameters.

    Args:
        job_queue_sys (string): Job queue system, used for submitting job.
            job_queue_sys should be one of ['SGE', 'SLURM', 'LocalHost'].
            'SGE': Return a class object, generated from dask_joequeue.SGECluster.
            'SLURM': Return a class object, generated from dask_joequeue.SLURMCluster.
            'LocalHost': Return a class object, generated from dask.distributed.LocalCluster.

        num_nodes (int, optional):[Default=1] Number of nodes requested in job system.

        num_worker_per_node (int, optional):[Default=1] Number of workers per node.
            One worker will handle one parallel job at the same time.
        num_threads_per_worker (int, optional):[Default=1] Number of threads needed for a worker.

        maximum_memory_per_node (str, optional):[Default=None] Maximum memory requested in a node.
            Job queue systems always impose a memory limit on each node. By default, it is deliberately relatively small — 100 MB per node.
            If your job uses more than that, you’ll get an error that your job Exceeded job memory limit.
            To set a larger limit, pass a string with the form as 100MB or 16GB.
            Also, you should check your cluster documentation to avoid exist the allowable memory.

        maximum_allowable_walltime (str, optional):[Default=None] Maxixmum allowable walltime.
            Job queue systems always impose a walltime limit on each submitted job in a node. By default, it is deliberately relatively small — only serval hours.
            If your job needs more time to finish, you'll pass a string with the form as D-HH:MM:SS.
            Also, you should check your cluster documentation to avoid exist the allowable walltime.

        infiniband_flag (str, optional): [Default=""] An optional string flag follows the submission command to request nodes with InfiniBand.
            Nodes with InfiniBand are requried by dask_jobqueue class.
            If all available nodes have InfiniBand, then you can ignore this flag.
            Otherwise, you should check your cluster documentation to find the specific flag and pass it to this function.

        par_env (str, optional): [Default=""] The openmp parallel environment in your job queue system.
            This is a specific option for SGE cluster.
            If you are using SGE cluster, you should check cluster documentation or run 'qconf -spl' to find those options.

        queue_sys_opt (list[str], optional)[Default=[]]: List of other job submitting options.

        death_timeout (float, optional):[Default=60] Seconds to wait for a scheduler before closing workers.
        local_directory (str, optional):[Default='./'] Dask worker local directory for file spilling.
            Recommend to set it to a location of fast local storage like /scratch or $TMPDIR
        log_directory (str, optional):[Default='./'] Directory to use for job scheduler logs.

    Returns:
        A two-element tuple including:

        - **cluster**: A dask cluster object from the given configuration parameters.
        - **min_nb_worker** (int): Minumum number of workers that a dask client needs to start jobs deployment.
    """

    if job_queue_sys not in ['SGE', 'SLURM', 'LocalHost']:
        print('The parameter job_queue_sys should be one of [\'SGE\', \'SLURM\', \'LocalHost\']')
        print('Run the code without dask parallel.')
        return None, 0

    if infiniband_flag == None:
        infiniband_flag = ""

    if par_env == None:
        par_env = "openmpi"

    if queue_sys_opt == None:
        queue_sys_opt = []

    if job_queue_sys == 'SGE':
        from dask_jobqueue import SGECluster

        # Append infiniband_flag and openmpi paralell environment to queue_sys_opt.
        # All flags in queue_sys_opt will add behind the submission command.

        queue_sys_opt.append(infiniband_flag)
        queue_sys_opt.append('-pe %s %d' % (par_env, num_threads_per_worker * num_worker_per_node))

        cluster = SGECluster(processes=num_worker_per_node,
                             n_workers=num_worker_per_node,
                             walltime=maximum_allowable_walltime,
                             memory=maximum_memory_per_node,
                             death_timeout=death_timeout,
                             cores=num_worker_per_node,
                             job_extra=queue_sys_opt,
                             # env_extra=['module load anaconda', 'source /home/%s/Documents/envs/mbircone/bin/activate'],
                             local_directory=local_directory,
                             log_directory=log_directory)
        cluster.scale(jobs=num_nodes)
        min_nb_worker = num_worker_per_node * num_nodes
        print(cluster.job_script())

    if job_queue_sys == 'SLURM':
        from dask_jobqueue import SLURMCluster

        # Append infiniband_flag and openmpi paralell environment to queue_sys_opt.
        # All flags in queue_sys_opt will add behind the submission command.
        queue_sys_opt.append(infiniband_flag)

        cluster = SLURMCluster(processes=num_worker_per_node,
                               n_workers=num_worker_per_node,
                               walltime=maximum_allowable_walltime,
                               memory=maximum_memory_per_node,
                               death_timeout=death_timeout,
                               # env_extra=['module load anaconda', 'source activate mbircone'],
                               cores=num_worker_per_node,
                               job_extra=queue_sys_opt,
                               job_cpu=num_threads_per_worker * num_worker_per_node,
                               local_directory=local_directory,
                               log_directory=log_directory)
        cluster.scale(jobs=num_nodes)
        min_nb_worker = num_worker_per_node * num_nodes
        print(cluster.job_script())

    if job_queue_sys == 'LocalHost':
        from dask.distributed import LocalCluster
        cluster = LocalCluster(n_workers=num_worker_per_node,
                               processes=True,
                               threads_per_worker=1)
        min_nb_worker = num_worker_per_node

    return cluster, min_nb_worker


def scatter_gather(func, variable_args_list=[], fixed_args={}, cluster=None, min_nb_worker=1, verbose=1):
    """Distribute a function with various groups of inputs to multiple processors. Return a list of value or tuple according to given variable_args_list.

    Args:
        func (callable): Any function we want to parallel compute.

        variable_args_list (list[dictionary]): [Default=[]] Each dictionary contains arguments changing during the parallel computation process.
        fixed_args(dictionary):  [Default={}}] A dictionary includes fixed arguments that should be inputed to the given function during the parallel computation process.
        cluster (Object): [Default=None] Cluster object created by dask_jobqueue.
            More information is on `dask_jobqueue <https://jobqueue.dask.org/en/latest/api.html>`_
        min_nb_worker (int): [Default=1] Minimum number of workers. The parallelization will wait until the number of workers >= min_nb_worker.
        verbose (int): [Default=0] Possible values are {0,1}, where 0 is quiet, 1 prints dask parallel progress information

    Returns:
         A list of return output of the parallel function, corresponding to the inps_list.
         Notice: For those functions with multiple output, each returned output will be a tuple, containing multiple output.

    """

    if variable_args_list == []:
        print("Input an empty variable_args_list to scatter_gather. Return an empty list.")
        print("variable_args_list is a dictionary includes fixed arguments that should be inputed to the given function during the parallel computation process.")
        return []

    if cluster == None:
        return_list=[]
        for variable_args in variable_args_list:
            x = func(**variable_args, **fixed_args)
            return_list.append(x)
        return return_list

    def parallel_func(t, variable_args):
        """
        # Define function that returns dictionary containing reconstruction, index, host name, PID, and time.
        """
        x = func(**variable_args, **fixed_args)
        return {'output': x,
                'index': t,
                'host': socket.gethostname(),
                'pid': os.getpid(),
                'time': time.strftime("%H:%M:%S")}


    # reference: `https://distributed.dask.org/en/latest/api.html#distributed.Client`
    client = Client(cluster)  # An interface that connect to and submit computation to cluster
    submission_list = list(range(len(variable_args_list)))

    # Start submit jobs until the client has enough workers.
    while True:
        nb_workers = len(client.scheduler_info()["workers"])
        print('Got {} workers'.format(nb_workers))
        if nb_workers >= min_nb_worker:
            break
        time.sleep(1)

    print('client:', client)
    pp.pprint(client.scheduler_info()["workers"])

    completed_list = []
    return_list = []

    # Job may get cancelled since communication to workers can fail.
    # Use a while loop to ensure dask finishs and gathers all submitted jobs.
    elapsed_time = -time.time()
    while submission_list:
        # For large input, we should distribute the dataset to cluster memeory.
        # reference: `https://distributed.dask.org/en/latest/api.html#distributed.Client.scatter`
        future_inp = client.scatter([variable_args_list[tp] for tp in submission_list])

        # client.map() map the function "par_recon" to a sequence of input arguments.
        # reference: `https://distributed.dask.org/en/latest/api.html#distributed.Client.map`
        # The class object as_completed() returns outputs in the order in which they complete
        # reference: `https://distributed.dask.org/en/latest/api.html#distributed.as_completed`
        for future in as_completed(
                client.map(parallel_func, submission_list, future_inp, pure=False)):
            if verbose:
                print(future)

            if not future.cancelled():
                try:
                    result = future.result()
                    return_list.append(result['output'])
                    completed_list.append(result['index'])
                    if verbose:
                        print('{')
                        print('index:', result['index'])
                        print('host:', result['host'])
                        print('pid:', result['pid'])
                        print('time:', result['time'])
                        print('}')
                except:
                    if verbose:
                        print("Cannot gather result from future:", future)
        # Once jobs completed, remove jobs from submission_list.
        submission_list = list(set(submission_list) - set(completed_list))
        if verbose:
            print(submission_list)
    elapsed_time += time.time()
    if verbose:
        print("Parallel Elapsed time: %f s" % elapsed_time)
    # Reorder the gathered reconstruction, corresponding to input sinogram list.
    sort_index = np.argsort(np.array(completed_list))
    return_list = [return_list[i] for i in sort_index]
    client.close()

    return return_list
