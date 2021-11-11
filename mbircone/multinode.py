import os
import numpy as np
from dask.distributed import Client, as_completed
import time
import pprint as pp
import socket


def get_cluster_ticket(job_queue_system_type,
                       num_nodes=1,
                       num_worker_per_node=1, num_threads_per_worker=1,
                       maximum_memory_per_node=None, maximum_allowable_walltime=None,
                       infiniband_flag="", par_env="", queue_sys_opt=[],
                       death_timeout=60, local_directory='./', log_directory='./'):
    """A utils function to help users easily use Dask-Jobqueue to deploy Dask on job-queuing systems like Slurm and SGE
    to use multiple nodes(computers) or directly use Dask on your local computer for parallel computation.

    - Dask is a flexible library for parallel computing in Python. More information is in `Dask documentation <https://docs.dask.org/en/latest/>`_.
    - Dask-Jobqueue is a Dask module to deploy Dask on some common job-queuing systems. More information is in `Dask-Jobqueue documentation <https://jobqueue.dask.org/en/latest/#>`_.

    Args:
        job_queue_system_type (string): Type of job-queuing systems(batch-queuing systems).
            Any high-performance computing(HPC) cluster needs a job-queuing system to share computational resources between users.
            Users need to specify which job-queuing system is used in their HPC cluster, before deploying their programs on the HPC cluster.
            This function currently supports 2 job-queuing systems for multi-nodes and a simulated job-queuing system for your local machine.

                1. If job_queue_system_type == 'SGE', deploy Dask on multi-nodes using job-queuing system Sun Grid Engine.
                2. If job_queue_system_type == 'SLURM', deploy Dask on multi-nodes using job-queuing system Slurm Wordload Manager.
                3. If job_queue_system_type == 'LocalHost', deploy Dask on your local machine.

        num_nodes (int, optional): [Default=1] Desire number of nodes for parallel computation.

        num_worker_per_node (int, optional): [Default=1] Desire number of workers to be used in a node.
            One worker can handle one parallel job at the same time.
        num_threads_per_worker (int, optional):[Default=1] Desire number of threads to be used in a worker.
            num_threads_per_worker and num_worker_per_node will be used to compute number of cpus you will use in a node.
            num_threads_per_worker * num_worker_per_node should be less than the total number of cpus in a single node.

        maximum_memory_per_node (str, optional): [Default=None] Desire maximum memory to be used in a node.
            Job queue systems always impose a memory limit on each node.
            By default, it is deliberately relatively small — 100 MB per node.
            If your job uses more than that, you’ll get an error that your job Exceeded job memory limit.
            To set a larger limit, pass a string with the form as 100MB or 16GB.
            Also, you should check your cluster documentation to avoid exceeding the allowable memory of a node.

        maximum_allowable_walltime (str, optional): [Default=None] Maximum allowable walltime.
            Job queue systems always impose a walltime limit on each submitted job in a node.
            By default, it is deliberately relatively small — only serval hours.
            If your job needs more time to finish, you'll pass a string with the form as D-HH:MM:SS.
            Also, you should check your cluster documentation to avoid exceeding the allowable walltime of your submitted job.

        infiniband_flag (str, optional): [Default=""] An optional string flag follows the submission command to
        request worker nodes using InfiniBand as network interface to communicate with the scheduler.
            Nodes with InfiniBand are requried by dask_jobqueue class. By default, it will be set to "".
            If all available nodes have InfiniBand, then you can ignore this option.
            Otherwise, you should check how to request worker nodes with infiniBand in your cluster documentation
            to find the specific option and pass it to this function.

        par_env (str, optional): [Default="openmpi"] The openmp parallel environment in your job-queuing system.
            This is a specific option for SGE cluster. By default, it will be set to "openmpi".
            If you are using SGE cluster, you can check available parallel environment option in your cluster documentation
            or run 'qconf -spl' in your terminal to find those options.

        queue_sys_opt (list[str], optional): [Default=[]] List of other job-queuing system's options, each option should a string.
            Different job-queuing systems may have different features like job scheduling and queuing.
            Users can customize those features by adding related options to queue_sys_opt.
            The related options may vary in different HPC cluster, you should check those information in your cluster documentation
            or ask maintainers of your institution's cluster.
            Each option in queue_sys_opt will be prepended with the submission command.

            For example,

                - In SGE, an option will be prepended with the #$ prefix.
                - In SLURM, an option will be prepended with the #SBATCH prefix.

        death_timeout (float, optional): [Default=60] Seconds to wait for a scheduler before closing workers.
            By default, it will be set to 60 seconds.
        local_directory (str, optional): [Default='./'] Desire local directory for file spilling in parallel computation.
            Recommend to set it to a location of fast local storage like /scratch or $TMPDIR.
        log_directory (str, optional): [Default='./'] Desire directory to store Dask's job scheduler logs.
            For each reserved node, there will be two different log files, error log and output log.
            Users can check those log files to find the information printed from the parallel functions.

    Returns:
        A two-element tuple including:

        - **cluster**: A cluster ticket to the dask deployment on the job-queuing system. This is an important input to the scatter_gather function.
        - **maximum_possible_nb_worker** (int): Maximum possible number of workers that we can request to start the jobs deployment.
    """

    # This function currently only support 3 types of job-queuing system.
    if job_queue_system_type not in ['SGE', 'SLURM', 'LocalHost']:
        print('The parameter job_queue_system_type should be one of [\'SGE\', \'SLURM\', \'LocalHost\']')
        print('Run the code without dask parallel.')
        return None, 0

    # Handle None input for some arguments. None input may happen when read those parameters from a configuration file.
    if infiniband_flag is None:
        infiniband_flag = ""

    if par_env is None or par_env == "":
        par_env = "openmpi"

    if queue_sys_opt is None:
        queue_sys_opt = []

    # Deploy Dask on multi-nodes using job-queuing system Sun Grid Engine.
    if job_queue_system_type == 'SGE':
        from dask_jobqueue import SGECluster

        # Append infiniband_flag and openmpi paralell environment to queue_sys_opt.
        # All options in queue_sys_opt will be added behind the submission command.
        if infiniband_flag != "":
            queue_sys_opt.append(infiniband_flag)
        queue_sys_opt.append('-pe %s %d' % (par_env, num_threads_per_worker * num_worker_per_node))

        cluster = SGECluster(processes=num_worker_per_node,
                             n_workers=num_worker_per_node,
                             walltime=maximum_allowable_walltime,
                             memory=maximum_memory_per_node,
                             death_timeout=death_timeout,
                             cores=num_worker_per_node,
                             job_extra=queue_sys_opt,
                             local_directory=local_directory,
                             log_directory=log_directory)
        cluster.scale(jobs=num_nodes)
        maximum_possible_nb_worker = num_worker_per_node * num_nodes
        print(cluster.job_script())

    # Deploy Dask on multi-nodes using job-queuing system Slurm Wordload Manager.
    if job_queue_system_type == 'SLURM':
        from dask_jobqueue import SLURMCluster

        # Append infiniband_flag to queue_sys_opt.
        # All options in queue_sys_opt will add behind the submission command.
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
        maximum_possible_nb_worker = num_worker_per_node * num_nodes
        print(cluster.job_script())

    # Deploy Dask on your local machine.
    if job_queue_system_type == 'LocalHost':
        from dask.distributed import LocalCluster
        cluster = LocalCluster(n_workers=num_worker_per_node,
                               processes=True,
                               threads_per_worker=1)
        maximum_possible_nb_worker = num_worker_per_node

    return cluster, maximum_possible_nb_worker


def scatter_gather(func, variable_args_list=[], constant_args={}, cluster=None, min_nb_start_worker=1, verbose=1):
    """Distribute a function with various groups of inputs to dask .
     Return a list of value or tuple(when the number of outputs of the given function is more than 1) with respect to the variable_args_list.

    Args:
        func (callable): Any callable function.

        variable_args_list (list[dictionary]): [Default=[]] A list of dictionary.
            Each dictionary contains arguments that will change during the parallel computation process.

        constant_args(dictionary): [Default={}] A dictionary contains arguments that will be constant during the parallel computation process.

        cluster (Object): [Default=None] A cluster ticket to the dask deployment on the job-queuing system.
            Users can obtain this ticket by running :py:func:`~multinode.get_cluster_ticket`.
            If your HPC cluster does not use SGE and SLRUM job-queuing system,
            you can also use functions in `dask_jobqueue <https://jobqueue.dask.org/en/latest/api.html>`_
            to deploy dask on a specific job-queuing system.
            If users do not provide the cluster ticket, cluster will be set to None by default and call a for loop to finish those jobs.

        min_nb_start_worker (int): [Default=1] Desire minimum number of workers to start parallel computation.
            The parallelization will wait until the number of workers >= min_nb_worker.

        verbose (int): [Default=0] Possible values are {0,1}, where 0 is quiet and 1 prints parallel computation process information.

    Returns:
        A list of return outputs of the parallel function, with respect to the variable_args_list.
        Notice: For those functions with multiple outputs, each returned output will be a tuple, containing multiple outputs.

    Examples:
        Here is an example to illustrate how to use the function on your local machine.

        >>> import numpy as np
        >>> from mbircone.multinode import get_cluster_ticket, scatter_gather
        >>> from psutil import cpu_count

        >>> #Deploy Dask on your local multi-core machine for parallel computation.
        >>> num_cpus = cpu_count(logical=False)
        >>> cluster, max_possible_num_worker = get_cluster_ticket('LocalHost', num_worker_per_node=num_cpus)

        >>> #Define a simple linear function.
        >>> def linear_func(x_1, a, b):
        ...     return a*x_1+b

        >>> #Parallel compute y=2*x+3 with respect to six different x_1.
        >>> variable_args_list = [{'x_1':i} for i in range(6)]
        >>> constant_args = {'a':2, 'b':3}
        >>> scatter_gather(linear_func, variable_args_list, constant_args, cluster=cluster, min_nb_start_worker=max_possible_num_worker)
        [3, 5, 7, 9, 11, 13]

    """

    if not variable_args_list:
        print("Input an empty variable_args_list to scatter_gather. Return an empty list.")
        print("variable_args_list is a dictionary includes fixed arguments that should be inputed to the given "
              "function during the parallel computation process.")
        return []

    if cluster is None:
        return_list = []
        for variable_args in variable_args_list:
            x = func(**variable_args, **constant_args)
            return_list.append(x)
        return return_list

    def parallel_func(t, variable_args):
        """
        # Define function that returns dictionary containing output, index, host name, processor ID, and completion time.
        """
        output = func(**variable_args, **constant_args)
        return {'output': output,
                'index': t,
                'host': socket.gethostname(),
                'pid': os.getpid(),
                'time': time.strftime("%H:%M:%S")}

    # Pass the cluster ticket to a dask client. The dask client will use the dask deployment to do parallel computation.
    # reference: `https://distributed.dask.org/en/latest/api.html#distributed.Client`
    client = Client(cluster)
    submission_list = list(range(len(variable_args_list)))

    # Start submit jobs until the client gets enough workers.
    while True:
        nb_workers = len(client.scheduler_info()["workers"])
        if verbose:
            print('Got {} workers'.format(nb_workers))
        if nb_workers >= min_nb_start_worker:
            print('Got {} workers, start parallel computation.'.format(nb_workers))
            break
        time.sleep(1)
    if verbose:
        print('client:', client)
        pp.pprint(client.scheduler_info()["workers"])

    completed_list = []
    return_list = []

    # Job may get cancelled since the communication between scheduler and workers can fail.
    # Use a while loop to ensure dask finishes all submitted jobs and gathers them back.
    elapsed_time = -time.time()
    while submission_list:
        # For large input, we should distribute the dataset to cluster memory.
        # reference: `https://distributed.dask.org/en/latest/api.html#distributed.Client.scatter`
        future_inp = client.scatter([variable_args_list[tp] for tp in submission_list])

        # client.map() map the function "parallel_func" to a sequence of input arguments.
        # reference: `https://distributed.dask.org/en/latest/api.html#distributed.Client.map`
        # The class object as_completed() returns outputs in the order of completion.
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
    print("Parallel Elapsed time: %f s" % elapsed_time)
    # Reorder the gathered reconstruction, corresponding to input sinogram list.
    sort_index = np.argsort(np.array(completed_list))
    return_list = [return_list[i] for i in sort_index]
    client.close()

    return return_list
