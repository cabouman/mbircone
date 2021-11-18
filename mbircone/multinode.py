import os
import numpy as np
from dask.distributed import Client, as_completed
import time
import pprint as pp
import socket
import getpass


def get_cluster_ticket(job_queue_system_type,
                       num_physical_cores_per_node,
                       num_nodes=1,
                       maximum_memory_per_node=None, maximum_allowable_walltime=None,
                       system_specific_args="",
                       local_directory='./', log_directory='./'):
    """A utility to return a ticket needed for :py:func:`~multinode.scatter_gather` to access a parallel cluster.
    The defaults are set to use one python thread per node under the assumption that the python thread calls C code
    that creates a number of threads equal to the number of physical cores in the node.

    - On SLURM, you can use sinfo to get information about your cluster configuration.
    - On SGE, you can use qhost to get information about your cluster configuration.

    Args:
        job_queue_system_type (string): One of 'SGE' (Sun Grid Engine), 'SLURM', 'LocalHost'

        num_physical_cores_per_node (int): Number of physical cores per node = (number of cpus) x (cores per cpu).

        num_nodes (int): [Default=1] Requested number of nodes for parallel computation.

        maximum_memory_per_node (str, optional): [Default=None] Requested maximum memory per node, e.g. '100MB' or '16GB'.
            If None, the scheduler will allocate a system-determined amount per node.

        maximum_allowable_walltime (str, optional): [Default=None] Maximum allowable walltime as a string in the
            form D-HH:MM:SS.  E.g., '0-01:00:00' for one hour. If None, the scheduler will allocate a
            system-determined maximum.

        system_specific_args (str, optional): [Default=None] Any additional arguments to pass to the job scheduling system.
            Consult your local documentation or system administrator.

        local_directory (str, optional): [Default='./'] Desired local directory for file spilling in parallel computation.
            Recommend to set it to a location of fast local storage like /scratch or $TMPDIR.

        log_directory (str, optional): [Default='./'] Desired directory to store Dask's job scheduler logs.
            For each reserved node, there will be two different log files, error log and output log.
            Users can check those log files to find the information printed from the parallel functions.

    Returns:
        A two-element tuple including:

        - **cluster_ticket**: A cluster ticket to access the job-queue system via the scatter_gather function.
        - **maximum_possible_nb_worker** (int): Maximum possible number of workers that we can request to start the jobs deployment.
    """

    # This function currently only supports 3 types of job-queuing system.
    if job_queue_system_type not in ['SGE', 'SLURM', 'LocalHost']:
        print('The parameter job_queue_system_type should be one of [\'SGE\', \'SLURM\', \'LocalHost\']')
        print('Run the code without dask parallel.')
        return None, 0

    # None type handling
    if not isinstance(system_specific_args, str):
        system_specific_args = ''

    if not isinstance(local_directory, str):
        local_directory = './'

    if not isinstance(log_directory, str):
        log_directory = './'

    local_directory = local_directory.replace('$USER', getpass.getuser())
    log_directory = log_directory.replace('$USER', getpass.getuser())

    # Deploy Dask on multi-nodes using job-queuing system Sun Grid Engine.
    if job_queue_system_type == 'SGE':
        from dask_jobqueue import SGECluster

        cluster = SGECluster(processes=num_worker_per_node,
                             n_workers=num_worker_per_node,
                             walltime=maximum_allowable_walltime,
                             memory=maximum_memory_per_node,
                             cores=num_worker_per_node,
                             # SGECluster does not support job_cpu, however, you can still request number of cores/node
                             # by passing argument to the job scheduling system with system_specific_args.
                             job_extra=[system_specific_args],
                             local_directory=local_directory,
                             log_directory=log_directory)
        cluster.scale(jobs=num_nodes)
        maximum_possible_nb_worker = num_worker_per_node * num_nodes
        print(cluster.job_script())

    # Deploy Dask on multi-nodes using job-queuing system Slurm Wordload Manager.
    if job_queue_system_type == 'SLURM':
        from dask_jobqueue import SLURMCluster

        cluster = SLURMCluster(processes=num_worker_per_node,
                               n_workers=num_worker_per_node,
                               walltime=maximum_allowable_walltime,
                               memory=maximum_memory_per_node,
                               # env_extra=['module load anaconda', 'source activate mbircone'],
                               cores=num_worker_per_node,
                               job_extra=[system_specific_args],
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

    make cluster_ticket a dictionary including the num_nodes and the original ticket

    return cluster_ticket, maximum_possible_nb_worker


def scatter_gather(cluster_ticket, func, constant_args={}, variable_args_list=[], min_nodes=None, verbose=1):
    """
    Distribute a function call across multiple nodes, as specified by the `cluster` argument.  The given function,
    func, is called with a set of keyword arguments, some that are the same for all calls, as specified in
    constant_args, and some that vary with each call, as specified in variable_args_list.

    Returns a list obtained by collecting the output from each call of func.  The length of the output list is the
    length of variable_args_list.

    Args:
        cluster_ticket (Object): A ticket used to access a specific cluster, that can be obtained from
            :py:func:`~multinode.get_cluster_ticket`.  If cluster_ticket=None, the process will run in serial.
            See `dask_jobqueue <https://jobqueue.dask.org/en/latest/api.html>`_ for more information.

        func (callable): A callable function with keyword arguments matching the entries in constant_args and
            variable_args_list.

        constant_args (dictionary): [Default={}] A dictionary of keyword arguments that are the same for all calls of func.

        variable_args_list (list[dictionary]): [Default=[]] A list of dictionaries of keyword arguments.  Each
            dictionary contains arguments for one call of func.

        min_nodes (int): [Default=None] Requested minimum number of workers to start parallel computation.
            The job will not start until the number of nodes >= min_nodes, and once it starts, no further nodes will
            be used.  The default is num_nodes from the cluster_ticket.

        verbose (int): [Default=0] Possible values are {0,1}, where 0 is quiet and 1 prints parallel computation
            process information.

    Returns:
        A list obtained by collecting the output from each call of func.  The length of the output list is the
        length of variable_args_list.  Each entry in the list will be the output of one call of func.
"""
    Check the example below to match the new interface.
    """
    Examples:
        In the example below, we define a function linear_func with arguments x_1, a, b.  We call this function
        6 times, with a=2 and b=3 for each call and x_1 set to each of 0, 1, 2, 3, 4, 5 in separate calls to linear_func.

        >>> import numpy as np
        >>> from mbircone.multinode import get_cluster_ticket, scatter_gather
        >>> from psutil import cpu_count

        >>> # Set up the local multi-core machine for parallel computation.
        >>> num_cpus = cpu_count(logical=False)
        >>> cluster_ticket, max_possible_num_worker = get_cluster_ticket('LocalHost',num_worker_per_node=num_cpus)

        >>> # Define a simple linear function.
        >>> def linear_func(x_1, a, b):
        ...     return a*x_1+b

        >>> # Parallel compute y=2*x+3 with respect to six different x_1.
        >>> variable_args_list = [{'x_1':i} for i in range(6)]
        >>> constant_args = {'a':2, 'b':3}
        >>> scatter_gather(linear_func,variable_args_list,constant_args,min_nodes=max_possible_num_worker)
        [3, 5, 7, 9, 11, 13]

    """

    if not variable_args_list:
        print("Input an empty variable_args_list to scatter_gather. Return an empty list.")
        print("variable_args_list is a dictionary includes fixed arguments that should be inputed to the given "
              "function during the parallel computation process.")
        return []

    if cluster_ticket is None:
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
    client = Client(cluster_ticket)
    submission_list = list(range(len(variable_args_list)))

    determine min_nodes from argument list or cluster_ticket

    # Start submit jobs until the client gets enough workers.
    while True:
        nb_workers = len(client.scheduler_info()["workers"])
        if verbose:
            print('Got {} workers'.format(nb_workers))
        if nb_workers >= min_nodes:
            print('Got {} workers, start parallel computation.'.format(nb_workers))
            break  does the scheduler keep getting nodes after this that go unused?
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
