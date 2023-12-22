"""
This is the script that contains the backend code. No need to look at this to implement new functionality
Functions that run separate processes. These processes run on GPUs, and are queried by processes running only CPUs
"""

import dill
import inspect
import queue
import torch
import torch.multiprocessing as mp
from rich.console import Console
from time import time
from typing import Callable, Union

from configs import config

console = Console(highlight=False)

if mp.current_process().name == 'MainProcess':
    # No need to initialize the models inside each process
    import vision_models
    # Create a list of all the defined models
    list_models = [m[1] for m in inspect.getmembers(vision_models, inspect.isclass)
                   if issubclass(m[1], vision_models.BaseModel) and m[1] != vision_models.BaseModel]
    # Sort by attribute "load_order"
    list_models.sort(key=lambda x: x.load_order)
    if config.multiprocessing:
        manager = mp.Manager()
    else:
        manager = None
else:
    list_models = None
    manager = None


def make_fn(model_class, process_name, counter):
    """
    model_class.name and process_name will be the same unless the same model is used in multiple processes, for
    different tasks
    """
    # We initialize each one on a separate GPU, to make sure there are no out of memory errors
    num_gpus = torch.cuda.device_count()
    gpu_number = counter % num_gpus

    model_instance = model_class(gpu_number=gpu_number)

    def _function(*args, **kwargs):
        if process_name != model_class.name:
            kwargs['process_name'] = process_name

        if model_class.to_batch and not config.multiprocessing:
            # Batchify the input. Model expects a batch. And later un-batchify the output.
            args = [[arg] for arg in args]
            kwargs = {k: [v] for k, v in kwargs.items()}

            # The defaults that are not in args or kwargs, also need to listify
            full_arg_spec = inspect.getfullargspec(model_instance.forward)
            if full_arg_spec.defaults is None:
                default_dict = {}
            else:
                default_dict = dict(zip(full_arg_spec.args[-len(full_arg_spec.defaults):], full_arg_spec.defaults))
            non_given_args = full_arg_spec.args[1:][len(args):]
            non_given_args = set(non_given_args) - set(kwargs.keys())
            for arg_name in non_given_args:
                kwargs[arg_name] = [default_dict[arg_name]]

        try:
            out = model_instance.forward(*args, **kwargs)
            if model_class.to_batch and not config.multiprocessing:
                out = out[0]
        except Exception as e:
            print(f'Error in {process_name} model:', e)
            out = None
        return out

    return _function


if config.multiprocessing:

    def make_fn_process(model_class, process_name, counter):

        if model_class.to_batch:
            seconds_collect_data = model_class.seconds_collect_data  # Window of seconds to group inputs
            max_batch_size = model_class.max_batch_size

            def _function(queue_in):

                fn = make_fn(model_class, process_name, counter)

                to_end = False
                while True:
                    start_time = time()
                    time_left = seconds_collect_data
                    batch_inputs = []
                    batch_queues = []
                    while time_left > 0 and len(batch_inputs) < max_batch_size:
                        try:
                            received = queue_in.get(timeout=time_left)
                            if received is None:
                                to_end = True
                                break
                            else:
                                batch_inputs.append(received[0])
                                batch_queues.append(received[1])
                        except queue.Empty:  # Time-out expired
                            break  # Break inner loop (or do nothing, would break anyway because time_left < 0)
                        time_left = seconds_collect_data - (time() - start_time)
                    if len(batch_inputs) > 0:
                        batch_kwargs = collate(batch_inputs, model_class.forward)
                        outs = fn(**batch_kwargs)
                        try:
                            for out, qu in zip(outs, batch_queues):
                                qu.put(out)
                        except Exception as e:
                            # No message, because we are just carrying the error from before
                            for qu in batch_queues:
                                qu.put(None)
                    if to_end:
                        print(f'{process_name} model exiting')
                        break

        else:
            def _function(queue_in):
                fn = make_fn(model_class, process_name, counter)
                while True:
                    received = queue_in.get()
                    if received is None:
                        print(f'{process_name} exiting')
                        return
                    (args, kwargs), queue_out = received
                    out = fn(*args, **kwargs)
                    queue_out.put(out)

        return _function


    if mp.current_process().name == 'MainProcess':
        queues_in: Union[dict[str, mp.Queue], None] = dict()
        consumers: dict[str, Union[mp.Process, Callable]] = dict()

        counter_ = 0
        for model_class_ in list_models:
            for process_name_ in model_class_.list_processes():
                if process_name_ in config.load_models and config.load_models[process_name_]:
                    queue_in_ = manager.Queue()  # For transfer of data from producer to consumer
                    queues_in[process_name_] = queue_in_

                    fn_process = make_fn_process(model_class_, process_name_, counter_)
                    # Otherwise, it is not possible to pickle the _function (not defined at top level)
                    aux = mp.reducer.dump
                    mp.reducer.dump = dill.dump
                    consumer = mp.Process(target=fn_process, kwargs={'queue_in': queue_in_})
                    consumer.start()
                    mp.reducer.dump = aux
                    consumers[process_name_] = consumer

                    counter_ += 1

    else:
        queues_in = None


    def finish_all_consumers():
        # Wait for consumers to finish
        for q_in in queues_in.values():
            q_in.put(None)
        for cons in consumers.values():
            cons.join()

else:

    consumers = dict()

    counter_ = 0
    for model_class_ in list_models:
        for process_name_ in model_class_.list_processes():
            if process_name_ in config.load_models and config.load_models[process_name_]:
                consumers[process_name_] = make_fn(model_class_, process_name_, counter_)
                counter_ += 1

    queues_in = None

    def finish_all_consumers():
        pass


def forward(model_name, *args, queues=None, **kwargs):
    """
    Sends data to consumer (calls their "forward" method), and returns the result
    """
    error_msg = f'No model named {model_name}. ' \
                'The available models are: {}. Make sure to activate it in the configs files'
    if not config.multiprocessing:
        try:
            out = consumers[model_name](*args, **kwargs)
        except KeyError as e:
            raise KeyError(error_msg.format(list(consumers.keys()))) from e
    else:
        if queues is None:
            consumer_queues_in, queue_results = None, None
        else:
            consumer_queues_in, queue_results = queues
        try:
            if consumer_queues_in is not None:
                consumer_queue_in = consumer_queues_in[model_name]
            else:
                consumer_queue_in = queues_in[model_name]
        except KeyError as e:
            options = list(consumer_queues_in.keys()) if consumer_queues_in is not None else list(queues_in.keys())
            raise KeyError(error_msg.format(options)) from e
        if queue_results is None:
            # print('No queue exists to get results. Creating a new one, but this is inefficient. '
            #       'Consider providing an existing queue for the process')
            queue_results = manager.Queue()  # To get outputs
        consumer_queue_in.put([(args, kwargs), queue_results])
        out = queue_results.get()  # Wait for result
    return out


def collate(batch_inputs, fn):
    """
    Combine a list of inputs into a single dictionary. The dictionary contains all the parameters of the
    function to be called. If the parameter is not defined in some samples, the default value is used. The
    value of the parameters is always a list.
    """
    # Separate into args and kwargs
    args_input, kwarg_input = list(zip(*batch_inputs))
    full_arg_spec = inspect.getfullargspec(fn)
    if full_arg_spec.defaults is None:
        default_dict = {}
    else:
        default_dict = dict(zip(full_arg_spec.args[-len(full_arg_spec.defaults):], full_arg_spec.defaults))
        if 'process_name' in default_dict:  # process_name is a special parameter filled in later
            del default_dict['process_name']

    args_list = full_arg_spec.args[1:]  # Remove self

    # process_name is a special parameter filled in later
    if 'process_name' in args_list:
        assert args_list[-1] == 'process_name', 'process_name must be the last argument'
        args_list.remove('process_name')

    kwargs_output = {k: [] for k in args_list}
    for i, (args, kwargs) in enumerate(zip(args_input, kwarg_input)):
        if len(args) + len(kwargs) > len(args_list):
            raise Exception(
                f'You provided more arguments than the function {fn.__name__} accepts, or some kwargs/args '
                f'overlap. The arguments are: {args_list}')
        for j, arg_name in enumerate(args_list):
            if len(args) > j:
                kwargs_output[arg_name].append(args[j])
            elif arg_name in kwargs:
                kwargs_output[arg_name].append(kwargs[arg_name])
            else:
                assert arg_name in default_dict, f'You did not provide a value for the argument {arg_name}.'
                kwargs_output[arg_name].append(default_dict[arg_name])

    return kwargs_output
