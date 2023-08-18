import time


def print_status(_log, start_time, epoch, num_epochs, **kwargs):
    '''
    Pretty loging for the console. Can log an arbitrary dictionary with floats along with the progress statistics,
    which one would like anyways.
    '''
    # Epoch logging
    msg = f"Epoch: {epoch+1}/{num_epochs}"
    # Time logging
    diff_time = time.time() - start_time
    secs_left = diff_time/(epoch+1) * (num_epochs-epoch)
    str_left_time = time.strftime("%H:%M:%S", time.gmtime(secs_left))
    str_elapsed_time = time.strftime("%H:%M:%S", time.gmtime(diff_time))
    str_total_time = time.strftime("%H:%M:%S", time.gmtime(diff_time+secs_left))
    msg += f"\tTime elapsed: {str_elapsed_time}\tTime left: {str_left_time}"
    msg += f"\tTotal time estimate: {str_total_time}"
    # Arbitrary float dict logging
    for key in kwargs.keys():
        msg += f"\t{key}: {kwargs[key]:.3f}"
    _log.info(msg)

