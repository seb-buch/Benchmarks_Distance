###Create own time function using decorator


import random
import time

###Decorator : Takes in a function object with all its parameters#####
def timerfunc(func):
    """
    A timer decorator
    """
    def function_timer(*args, **kwargs):
        """
        A nested function for timing other functions
        """
        start = time.time()
        value = func(*args, **kwargs)
        end = time.time()
        runtime = end - start
        msg = "The runtime for {func} took {time} seconds to complete"
        print(msg.format(func=func.__name__,
                         time=runtime))
        return value
    return function_timer
    ##Calling to timerfunc returns the function object of function_timer
 
 
@timerfunc
def long_runner():
    for x in range(5):
        sleep_time = random.choice(range(1,5))
        time.sleep(sleep_time)
##long_runner = timerfunc(long_runner)
##long runner is an object with its own functionality
##which chooses any random number from 1 to 5 
##when called from the decorator
 
## If running directly from this script
if __name__ == '__main__':
    long_runner()