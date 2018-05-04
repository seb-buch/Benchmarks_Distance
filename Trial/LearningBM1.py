##Using Timeit module
###Can use directly from the commandline
###python -m timeit -s "ord(x) for x in 'abcdefg']"

#### -m ----- module; -s ------run setup once
####
#### python -m timeit -s "import LearningBM;LearningBM.trial_func()"

def trial_func():
	try:
		1/0
	except ZeroDivisionError:
		pass 

###To run from the script itself
if __name__ == "__main__": 
###Sets the __name variable to __main__ module.
###Executes only If running from the same module
####Used in cases where we dont want to run it everytime
	import timeit
	setup = "from __main__ import trial_func"
### define the string to define the setup and brings the function to the namespace
### We dont want to count the time for importing, therefore use setup
	print timeit.timeit("trial_func()",setup=setup)
###Note that call to the function is in quotes
