import os, sys

def if_not_exit(path):
	if not os.path.exists(path):
		print(path, 'is not exist.')
		exit()
		
def if_not_make(path):
	if not os.path.exists(path):
		os.makedirs(path)
