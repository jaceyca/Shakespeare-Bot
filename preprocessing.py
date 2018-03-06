file = open("./data/shakespeare.txt")

# read in the file 
for index, line in enumerate(file):
	# super jank way to get rid of line numbers, but it works!
	if line != "\n" and len(line) != 23 and len(line) != 22 and len(line) != 21:
		

