import sys, pprint, math
from sets import Set 


def read_training_file(filename):
	animals_count = 0
	others_count = 0

	animal_class_set = None
	other_class_set = None
	animal_class_doc = None
	other_class_doc = None

	# Open file and read it
	with open(filename) as f:
		
		for line in f:

			(text, category) = line.split('::')

			# Filter out noise
			text = text.translate(None,'!,.?;:(){}')
			text = text.strip()

			words = text.split(' ')

			# Strip whitespaces
			category = category.strip()

			for index, word in enumerate(words):
				words[index] = words[index].strip()


			if category == 'ANIMAL':
				# Remove empty string
				animal_wordset = Set(words).difference([''])

				# Generate class doc
				if animal_class_set == None:
					animal_class_set = animal_wordset
					animal_class_doc = [x for x in words if x != '']
				else:
					animal_class_set = animal_class_set.union(animal_wordset)
					animal_class_doc = animal_class_doc + [x for x in words if x != '']

				animals_count = animals_count + 1
			elif category == 'OTHER':
				# Remove empty string
				other_wordset = Set(words).difference([''])

				# Generate class doc
				if other_class_set == None:
					other_class_set = other_wordset
					other_class_doc = [x for x in words if x != '']
				else:
					other_class_set = other_class_set.union(other_wordset)
					other_class_doc = other_class_doc + [x for x in words if x != '']

				others_count = others_count + 1

	
	# print "ANIMAL DOC:\n%s\n\nOTHER DOC:\n%s\n\n" % (','.join(animal_class_set), ','.join(other_class_set))

	
	# print "Animals: %d Others: %d" % (animals_count, others_count)

	# Calculate priors
	animals_prior = animals_count/float(animals_count+others_count)
	others_prior = others_count/float(animals_count+others_count)

	priors = {'ANIMAL':animals_prior, 'OTHER': others_prior}

	# Calculate conditional probabilities and conditional probabilities
	universal_set = animal_class_set.union(other_class_set)

	conditional_probs = {}

	for word in universal_set:
		# probability in class animal with laplace smoothing
		# = (occurences of word in docs of class animal + 1)/(total words in docs of class animal + |unique words in all documents|)
		probably_in_animal = (animal_class_doc.count(word) + 1)/float((len(animal_class_doc)+len(universal_set)))

		# probability in class other with laplace smoothing
		# = (occurences of word in docs of class other + 1)/(total words in docs of class other + |unique words in all documents|) 
		probably_in_other = (other_class_doc.count(word) + 1)/float((len(other_class_doc)+len(universal_set)))

		conditional_probs[word] = {'ANIMAL':probably_in_animal, 'OTHER':probably_in_other}

	# Return a tuple of priors and conditional probabilities
	return (priors,conditional_probs,len(animal_class_doc),len(other_class_doc),len(universal_set))



def classify(filename, priors, conditional_probs,animal_doc_count,other_doc_count,vocabulary_count):

	# Results of the classification
	results = {}

	# Open file and read it
	with open(filename) as f:
		
		for line_num, line in enumerate(f,1):

			# Remove irrelevant punctuation; remove whitespace at the ends of the line
			line = line.translate(None,'!,.?;:(){}').strip()

			# Get each word
			words = line.split(' ')

			# Strip whitespace at the ends of each word
			for index, word in enumerate(words):
				words[index] = words[index].strip()

			# Remove elements with an empty string
			words = [x for x in words if x != '']

			# Get probability that line is in class animal
			# = log(P(animal)) + {log(P(W1)) + ... + log(P(Wn))}
			summation_of_animal_probs=0
			for word in words:

				if word in conditional_probs:
					summation_of_animal_probs = summation_of_animal_probs + math.log(conditional_probs[word]['ANIMAL'])
				else:
					# Dealing with unknown words
					summation_of_animal_probs = summation_of_animal_probs + math.log(1/float(animal_doc_count+(vocabulary_count+1)))

			probability_of_animal = math.log(priors['ANIMAL']) + summation_of_animal_probs


			# Get probability that line is in class other
			# = log(P(other)) + {log(P(W1)) + ... + log(P(Wn))}
			summation_of_other_probs=0
			for word in words:

				if word in conditional_probs:
					summation_of_other_probs = summation_of_other_probs + math.log(conditional_probs[word]['OTHER'])
				else:
					# Dealing with unknown words
					summation_of_other_probs = summation_of_other_probs + math.log(1/float(other_doc_count+(vocabulary_count+1)))

			probability_of_other = math.log(priors['OTHER']) + summation_of_other_probs


			# Get the best class

			if probability_of_animal > probability_of_other:
				results[line_num] = 'ANIMAL'
			elif probability_of_animal < probability_of_other:
				results[line_num] = 'OTHER'
			else:
				results[line_num] = 'UNKNOWN'

	# Return results
	return results
















def main():

	# Get training file 
	training_file = sys.argv[1]
	testing_file = sys.argv[2]

	results = classify(testing_file,*read_training_file(training_file))

	pp = pprint.PrettyPrinter(indent=4)

	print "Results:\n"
	pp.pprint(results)
	print "\n"







if __name__ == '__main__':
	main()
