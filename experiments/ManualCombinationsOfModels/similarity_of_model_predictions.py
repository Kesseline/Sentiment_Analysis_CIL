
rcnn = "submission_files/rcnn_submission.csv"
dcnn = "submission_files/dcnn_submission.csv"
fasttext = "submission_files/fasttext_submission.csv"
sgd = "submission_files/sgd_submission.csv"
xgb = "submission_files/xgb_submission.csv"

models_to_compare = {
	rcnn
	,dcnn
	,fasttext
	,sgd
	,xgb
}

def read_submissions(filepath):
	file = open(filepath, "r")
	lines = file.readlines()[1:]
	pred = [l.split(",")[1] for l in lines]
	file.close()
	return pred

def print_differences()
	count_differences = 0
	for model1 in models_to_compare:
		for model2 in models_to_compare:
			if model1 == model2:
				continue
			
			model1_pred = read_submissions(model1)
			model2_pred = read_submissions(model2)
			
			print("comparing ",model1," with ",model2)
			diff = len(list(filter(lambda x: x[0] != x[1], zip(model1_pred, model2_pred))))
			
			print("number of different entries: ", str(diff))

print_differences()
