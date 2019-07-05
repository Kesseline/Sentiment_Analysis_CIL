import math
import pickle

rcnn_prob = "submission_files/rcnn_test.pkl"
fasttext_prob = "submission_files/fast_text_test.pkl"
simple_lstm = "submission_files/simple_lstm_test.pkl"
simple_conv = "submission_files/simple_conv_test.pkl"
dcnn_prob = "submission_files/dcnn_test.pkl"

# models can be excluded by commenting them out
models_to_compare = {
    rcnn_prob
#    ,fasttext_prob
#    ,simple_lstm
#    ,sgd_prob
#    ,simple_conv
    ,dcnn_prob
}

def read_submission_probs(filepath):
	with open(filepath,"rb") as file:
		data = pickle.load(file)
	return [float(d[0]) for d in data]

def combine_models():
    probabilities = []
    # averaging probabilities for the models
    for model in models_to_compare:
        model_probabilities = read_submission_probs(model)
        if len(probabilities) <= 0:
            probabilities = model_probabilities
        else:
            probabilities = [t[0] + t[1] for t in zip(probabilities,model_probabilities)]


    probabilities = [x / len(models_to_compare) for x in probabilities]

    print("creating submission file")
    with open("submission.csv", "w") as out:
        out.write("Id,Prediction\n")
        for nr,entry in enumerate(probabilities):
            if entry <= 0.5:
                p = -1
            else:
                p = 1
            out.write(str(nr+1)+","+str(p)+"\n")
    
combine_models()
