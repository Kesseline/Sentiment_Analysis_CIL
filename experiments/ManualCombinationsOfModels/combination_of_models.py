import math

rcnn_prob = "submission_files/rcnn_submission_prob.csv"
fasttext_prob = "submission_files/fasttext_submission_prob.csv"
sgd_prob = "submission_files/sgd_submission_prob.csv"
xgb_prob = "submission_files/xgb_submission_prob.csv"

# models can be excluded by commenting them out
models_to_compare = {
    rcnn_prob
#    ,fasttext_prob
#    ,sgd_prob
    ,xgb_prob
}

def read_submission_probs(filepath):
    file = open(filepath, "r")
    lines = file.readlines()[1:]
    # This returns a list of the probabilities of the tweets being positive tweets
    prob = [float(l.split(",")[1]) for l in lines]
    file.close()
    return prob

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
