import sys
import argparse

sys.path.insert(0,"utils")
sys.path.insert(0,"models")
import model as m

import ngrams_sgd
import simple_lstm
import simple_conv
import glove_svm
import xgboost_ensemble
import fast_text
import rcnn
import dcnn
import transfer_sgd

def print_bold(text):
    print ('\033[1m' + text + '\033[0m')

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='XGBoost ensemble')
    parser.add_argument('--trainNegPath', type=str, default=m.def_trainneg, help='path to negative dataset')
    parser.add_argument('--trainPosPath', type=str, default=m.def_trainpos, help='path to positive dataset')
    parser.add_argument('--testPath', type=str, default=m.def_test, help='path to test dataset')
    parser.add_argument('--submPath', type=str, default=m.def_subm, help='submission path')
    parser.add_argument('--probsPath', type=str, default=m.def_probs, help='Confidence score path')
    
    parser.add_argument('--skipBuild', type=bool, nargs='?', const=True, default=False, help='whether to skip pre-building models')
    parser.add_argument('--skipProbs', type=bool, nargs='?', const=True, default=False, help='whether to skip generating confidence scores')
    parser.add_argument('--validateAll', type=bool, nargs='?', const=True, default=False, help='whether to validate all models, ignores ensemble')
    parser.add_argument('--submitAll', type=bool, nargs='?', const=True, default=False, help='whether to submit all models, including ensemble')
    
    args = parser.parse_args()
    
    # Active models are the ones responsible for our final submission
    models = [
        simple_lstm.simple_lstm(),
        simple_conv.simple_conv(),
	fast_text.fast_text(),
        rcnn.rcnn(),
        # dcnn.dcnn(),
        # ngrams_sgd.ngrams_sgd(),
        # glove_svm.glove_svm()
        # transfer_sgd.transfer_sgd()
    ]
    
    # Assign paths (could also use constructor but this is simpler)
    for model in models:
        model.subm = args.submPath
        model.probs = args.probsPath
        model.trainneg = args.trainNegPath
        model.trainpos = args.trainPosPath
        model.test = args.testPath
    
    if args.validateAll:
    
        # Build models before training
        if not args.skipBuild:
            for model in models:
                print_bold("Building %s..." % model.name)
                model.build()
                
        # Validate models
        for model in models:
            print_bold("Validating %s..." % model.name)
            model.validate()
            
    else: # Someone validating probably doesn't care about the ensemble (right?)
    
        # Create ensemble from model names
        names = [model.name for model in models]
        ensemble = xgboost_ensemble.xgboost_ensemble(names)

        # Build models before training
        if not args.skipBuild:
            for model in models:
                print_bold("Building %s..." % model.name)
                model.build()
            ensemble.build()
    
    
        # Generate confidence scores
        if not args.skipProbs:
            for model in models:
                print_bold("Computing confidence scores for %s..." % model.name)
                model.generate_probs()
            
        # Generate submissions
        if args.submitAll:
            for model in models:
                print_bold("Predict %s..." % model.name)
                model.generate_predict()
        
        # Train and predict ensemble
        # (We always validate since it's fast, since it validates using training set scores will always be high though)
        print_bold("Validating ensemble...")
        ensemble.validate()
        
        print_bold("Predicting ensemble...")
        ensemble.generate_predict()
        
   
