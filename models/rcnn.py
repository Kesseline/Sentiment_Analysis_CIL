import sys

sys.path.insert(0,"../utils")
sys.path.insert(0,"src")

import model as m
import utils as u
import numpy as np
import pickle
import time
import datetime
import os

import pickle_vocab as pv
import cooc as coo
import glove_embeddings as ge

from sklearn import metrics
import tensorflow as tf
import TextRCNN as trcnn

###############################
#
# Reference: This code is partially based on the model from https://github.com/roomylee/rcnn-text-classification
#
###############################

class rcnn(m.model):
    # This model uses the recurrent convolutional structure

    # Use Glove embeddings by default
    def_vocab = "../data/embeddings/rcnn/vocab.pkl"
    def_cooc = "../data/embeddings/rcnn/cooc.pkl"
    def_embeddings = "../data/embeddings/rcnn/embeddings.npz"

    def __init__(self, vocab = def_vocab, cooc = def_cooc, embeddings = def_embeddings, subm = m.def_subm, probs = m.def_probs, trainneg = m.def_trainneg, trainpos = m.def_trainpos, test = m.def_test, word2vec = None, cp_dir = None):
        m.model.__init__(self, subm, probs, trainneg, trainpos, test)
        
        # Model Hyperparameters
        self.max_sentence_length = 50       # Max sentence length in train/test data (Default: 50)
        self.cell_type = "vanilla"          # Type of RNN cell. Choose 'vanilla' or 'lstm' or 'gru' (Default: vanilla)
        self.glove = embeddings             # Glove file with pre-trained embeddings
        self.glove_voc = vocab              # Lookup-file for glove-embeddings
        self.cooc = cooc                    # Cooc matrix file
        self.word2vec = word2vec            # Word2vec file with pre-trained embeddings
        self.word_embedding_dim = 300       # Dimensionality of word embedding (Default: 300)
        self.context_embedding_dim = 512    # Dimensionality of context embedding(= RNN state size)  (Default: 512)
        self.hidden_size = 512              # Size of hidden layer (Default: 512)
        self.dropout_keep_prob =  0.7       # Dropout keep probability (Default: 0.7)
        self.l2_reg_lambda =  0.5           # L2 regularization lambda (Default: 0.5)
        # Training parameters
        self.batch_size = 64                # Batch Size (Default: 64)
        self.num_epochs = 3                 # Number of training epochs (Default: 10)
        self.display_every = 100            # Number of iterations to display training info.
        self.checkpoint_dir = cp_dir        # Checkpoint directory from training run")
        self.checkpoint_every = 1000        # Save model after this many steps
        self.num_checkpoints = 1            # Number of checkpoints to store
        self.learning_rate = 1e-3           # Which learning rate to start with. (Default: 1e-3)

        # Misc Parameters
        self.allow_soft_placement = True    # Allow device soft device placement
        self.log_device_placement = False   # Log placement of ops on devices

        self.text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(self.max_sentence_length)

    def build(self):
        # Initialize vocabulary 
        pv.generate_vocab(self.trainpos, self.trainneg, self.glove_voc)
        coo.create_cooc(self.trainpos, self.trainneg, self.glove_voc, self.cooc)
        ge.create_embeddings(self.cooc, self.glove)
        
    def load_train(self):
        # Load and prepare training tweets
        print("Loading training data")
        with tf.device('/cpu:0'):
            x_text, y = u.load_data_and_labels_train(self.trainpos, self.trainneg)

        x = np.array(list(self.text_vocab_processor.fit_transform(x_text)))
        print("Text Vocabulary Size: {:d}".format(len(self.text_vocab_processor.vocabulary_)))

        print("x = {0}".format(x.shape))
        print("y = {0}".format(y.shape))
        print("")

        return x, y

    def load_test(self):
        # Load and prepare test tweets (a little hacky)
        print("Loading test data")
        with tf.device('/cpu:0'):
            x_text = u.load_data_and_labels_test(self.test)

        x_eval = np.array(list(self.text_vocab_processor.transform(x_text)))
        return x_eval

    def fit(self, data, labels):
        # Build and train model 
        seq_length = data.shape[1]

        with tf.Graph().as_default():
            session_conf = tf.ConfigProto(
                allow_soft_placement=self.allow_soft_placement,
                log_device_placement=self.log_device_placement)
            sess = tf.Session(config=session_conf)
            with sess.as_default():
                rcnn = trcnn.TextRCNN(
                    sequence_length=seq_length,
                    num_classes=2,
                    vocab_size=len(self.text_vocab_processor.vocabulary_),
                    word_embedding_size=self.word_embedding_dim,
                    context_embedding_size=self.context_embedding_dim,
                    cell_type=self.cell_type,
                    hidden_size=self.hidden_size,
                    l2_reg_lambda=self.l2_reg_lambda,
                )

                # Define Training procedure
                global_step = tf.Variable(0, name="global_step", trainable=False)
                train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(rcnn.loss, global_step=global_step)

                # Output directory for models and summaries
                timestamp = str(int(time.time()))
                out_dir = os.path.abspath(os.path.join(os.path.pardir, "data/rcnn_runs", timestamp))
                print("Writing to {}\n".format(out_dir))

                # Summaries for loss and accuracy
                loss_summary = tf.summary.scalar("loss", rcnn.loss)
                acc_summary = tf.summary.scalar("accuracy", rcnn.accuracy)

                # Train Summaries
                train_summary_op = tf.summary.merge([loss_summary, acc_summary])
                train_summary_dir = os.path.join(out_dir, "summaries", "train")
                train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

                # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
                checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
                checkpoint_prefix = os.path.join(checkpoint_dir, "model")
                if not os.path.exists(checkpoint_dir):
                    os.makedirs(checkpoint_dir)
                saver = tf.train.Saver(tf.global_variables(), max_to_keep=self.num_checkpoints)

                # Write vocabulary
                self.text_vocab_processor.save(os.path.join(out_dir, "text_vocab"))

                # Initialize all variables
                sess.run(tf.global_variables_initializer())

                # Pre-trained word2vec
                if self.word2vec:
                    # initial matrix with random uniform
                    initW = np.random.uniform(-0.25, 0.25, (len(self.text_vocab_processor.vocabulary_), self.word_embedding_dim))
                    # load any vectors from the word2vec
                    print("Load word2vec file {0}".format(self.word2vec))
                    with open(self.word2vec, "rb") as f:
                        header = f.readline()
                        vocab_size, layer1_size = map(int, header.split())
                        binary_len = np.dtype('float32').itemsize * layer1_size
                        print(vocab_size)
                        print(layer1_size)
                        for line in range(vocab_size):
                            word = []
                            while True:
                                ch = f.read(1).decode('latin-1')
                                if ch == ' ':
                                    word = ''.join(word)
                                    break
                                if ch != '\n':
                                    word.append(ch)
                            idx = self.text_vocab_processor.vocabulary_.get(word)
                            if idx != 0:
                                initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                            else:
                                f.read(binary_len)
                    sess.run(rcnn.W_text.assign(initW))
                    print("Success to load pre-trained word2vec model!\n")

                # Glove Embeddings
                elif self.glove and self.glove_voc:
                    # initial matrix with random uniform
                    initW = np.random.uniform(-0.25, 0.25, (len(self.text_vocab_processor.vocabulary_), self.word_embedding_dim))
                    # load any glove vectors
                    gl_emb = np.load(self.glove)
                    gl_embx = gl_emb["arr_0"]
                    print("Load Glove Embeddings {0}".format(self.glove))
                    with open(self.glove_voc, "rb") as f:
                        vocab = pickle.load(f)
                        vocab_size, layer1_size = len(vocab), self.word_embedding_dim
                        for key, value in vocab.items():
                            idx = self.text_vocab_processor.vocabulary_.get(key)
                            if idx != 0 and value < len(gl_embx):
                                initW[idx] = gl_embx[value]
                    sess.run(rcnn.W_text.assign(initW))
                    print("Success to load Glove model!\n")

                # Generate batches
                batches = u.batch_iter(
                    list(zip(data, labels)), self.batch_size, self.num_epochs)
                # Training loop. For each batch...
                print("Training model")
                for batch in batches:
                    x_batch, y_batch = zip(*batch)
                    # Train
                    feed_dict = {
                        rcnn.input_text: x_batch,
                        rcnn.input_y: y_batch,
                        rcnn.dropout_keep_prob: self.dropout_keep_prob
                    }

                    _, step, summaries, loss, accuracy = sess.run(
                        [train_op, global_step, train_summary_op, rcnn.loss, rcnn.accuracy], feed_dict)
                    train_summary_writer.add_summary(summaries, step)

                    # Training log display
                    if step % self.display_every == 0:
                        time_str = datetime.datetime.now().isoformat()
                        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

                    # Model checkpoint
                    if step % self.checkpoint_every == 0:
                        path = saver.save(sess, checkpoint_prefix, global_step=step)
                        print("Saved model checkpoint to {}\n".format(path))

                self.session = sess
                self.model = rcnn


    def validate(self):
        # Validate model
        train_x, y = self.load_train()
        train_x, test_x, train_y, test_y = u.split_data(train_x, y)

        self.fit(train_x, train_y)

        print("Validating")
        feed_dict_dev = {
            self.model.input_text: test_x,
            self.model.input_y: test_y,
            self.model.dropout_keep_prob: 1.0
        }
        loss, accuracy = self.session.run(
            [self.model.loss, self.model.accuracy], feed_dict_dev)

        time_str = datetime.datetime.now().isoformat()
        print("{}: loss {:g}, acc {:g}\n".format(time_str, loss, accuracy))

    def predict(self):
        # Train model and create a submission

        # If checkpoint available do not train anew
        if self.checkpoint_dir:
            text_path = os.path.join(os.path.curdir, self.checkpoint_dir, "..", "text_vocab")
            self.text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

            checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)

            graph = tf.Graph()
            with graph.as_default():
                session_conf = tf.ConfigProto(
                    allow_soft_placement=self.allow_soft_placement,
                    log_device_placement=self.log_device_placement)
                sess = tf.Session(config=session_conf)
                with sess.as_default():
                    # Load the saved meta graph and restore variables
                    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                    saver.restore(sess, checkpoint_file)
                self.session = sess
                # Get params
                predictions = graph.get_operation_by_name("output/predictions").outputs[0]
                input_text = graph.get_operation_by_name("input_text").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        else:
            train_x, y = self.load_train()
            self.fit(train_x, y)
            # Get params
            predictions = self.model.predictions
            input_text = self.model.input_text
            dropout_keep_prob = self.model.dropout_keep_prob

        test_x = self.load_test()
        batches = u.batch_iter(list(test_x), self.batch_size, 1, shuffle=False)

        print("Creating submission")
        all_predictions = []
        # Collect the predictions here
        for x_batch in batches:
            batch_predictions = self.session.run(predictions, {input_text: x_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

        # Map to correct labels
        preds = np.asarray([-1 if pred == 0 else pred for pred in all_predictions])

        u.write_submission(preds.astype(int), self.subm + self.name + "_submission.csv")
        print("Submission successfully created")

    def compute_probs(self):
        # Train model and compute confidence score on train-(negative tweets first) and test-data

        # If checkpoint available do not train anew
        if self.checkpoint_dir:
            text_path = os.path.join(os.path.curdir, self.checkpoint_dir, "..", "text_vocab")
            self.text_vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor.restore(text_path)

            checkpoint_file = tf.train.latest_checkpoint(self.checkpoint_dir)

            graph = tf.Graph()
            with graph.as_default():
                session_conf = tf.ConfigProto(
                    allow_soft_placement=self.allow_soft_placement,
                    log_device_placement=self.log_device_placement)
                sess = tf.Session(config=session_conf)
                with sess.as_default():
                    # Load the saved meta graph and restore variables
                    saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                    saver.restore(sess, checkpoint_file)
                self.session = sess
                # Get params
                logits = graph.get_operation_by_name("output/logits").outputs[0]
                input_text = graph.get_operation_by_name("input_text").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        else:
            train_x, y = self.load_train()
            self.fit(train_x, y)
            # Get params
            logits = self.model.logits
            input_text = self.model.input_text
            dropout_keep_prob = self.model.dropout_keep_prob

        print("Probs train set")
        train_x, _ = self.load_train()
        batches = u.batch_iter(list(train_x), self.batch_size, 1, shuffle=False)

        all_probs = []
        # Collect the probabilities here
        for x_batch in batches:
            batch_probs = self.session.run(logits, {input_text: x_batch, dropout_keep_prob: 1.0})
            if(len(all_probs)==0):
                all_probs = np.exp(batch_probs)/np.sum(np.exp(batch_probs),axis=1)[:,None]
            else:
                all_probs = np.concatenate([all_probs,np.exp(batch_probs)/np.sum(np.exp(batch_probs),axis=1)[:,None]]) 

        with open(self.probs + self.name + "_train.pkl","wb") as f:
            pickle.dump(all_probs,f)

        print("Probs test set")
        test_x = self.load_test()
        batches = u.batch_iter(list(test_x), self.batch_size, 1, shuffle=False)

        all_probs = []
        # Collect the probabilities here
        for x_batch in batches:
            batch_probs = self.session.run(logits, {input_text: x_batch, dropout_keep_prob: 1.0})
            if(len(all_probs)==0):
                all_probs = np.exp(batch_probs)/np.sum(np.exp(batch_probs),axis=1)[:,None]
            else:
                all_probs = np.concatenate([all_probs,np.exp(batch_probs)/np.sum(np.exp(batch_probs),axis=1)[:,None]]) 

        with open(self.probs + self.name + "_test.pkl","wb") as f:
            pickle.dump(all_probs,f)

        print("All probabilities successfully computed")
