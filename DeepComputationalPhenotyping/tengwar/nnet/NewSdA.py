import os
import sys
import time

import numpy as np
import cPickle

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from tengwar.theano_tutorial.logistic_sgd import LogisticRegression, load_data
from tengwar.theano_tutorial.mlp import HiddenLayer
from dA import dA

from sklearn.metrics import roc_auc_score

class MultitaskLogisticRegression(object):
    """Multitask Logistic Regression Class
    """

    def __init__(self, input, n_in, n_out, Py_emp=0.5):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        :type Py_emp: float or np.array(float)
        :param Py_emp: class weights
        """
        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(
            value=np.zeros(
                (n_in, n_out),
                dtype=theano.config.floatX
            ),
            name='W',
            borrow=True
        )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX
            ),
            name='b',
            borrow=True
        )
 
        self.n_in = n_in
        self.n_out = n_out

        assert(type(Py_emp) is float or (type(Py_emp) is np.ndarray and Py_emp.shape[0] == n_out))
        self.weight_y1 = (1-Py_emp)
        self.weight_y0 = Py_emp
        #self.weight_y1 = 1./(Py_emp)
        #self.weight_y0 = 1./(1.-Py_emp)

        # symbolic expression for computing the matrix of class-membership
        # probabilities
        # Where:
        # W is a matrix where column-k represent the separation hyper plain for
        # class-k
        # x is a matrix where row-j  represents input training sample-j
        # b is a vector where element-k represent the free parameter of hyper
        # plain-k
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)

        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y, reg = None):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::
            \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
            \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|}
                \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
            \ell (\theta=\{W,b\}, \mathcal{D})

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        return -self.n_out * T.mean(y*T.log(self.p_y_given_x)*self.weight_y1 + (1-y)*T.log(1-self.p_y_given_x)*self.weight_y0)

    def errors(self, y, p_y_threshold=0.5):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        self.y_pred = self.p_y_given_x >= p_y_threshold

        # check if y has same dimension of y_pred
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, y.shape, 'y_pred', self.y_pred.type, self.y_pred.shape)
            )
        # check if y is of the correct datatype
        if y.dtype.startswith('int'):
            # the T.neq operator returns a vector of 0s and 1s, where 1
            # represents a mistake in prediction
            return T.mean(T.neq(self.y_pred, y))
        else:
            raise NotImplementedError()

    def errors_per_output(self, y, p_y_threshold=0.5):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch per task; zero one
        loss over the size of the minibatch

        :type y: theano.tensor.TensorType
        :param y: corresponds to a vector that gives for each example the
                  correct label
        """
        self.y_pred = self.p_y_given_x >= p_y_threshold
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y', y.type, y.shape, 'y_pred', self.y_pred.type, self.y_pred.shape)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred, y), axis=0)
        else:
            raise NotImplementedError()

class NewSdA(object):
    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        n_ins=784,
        hidden_layers_sizes=[1000, 1000],
        n_outs=10,
        corruption_levels=[0.1, 0.1],
        Py_emp = 0.5,
        log_layer_type=MultitaskLogisticRegression,
        S_matrix = None,
        S_type = None,
        lambda_S = 0.0001,
        lambda_O_l2 = 0.0001,
        lambda_O_l1 = 0.0001, 
        lambda_H_l2 = 0.0001
    ):
        """ This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the sdA

        :type n_layers_sizes: list of ints
        :param n_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network

        :type corruption_levels: list of float
        :param corruption_levels: amount of corruption to use for each
                                  layer
        """

        self.sigmoid_layers = []
        self.dA_layers = []
        self.params = []
        self.n_layers = len(hidden_layers_sizes)
        self.corruption_levels = corruption_levels
        self.L2_sqr = 0
        self.use_auc = False
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))
        # allocate symbolic variables for the data
        self.x = T.matrix('x')  # the data is presented as rasterized images
        self.y = T.imatrix('y')  # the labels are presented as 1D vector of
                                 # [int] labels

        for i in xrange(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden units of
            # the layer below or the input size if we are on the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the hidden
            # layer below or the input of the SdA if you are on the first
            # layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output

            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)
            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            self.params.extend(sigmoid_layer.params)
            self.L2_sqr = (
                self.L2_sqr + 
                (sigmoid_layer.W ** 2).sum() + 
                (sigmoid_layer.b ** 2).sum()
            )
            # Construct a denoising autoencoder that shared weights with this
            # layer
            dA_layer = dA(numpy_rng=numpy_rng,
                          theano_rng=theano_rng,
                          input=layer_input,
                          n_visible=input_size,
                          n_hidden=hidden_layers_sizes[i],
                          W=sigmoid_layer.W,
                          bhid=sigmoid_layer.b)
            self.dA_layers.append(dA_layer)

        if self.n_layers > 0:
            self.logLayer = log_layer_type(
                input=self.sigmoid_layers[-1].output,
                n_in=hidden_layers_sizes[-1],
                n_out=n_outs, Py_emp=Py_emp
            )
        else:
            self.logLayer = log_layer_type(
                input=self.x,
                n_in=n_ins,
                n_out=n_outs, Py_emp=Py_emp
            )

        self.params.extend(self.logLayer.params)

        lambda_O_l2 = lambda_O_l2 if (lambda_O_l2 is not None and lambda_O_l2 > 0) else 0
        lambda_O_l1 = lambda_O_l1 if (lambda_O_l1 is not None and lambda_O_l1 > 0) else 0
        lambda_H_l2 = lambda_H_l2 if (lambda_H_l2 is not None and lambda_H_l2 > 0) else 0

        # compute the cost for second phase of training,
        # defined as the negative log likelihood
        self.reg_cost = 0
        if S_matrix is not None:
            lambda_S = lambda_S if (lambda_S is not None and lambda_S > 0) else 0.0001
            if S_type == 'l1':
                sys.stdout.write('using L1 multi-task regularizer, lambda={0}\n'.format(lambda_S))
                L_loss = 0
                myW = self.logLayer.W
                for i in range(10):
                    for j in range(i+1,10):
                        L_loss = L_loss + T.sum(T.abs_(myW[:,i]-myW[:,j]))
            else:
                sys.stdout.write('using Laplacian multi-task regularizer, lambda={0}\n'.format(lambda_S))
                D_matrix = np.diag(S_matrix.sum(axis = 0))
                L_matrix = D_matrix - S_matrix
                L_loss = (T.dot(T.dot(self.logLayer.W, L_matrix), self.logLayer.W.T)).trace()
            self.reg_cost = self.reg_cost + lambda_S * L_loss

        if lambda_H_l2 > 0:
            sys.stdout.write('using L2 hidden unit penalty, lambda={0}\n'.format(lambda_H_l2))
            self.reg_cost = self.reg_cost + lambda_H_l2 * self.L2_sqr

        if lambda_O_l2 > 0:
            sys.stdout.write('using L2 penalty, lambda={0}\n'.format(lambda_O_l2))
            myW = self.logLayer.W
            self.reg_cost = self.reg_cost + lambda_O_l2 * ((myW**2).sum()) # + (self.b**2).sum())

        if lambda_O_l1 > 0:
            sys.stdout.write('using L1 penalty, lambda={0}\n'.format(lambda_O_l1))
            myW = self.logLayer.W
            self.reg_cost = self.reg_cost + lambda_O_l1 * ((abs(myW)).sum()) #+ (abs(self.b)).sum())
           
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y) + self.reg_cost

        self.errors = self.logLayer.errors(self.y)
        self.errors_per_output = self.logLayer.errors_per_output(self.y)

    def pretraining_functions(self, train_set_x, batch_size):
        ''' Generates a list of functions, each of them implementing one
        step in trainnig the dA corresponding to the layer with same index.
        The function will require as input the minibatch index, and to train
        a dA you just need to iterate, calling the corresponding function on
        all minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared variable that contains all datapoints used
                            for training the dA

        :type batch_size: int
        :param batch_size: size of a [mini]batch

        :type learning_rate: float
        :param learning_rate: learning rate used during training for any of
                              the dA layers
        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        corruption_level = T.scalar('corruption')  # % of corruption to use
        learning_rate = T.scalar('lr')  # learning rate to use
        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for dA in self.dA_layers:
            # get the cost and the updates list
            cost, updates = dA.get_cost_updates(corruption_level,
                                                learning_rate)
            # compile the theano function
            fn = theano.function(
                inputs=[
                    index,
                    theano.Param(corruption_level, default=0.2),
                    theano.Param(learning_rate, default=0.1)
                ],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin: batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns

    def build_finetune_functions(self, datasets, batch_size, learning_rate):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on
        a batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                         the has to contain three pairs, `train`,
                         `valid`, `test` in this order, where each pair
                         is formed of two Theano variables, one for the
                         datapoints, the other for the labels

        :type batch_size: int
        :param batch_size: size of a minibatch

        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage
        '''

        (train_set_x, train_set_y) = datasets[0]
        (valid_set_x, valid_set_y) = datasets[1]
        (test_set_x, test_set_y) = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_valid_batches /= batch_size
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]
        n_test_batches /= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        # compute the gradients with respect to the model parameters
        gparams = T.grad(self.finetune_cost, self.params)

        # compute list of fine-tuning updates
        updates = [
            (param, param - gparam * learning_rate)
            for param, gparam in zip(self.params, gparams)
        ]

        train_fn = theano.function(
            inputs=[index],
            outputs=self.finetune_cost,
            updates=updates,
            givens={
                self.x: train_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: train_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            },
            name='train'
        )

        if self.use_auc:
            test_score_i = theano.function(
                [index],
                outputs=self.logLayer.p_y_given_x,
                givens={
                    self.x: test_set_x[
                        index * batch_size: (index + 1) * batch_size
                    ]
                },
                name='test'
            )

            valid_score_i = theano.function(
                [index],
                outputs=self.logLayer.p_y_given_x,
                givens={
                    self.x: valid_set_x[
                        index * batch_size: (index + 1) * batch_size
                    ]
                },
                name='valid'
            )
        else:
            test_score_i = theano.function(
                [index],
                outputs=self.errors_per_output,
                givens={
                    self.x: test_set_x[
                        index * batch_size: (index + 1) * batch_size
                    ],
                    self.y: test_set_y[
                        index * batch_size: (index + 1) * batch_size
                    ]
                },
                name='test'
            )

            valid_score_i = theano.function(
                [index],
                outputs=self.errors_per_output,
                givens={
                    self.x: valid_set_x[
                        index * batch_size: (index + 1) * batch_size
                    ],
                    self.y: valid_set_y[
                        index * batch_size: (index + 1) * batch_size
                    ]
                },
                name='valid'
            )


        # Create a function that scans the entire validation set
        def valid_score():
            return np.vstack([valid_score_i(i) for i in xrange(n_valid_batches)])

        # Create a function that scans the entire test set
        def test_score():
            return np.vstack([test_score_i(i) for i in xrange(n_test_batches)])

        return train_fn, valid_score, test_score

    def do_unsupervised_pretraining(self, train_set_x, epochs=15, learn_rate=0.001, batch_size=1,
                                    save_fnbase='weights_pretrain'):
        #########################
        # PRETRAINING THE MODEL #
        #########################
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size

        print '... getting the pretraining functions'
        pretraining_fns = self.pretraining_functions(train_set_x=train_set_x,
                                                     batch_size=batch_size)

        print '... pre-training the model'
        start_time = time.clock()
        ## Pre-train layer-wise
        param_values = []
        print self.corruption_levels, self.n_layers
        assert(len(self.corruption_levels)==self.n_layers)
        assert(len(pretraining_fns)==self.n_layers)
        for i in xrange(self.n_layers):
            # go through pretraining epochs
            for epoch in xrange(epochs):
                # go through the training set
                c = []
                for batch_index in xrange(n_train_batches):
                    c.append(pretraining_fns[i](index=batch_index,
                             corruption=self.corruption_levels[i],
                             lr=learn_rate))
                print 'Pre-training layer %i, epoch %d, cost ' % (i, epoch),
                print np.mean(c)
            (W, b, W_prime, b_prime) = self.dA_layers[i].get_param_values()
            sys.stderr.write('Saving layer {0} weights to {1}_layer{0}.npz...'.format(i, save_fnbase))
            np.savez_compressed(save_fnbase + '_layer{0}.npz'.format(i), W_h=W, b_h=b, W_h_prime=W_prime, b_h_prime=b_prime)
            param_values.append((W,b,W_prime,b_prime))
            sys.stderr.write('DONE!\n')

        end_time = time.clock()

        print >> sys.stderr, ('The pretraining code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        sys.stderr.write('Saving all pretrain weights to ' + save_fnbase + '.npz...')
        np.savez_compressed(save_fnbase + '.npz', W_h=[ p[0] for p in param_values ],
                               b_h=[ p[1] for p in param_values ],
                               W_h_prime=[ p[2].T for p in param_values ],
                               b_h_prime=[ p[3] for p in param_values])
        sys.stderr.write('DONE!\n')

    def do_supervised_finetuning(self, train_set_x, train_set_y,
                                 valid_set_x, valid_set_y,
                                 test_set_x, test_set_y,
                                 epochs=1000, learn_rate=0.1, batch_size=1,
                                 use_auc = False,
                                 save_fnbase='weights_finetune'):
        ########################
        # FINETUNING THE MODEL #
        ########################

        self.use_auc = use_auc

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size

        # get the training, validation and testing function for the model
        print '... getting the finetuning functions'
        train_fn, validate_model, test_model = self.build_finetune_functions(
            datasets=((train_set_x,train_set_y), (valid_set_x,valid_set_y), (test_set_x,test_set_y)),
            batch_size=batch_size,
            learning_rate=learn_rate
        )

        print '... finetuning the model'
        # early-stopping parameters
        patience = 100 * n_train_batches  # look as this many examples regardless
        patience_increase = 2.  # wait this much longer when a new best is
                                # found
        improvement_threshold = 0.9999  # a relative improvement of this much is
                                       # considered significant
        validation_frequency = min(n_train_batches, patience / 2)
                                      # go through this many
                                      # minibatch before checking the network
                                      # on the validation set; in this case we
                                      # check every epoch

        if self.use_auc:
            best_validation_loss = -np.inf
        else:
            best_validation_loss = np.inf
        test_score = 0.
        start_time = time.clock()

        done_looping = False
        epoch = 0

        validation_losses = validate_model()
        if self.use_auc:
            Yv  = valid_set_y.eval()
            Yte = test_set_y.eval()
            this_validation_loss = roc_auc_score(Yv, validation_losses, average='macro')
            validation_loss_per_output = np.hstack([ roc_auc_score(Yv[:,task], validation_losses[:,task]) for task in range(Yv.shape[1]) ])
            loss_str = 'AUC'
        else:
            this_validation_loss = np.mean(validation_losses)
            validation_loss_per_output = np.mean(validation_losses, axis=0)
            loss_str = 'error'

        print('INITIAL validation %s %f (%s) %%' % (loss_str, this_validation_loss,
               ' '.join([ str(l) for l in (validation_loss_per_output * 100.) ])))

        while (epoch < epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):
                minibatch_avg_cost = train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    validation_losses = validate_model()
                    if self.use_auc:
                        this_validation_loss = roc_auc_score(Yv, validation_losses, average='macro')
                        validation_loss_per_output = np.hstack([ roc_auc_score(Yv[:,task], validation_losses[:,task]) for task in range(Yv.shape[1]) ])
                    else:
                        this_validation_loss = np.mean(validation_losses)
                        validation_loss_per_output = np.mean(validation_losses, axis=0)

                    #print('epoch %i, minibatch %i/%i, validation %s %f (%s) %%' %
                    #      (epoch, minibatch_index + 1, n_train_batches, loss_str, this_validation_loss,
                    #       ' '.join([ str(l) for l in (validation_loss_per_output * 100.) ])))
                    print('epoch %i, minibatch %i/%i, iter %i/%i, validation %s %f %%' %
                          (epoch, minibatch_index + 1, n_train_batches, iter, patience,
                           loss_str, this_validation_loss)) 

                    # if we got the best validation score until now
                    if (self.use_auc and this_validation_loss > best_validation_loss) or \
                       (not self.use_auc and this_validation_loss < best_validation_loss):

                        if self.use_auc:
                            print '\n%     improvement: {0}'.format((this_validation_loss - best_validation_loss) / this_validation_loss)
                        else:
                            print '\n%     improvement: {0}'.format((best_validation_loss - this_validation_loss) / best_validation_loss)

                        #improve patience if loss improvement is good enough
                        if (self.use_auc and best_validation_loss > this_validation_loss * improvement_threshold) or \
                           (not self.use_auc and this_validation_loss < best_validation_loss * improvement_threshold):
                            old_patience = patience
                            patience = max(patience, int(iter * patience_increase))
                            print('%     increasing patience: {0:d} -> {1:d}'.format(old_patience, patience))

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = test_model()
                        if self.use_auc:
                            test_score = roc_auc_score(Yte, test_losses, average='macro')
                            test_scores = np.hstack([ roc_auc_score(Yte[:,task], test_losses[:,task]) for task in range(Yte.shape[1]) ])
                        else:
                            test_score = np.mean(test_losses)
                            test_scores = np.mean(test_losses, axis=0)
                        print(('%%     epoch %i, minibatch %i/%i, test %s of '
                               'best model %f %%\n') %
                              (epoch, minibatch_index + 1, n_train_batches, loss_str, test_score))

                        Wh = []
                        bh = []
                        for l in self.sigmoid_layers:
                            Wh.append(l.W.get_value())
                            bh.append(l.b.get_value())
                        sys.stderr.write('Saving all finetuned weights to ' + save_fnbase + '-best.npz...')
                        np.savez_compressed(save_fnbase + '-best.npz', W_h=Wh, b_h=bh,
                                            W_o=self.logLayer.W.get_value(),
                                            b_o=self.logLayer.b.get_value())
                        sys.stderr.write('DONE!\n')

                if patience <= iter:
                    done_looping = True
                    print '\n----\nPATIENCE EXCEEDED!\n----'
                    break

        end_time = time.clock()
        print(
            (
                'Optimization complete with best validation score of %f %%, '
                'on iteration %i, '
                'with test performance %s %%'
            )
            % (best_validation_loss * 100., best_iter + 1, ' '.join([ str(l) for l in (test_scores * 100.) ]) )
        )

        print >> sys.stderr, ('The training code for file ' +
                              os.path.split(__file__)[1] +
                              ' ran for %.2fm' % ((end_time - start_time) / 60.))
        Wh = []
        bh = []
        for l in self.sigmoid_layers:
            Wh.append(l.W.get_value())
            bh.append(l.b.get_value())
        sys.stderr.write('Saving all finetuned weights to ' + save_fnbase + '.npz...')
        np.savez_compressed(save_fnbase + '.npz', W_h=Wh, b_h=bh,
                               W_o=self.logLayer.W.get_value(),
                               b_o=self.logLayer.b.get_value())
        sys.stderr.write('DONE!\n')

    def load_pretrained_params(self, pretrained_file):
        params = np.load(pretrained_file)
        W = params['W'] if 'W' in params else params['W_h']
        b = params['b'] if 'b' in params else params['b_h']
        assert(len(W) >= self.n_layers and len(b) >= self.n_layers)
        for i in xrange(self.n_layers):
            layer_shape = self.sigmoid_layers[i].W.get_value().shape
            if (layer_shape[0] != W[i].shape[0] or layer_shape[1] != W[i].shape[1]) and (input_values is not None):
                print 'load pretrained params from ', pretrained_file, ' and initialize incrementally'
                W_sampled = np.asarray(
                        self.numpy_rng.normal(
                                loc=np.mean(W[i]), 
                                scale=np.std(W[i]), 
                                size=(W[i].shape[0], layer_shape[1]-W[i].shape[1])),
                        dtype=theano.config.floatX)
                
                b_sampled = np.asarray(
                        self.numpy_rng.normal(
                                loc=np.mean(b[i]), 
                                scale=np.std(b[i]), 
                                size=(layer_shape[1] - b[i].shape[0])),
                        dtype=theano.config.floatX)
                # get sim from input_values and change 
                W_raw_sampled = np.hstack((W[i], W_sampled))
                
                def euclidean(a,b):
                    return np.sqrt(np.sum(np.square(a-b)))

                input_1 = input_values[:, :W[i].shape[0]].T
                input_2 = input_values[:, W[i].shape[0]:].T

                ed_sim_mat = np.asarray(
                        [[euclidean(x, x2) for x in input_1] for x2 in input_2],
                        dtype=theano.config.floatX)
                ed_sim_mat = 1 - ed_sim_mat / np.amax(ed_sim_mat)
                                
                W_similar = np.asarray(
                        [np.dot(ed_sim_mat[i_n], W_raw_sampled)/np.sum(ed_sim_mat[i_n]) for i_n in xrange(layer_shape[0]-W[i].shape[0])],
                        dtype=theano.config.floatX)                
                
                W_value = np.vstack((W_raw_sampled, W_similar))
                assert(W_value.shape[0] == layer_shape[0] and W_value.shape[1] == layer_shape[1])
                #print i, len(self.sigmoid_layers)
                self.sigmoid_layers[i].W.set_value(W_value)
                b_value = np.hstack((b[i], b_sampled))
                self.sigmoid_layers[i].b.set_value(b_value)
                # update input_values for upper layers
                input_values = 1/(1+np.exp(-input_values.dot(W_value)-b_value))
            
            else:
                print 'load pretrained params from ', pretrained_file
                self.sigmoid_layers[i].W.set_value(W[i])
                self.sigmoid_layers[i].b.set_value(b[i]) 


def test_SdA(finetune_lr=0.1, pretraining_epochs=15,
             pretrain_lr=0.001, training_epochs=1000,
             dataset='mnist.pkl.gz', batch_size=1):
    """
    Demonstrates how to train and test a stochastic denoising autoencoder.

    This is demonstrated on MNIST.

    :type learning_rate: float
    :param learning_rate: learning rate used in the finetune stage
    (factor for the stochastic gradient)

    :type pretraining_epochs: int
    :param pretraining_epochs: number of epoch to do pretraining

    :type pretrain_lr: float
    :param pretrain_lr: learning rate to be used during pre-training

    :type n_iter: int
    :param n_iter: maximal number of iterations ot run the optimizer

    :type dataset: string
    :param dataset: path the the pickled dataset

    """

    datasets = load_data(dataset)

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # numpy random generator
    numpy_rng = np.random.RandomState(89677)
    print '... building the model'
    # construct the stacked denoising autoencoder class
    sda = NewSdA(
        numpy_rng=numpy_rng,
        n_ins=28 * 28,
        n_outs=10
    )

    sda.do_unsupervised_pretraining(train_set_x=train_set_x, epochs=pretraining_epochs,
                                    learn_rate=pretrain_lr, batch_size=batch_size)

    sda.do_supervised_finetuning(train_set_x=train_set_x, train_set_y=train_set_y,
                                 valid_set_x=valid_set_x, valid_set_y=valid_set_y,
                                 test_set_x=test_set_x, test_set_y=test_set_y,
                                 epochs=training_epochs, learn_rate=finetune_lr,
                                 batch_size=batch_size)

    return sda

if __name__ == '__main__':
    test_SdA(finetune_lr=0.1, pretraining_epochs=2,
             pretrain_lr=0.1, training_epochs=2,
             dataset='mnist.pkl.gz', batch_size=50)
