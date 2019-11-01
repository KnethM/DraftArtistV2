from sklearn.neural_network import MLPClassifier

# class for the MLPClassifier Neural network.
class classifier:

    Dotaclf = MLPClassifier(activation='relu', alpha=0.0001, batch_size='auto', beta_1=0.9,
              beta_2=0.999, early_stopping=False, epsilon=1e-08,
              hidden_layer_sizes=(150,10), learning_rate='constant',
              learning_rate_init=0.001, max_iter=200, momentum=0.9,
              n_iter_no_change=5, nesterovs_momentum=True, power_t=0.5,
              random_state=None, shuffle=True, solver='adam', tol=0.0001,
              validation_fraction=0.1, verbose=False, warm_start=False)


# The classifier uses two arrays as its input array1 consist of (n_samples, n_features) the samples is our training data
# and features is the features that we have in the sample
# array2 is the output that we would like, in this case it will be either 1 for radiant win or 0 for radiant loss.

