from tensorflow import keras

import neptune

def test(model: keras.Sequential, 
         test_X, 
         test_y, 
         run:neptune.Run):
    eval_metrics = model.evaluate(test_X, test_y, verbose=0)
    for j, metric in enumerate(eval_metrics):
        run["eval/{}".format(model.metrics_names[j])] = metric
    print("Accuracy on testing data: {:.2%}".format(eval_metrics[1]))