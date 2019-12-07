from captain_mode_draft import Draft
from math import sqrt


d = Draft('NN_hiddenunit120_dota.pickle', 'mlp.pickle', 'knn_5_euclid', 'mcts_300_2')  # instantiate board
knn = d.player_models[0]
predictions = []
facts = []
test = knn.controllergrid[-200:]
train = knn.controllergrid[:-200]
for controller in test:
    predictions.append(knn.predict_classification(train, controller, 5, 5))
    facts.append(controller[5])

j = 0
for i in range(len(predictions)):
    j += (predictions[i]-facts[i])**2

RMSE = sqrt(j/len(test))
print(RMSE)

