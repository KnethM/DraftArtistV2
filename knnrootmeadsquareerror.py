from captain_mode_draft import Draft
from math import sqrt


d = Draft('NN_hiddenunit120_dota.pickle', 'mlp.pickle', 'knn2_5_vwhd', 'mcts_300_2')  # instantiate board
knn = d.player_models[0]
print("hej")
predictions = []
facts = []
for team in d.controllers:
    for controller in team:
        predictions.append(knn.predict_classification(knn.controllergrid, controller, 5, 5))
        facts.append(controller[5])

j = 0
for i in range(len(predictions)):
    j += (predictions[i]-facts[i])**2

RMSE = sqrt(j/10)
print(RMSE)

