import numpy as np

predicted_probs = np.array([0.2, 0.3, 0.5])

true_label = 1

cross_entropy = -np.log(predicted_probs[true_label])

weight_value1 = 0.91
weight_value2 = 0.9

weight_distance = np.abs(weight_value1 - weight_value2)

mutual_information = cross_entropy / (weight_distance + 1e-8)

print("크로스 엔트로피:", cross_entropy)
print("가중치 거리:", weight_distance)
print("상호 정보량:", mutual_information)

