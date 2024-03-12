import numpy as np
import matplotlib.pyplot as plt
from numpy import log,dot,exp,shape
import random
import pandas as pd

# Creating the Data Set

SAMPLES = 2500
IMAGESIZE = 20
FEATURES = IMAGESIZE * IMAGESIZE
CLASSES = 5

def generate_wires():
    # Colors: 1 (Red), 2 (Blue), 3 (Yellow), 4 (Green)
    imagearr = np.full((IMAGESIZE,IMAGESIZE) , 0)
    wires = []
    redDone = False
    blueDone = False
    yellowDone = False
    greenDone = False
    RowDone = np.full((IMAGESIZE), 0)
    ColDone = np.full((IMAGESIZE), 0)
    wireToCut = 0
    layout = np.full(4,0)
    start_with_row = True
    iter = 0;
    for _ in range(4):
        iter = iter + 1
        if start_with_row:
          rowFound = False
          while rowFound == False:
            row = random.randint(0, IMAGESIZE - 1)
            if RowDone[row] == 0:
              rowFound = True
              RowDone[row] = 1
          found = False
          while (found == False):
            color = random.randint(1, 4)
            if color == 1 and redDone == False:
              redDone = True
              found = True
            if color == 2 and blueDone == False:
              blueDone = True
              found = True
            if color == 3 and yellowDone == False:
              yellowDone = True
              found = True
            if color == 4 and greenDone == False:
              greenDone = True
              found = True
            imagearr[row, :] = color
            wires.append(('row', color, row))
        else:
            colFound = False
            while colFound == False:
              col = random.randint(0, IMAGESIZE - 1)
              if ColDone[col] == 0:
                colFound = True
                ColDone[col] = 1
            found = False
            while (found == False):
              color = random.randint(1, 4)
              if color == 1 and redDone == False:
                redDone = True
                found = True
              if color == 2 and blueDone == False:
                blueDone = True
                found = True
              if color == 3 and yellowDone == False:
                yellowDone = True
                found = True
              if color == 4 and greenDone == False:
                greenDone = True
                found = True
            imagearr[:, col] = color
            wires.append(('col', color, col))
        layout[iter - 1] = color
        start_with_row = not start_with_row

    if ((layout[0] == 1) and (layout[1]==3 or layout[2]==3 or layout[3] == 3)) or ((layout[1] == 1) and (layout[2]==3 or layout[3] == 3)) or ((layout[2] == 1) and (layout[3]==3)):
      wireToCut = layout[3]
    else:
      wireToCut = 0
    dangerous = any(w1[1] == 1 and w2[1] == 3 for w1, w2 in zip(wires, wires[1:]))

    # Identify the third wire color if dangerous
    wire_to_cut = wires[2][1] if dangerous else None

    vectorized_image = imagearr.flatten()

    return (imagearr, wireToCut)

"""Model 1"""

#Using Min-Max  xi = (xi - min(x))/(max(x) - min(x))
#Feature 1
def standardize(X_tr):
    for i in range(shape(X_tr)[1]):
      X_tr[:,i] = (X_tr[:,i] - np.min(X_tr[:,i]))/(np.max(X_tr[:,i]) - np.min(X_tr[:,i]))

def F1_score(y,y_hat):
    tp,tn,fp,fn = 0,0,0,0
    for i in range(len(y)):
        if y[i] == 1 and y_hat[i] == 1:
            tp += 1
        elif y[i] == 1 and y_hat[i] == 0:
            fn += 1
        elif y[i] == 0 and y_hat[i] == 1:
            fp += 1
        elif y[i] == 0 and y_hat[i] == 0:
            tn += 1
    print(f'tp {tp}, tn {tn}, fp {fp}, fn {fn}')
    precision = 0
    if (tp+fp != 0):
      precision = tp/(tp+fp)
    recall = 0
    if (tp+fn != 0):
      recall = tp/(tp+fn)
    f1_score = 0
    if (precision+recall != 0):
      f1_score = 2*precision*recall/(precision+recall)
    return f1_score
  # F1 score is for precision and recall to increase precision

def initialize():
  X = np.zeros((SAMPLES,4))
  y = np.zeros(SAMPLES)
  for i in range(SAMPLES):
    (img, wireToCut) = generate_wires()
    y[i] = wireToCut
    for m in range(IMAGESIZE):
      for n in range(IMAGESIZE):
        if (img[m][n] == 1):
          X[i][0] += 1
        if (img[m][n] == 2):
          X[i][1] += 1
        if (img[m][n] == 3):
          X[i][2] += 1
        if (img[m][n] == 4):
          X[i][3] += 1

  return X,y

X,y = initialize()
print(X)
  #change this to count the pixels and send the count of pixels, instead of an image - send in an array with the count of 4 colors
 # extracting 4 features which is the count of the pixels

class LogisticRegression():
    def __init__(self, lr=0.01, n_iters=2000, call_type = 0):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.weights2 = None
        self.weights1 = 1.0
        self.bias = None
        self.call_type = call_type # (0 linear else NonLinear)


    def predictBase(self, X):
        n_samples, n_features = X.shape
        linear_pred = np.dot(X, self.weights)
        if (self.call_type == 0):
           return linear_pred + self.bias
        else:
          # x + xi**2 - 2 prod(xi)
          #taking the predict function and making it nonlinear
          X2 = np.power(X,2)
          nonlinear_pred = np.dot(X2, self.weights2)
          eff = np.prod(X, axis = 1) * (2 * self.weights1 * -1)
          totpred = np.add(linear_pred,nonlinear_pred, eff)
          return totpred + self.bias

    def cost(self,X,y):
       pred = self.predictBase(X)
       cost0 = y.T.dot(log(self.sigmoid(pred)))
       cost1 = (1-y).T.dot(log(1-self.sigmoid(pred)))
       cost = -((cost1 + cost0))/len(y)
       return cost

    def sigmoid(self,x):
        return 1/(1+np.exp(-x))

    def fit(self, Xtrain, y):
        n_samples, n_features = np.shape(Xtrain)
        x0 = np.ones(n_samples)
        X = np.c_[x0,Xtrain]
        self.weights = np.zeros(n_features + 1)
        self.weights2 = np.zeros(n_features + 1)
        self.bias = 0

        for i in range(self.n_iters):
            pred = self.predictBase(X)
            predictions = self.sigmoid(pred)

            dw = (1/n_samples) * np.dot(X.T, (predictions - y))
            db = (1/n_samples) * np.sum(predictions-y)

            self.weights = self.weights - self.lr*dw
            self.weights2 = self.weights2 - self.lr*dw
            self.weights1 = self.weights1 - self.lr*np.mean(dw)
            self.bias = self.bias - self.lr*db
            epochCost = self.cost(X, y)
            if (i%250 == 0):
               print(f'Interation {i} loss {epochCost}')
            if (epochCost < 0.2):
              return

    def predict(self, Xpred):
      n_samples, n_features = np.shape(Xpred)
      x0 = np.ones(n_samples)
      X = np.c_[x0,Xpred]
      pred = self.predictBase(X)
      y_pred = self.sigmoid(pred)
      class_pred = [0 if y<=0.5 else 1 for y in y_pred]
      return class_pred

def trainTestSplit(X, y, test_size=0.2):
    n_samples, n_features = np.shape(X)
    trains = int(n_samples * (1-test_size))
    X_tr = np.zeros((trains, n_features))
    y_tr = np.zeros(trains)
    X_te = np.zeros((n_samples - trains, n_features))
    y_te = np.zeros((n_samples - trains))
    print(np.shape(X))
    for i in range(n_samples):
      if (i < trains):
        X_tr[i] = X[i]
        y_tr[i] = y[i]
      else:
        X_te[i-trains] = X[i]
        y_te[i-trains] = y[i]

    return X_tr, X_te, y_tr, y_te

print(f'Initializing Training Array.')
SAMPLES=2500
X,y = initialize()
standardize(X)
yn = np.zeros(SAMPLES)
for i in range(SAMPLES):
  if (y[i] != 0):
     yn[i] = 1
  #print(f'output is {yn[i]}')
print(f'Splitting Training/Testing Array.')
X_tr,X_te,y_tr,y_te = trainTestSplit(X, yn,test_size=0.2)
print(f'Starting Regression Training.')
obj1 = LogisticRegression(lr=0.01, n_iters=500, call_type=1)
model= obj1.fit(X_tr,y_tr)
print(f'Starting Testing.')
y_pred = obj1.predict(X_te)
y_train = obj1.predict(X_tr)
#Let's see the f1-score for training and testing data
print('Training result')
#for i in range(len(y_train)):
#  print(f'Response {y_train[i]},  Predict {y_tr[i]}')
f1_score_tr = F1_score(y_tr,y_train)
print('Testing result')
#for i in range(len(y_pred)):
#  print(f'Response {y_pred[i]},  Predict {y_te[i]}')
f1_score_te = F1_score(y_te,y_pred)
print(f'Training Score: {f1_score_tr}')
print(f'Prediction Score: {f1_score_te}')

"""Model 2"""

def generate_wires():
    #Colors: 1 (Red), 2 (Blue), 3 (Yellow), 4 (Green)
    imagearr = np.full((20,20) , 0)
    output = np.full((4,4), 0)
    wires = []
    redDone = False
    blueDone = False
    yellowDone = False
    greenDone = False
    RowDone = np.full((20), 0)
    ColDone = np.full((20), 0)

    start_with_row = True

    for _ in range(4):
        if start_with_row:
          rowFound = False
          while rowFound == False:
            row = random.randint(0, 19)
            if RowDone[row] == 0:
              rowFound = True
              RowDone[row] = 1
          found = False
          while (found == False):
            color = random.randint(1, 4)
            if color == 1 and redDone == False:
              redDone = True
              found = True
            if color == 2 and blueDone == False:
              blueDone = True
              found = True
            if color == 3 and yellowDone == False:
                yellowDone = True
                found = True
            if color == 4 and greenDone == False:
              greenDone = True
              found = True
            imagearr[row, :] = color
            wires.append(('row', color, row))
        else:
            colFound = False
            while colFound == False:
              col = random.randint(0, 19)
              if ColDone[col] == 0:
                colFound = True
                ColDone[col] = 1
            found = False
            while (found == False):
              color = random.randint(1, 4)
              if color == 1 and redDone == False:
                redDone = True
                found = True
              if color == 2 and blueDone == False:
                blueDone = True
                found = True
              if color == 3 and yellowDone == False:
                yellowDone = True
                found = True
              if color == 4 and greenDone == False:
                greenDone = True
                found = True
            imagearr[:, col] = color
            wires.append(('col', color, col))

        start_with_row = not start_with_row

    dangerous = any(w1[1] == 1 and w2[1] == 3 for w1, w2 in zip(wires, wires[1:]))

    #Identify the third wire color if dangerous
    wire_to_cut = wires[2][1] if dangerous else None

    vectorized_image = imagearr.flatten()

    return imagearr, dangerous, wire_to_cut

def generate_dataset(size):
    data = []

    for _ in range (size):
        vectorized_image, dangerous, wire_to_cut = generate_wires()
        data.append({
            'vectorized_image': vectorized_image,
            'dangerous': dangerous,
            'wire_to_cut': wire_to_cut
        })

    return pd.DataFrame(data)

dataset = generate_dataset(100)
print(dataset)

def one_hot_encode_flatten(imagearr):
    num_colors = 4

    one_hot_encoded = np.zeros((imagearr.size, num_colors))

    flattened_imagearr = imagearr.flatten()

    for i, color in enumerate(flattened_imagearr):
        if color > 0:
            one_hot_encoded[i, color - 1] = 1

    return one_hot_encoded.flatten()

def generate_one_hot_dataset(size):
    data = []

    for _ in range(size):
        vectorized_image, dangerous, wire_to_cut = generate_wires()
        encoded_image = one_hot_encode_flatten(vectorized_image)
        data.append({
            'encoded_image': encoded_image,
            'dangerous': dangerous,
            'wire_to_cut': wire_to_cut
        })

    return pd.DataFrame(data)

dataset = generate_one_hot_dataset(100)

def generate_dangerous_dataset(target_size):
    dangerous_dataset = pd.DataFrame()

    while len(dangerous_dataset) < target_size:
        batch = generate_one_hot_dataset(100)
        dangerous_batch = batch[batch['dangerous'] == 1]
        dangerous_dataset = pd.concat([dangerous_dataset, dangerous_batch])

    if len(dangerous_dataset) > target_size:
        dangerous_dataset = dangerous_dataset.iloc[:target_size]

    return dangerous_dataset

# target_size = 100  # Set your target size
# dangerous_dataset = generate_dangerous_dataset(target_size)

def train_test_split(dataset, test_size=0.2):
    shuffled_dataset = dataset.sample(frac=1).reset_index(drop=True)

    test_set_size = int(len(shuffled_dataset) * test_size)
    train_set_size = len(shuffled_dataset) - test_set_size

    train_set = shuffled_dataset[:train_set_size]
    test_set = shuffled_dataset[-test_set_size:]

    return train_set, test_set

dataset = generate_one_hot_dataset(100)

train_set, test_set = train_test_split(dataset)

target_size = 100
dangerous_dataset = generate_dangerous_dataset(target_size)

X = np.array(dangerous_dataset['encoded_image'].tolist())

Y = np.array(dangerous_dataset['wire_to_cut'].tolist())

train_set, test_set = train_test_split(dangerous_dataset)

X_train = np.array(train_set['encoded_image'].tolist())
Y_train = np.array(train_set['wire_to_cut'].tolist())

X_test = np.array(test_set['encoded_image'].tolist())
Y_test = np.array(test_set['wire_to_cut'].tolist())

print("X (Features):")
print(X[:5])

print("\nY (Labels):")
print(Y[:5])

def softmax(z):
    exp_z = np.exp(z - np.max(z, axis = 1, keepdims = True))
    return exp_z / np.sum(exp_z, axis = 1, keepdims = True)

def compute_gradients_softmax(X, y_true, y_pred):
    m = X.shape[0]
    dw = np.dot(X.T, (y_pred - y_true)) / m
    db = np.sum(y_pred - y_true, axis=0, keepdims=True) / m

    return {'dw': dw, 'db': db}

def train_softmax_regression(X_train, y_train, learning_rate=0.01, num_iterations=2000):
    n_features = X_train.shape[1]
    n_classes = y_train.shape[1]
    w = np.zeros((n_features, n_classes))
    b = np.zeros((1, n_classes))
    losses = []

    for i in range(num_iterations):
        scores = np.dot(X_train, w) + b
        y_pred = softmax(scores)

        loss = -np.sum(y_train * np.log(y_pred)) / X_train.shape[0]
        losses.append(loss)

        gradients = compute_gradients_softmax(X_train, y_train, y_pred)

        w -= learning_rate * gradients['dw']
        b -= learning_rate * gradients['db']

        if i % 100 == 0:
            print(f"Iteration {i}: Loss = {loss}")

    return w, b, losses

def one_hot_encode_labels(labels, num_classes):
    one_hot_encoded = np.zeros((len(labels), num_classes))
    for i, label in enumerate(labels):
        int_label = int(label)  # Convert float to int
        one_hot_encoded[i, int_label - 1] = 1
    return one_hot_encoded

num_classes = 4
Y_train_encoded = one_hot_encode_labels(Y_train, num_classes)

w_softmax, b_softmax, losses_softmax = train_softmax_regression(X_train, Y_train_encoded)

def test_softmax_regression(X_test, Y_test, w, b):
    if Y_test.ndim == 1:
        Y_test = one_hot_encode_labels(Y_test, num_classes)

    scores = np.dot(X_test, w) + b
    predictions = softmax(scores)

    if predictions.ndim == 1:
        predictions = np.expand_dims(predictions, axis=0)

    predicted_classes = np.argmax(predictions, axis=1)
    actual_classes = np.argmax(Y_test, axis=1)

    accuracy = np.mean(predicted_classes == actual_classes)
    return accuracy

# accuracy = test_softmax_regression(X_test, Y_test, w_softmax, b_softmax)
# print(f"Model Accuracy: {accuracy * 100:.2f}%")

def count_colors(image, num_colors=4):
    color_counts = [np.count_nonzero(image == color) for color in range(1, num_colors + 1)]
    return color_counts

def add_polynomial_features(X, degree):

    if degree < 2:
        return X

    n_samples, n_features = X.shape
    features = [X]

    for d in range(2, degree + 1):
        for i in range(n_features):
            # Raise feature to the power of d
            powered_feature = X[:, i:i+1] ** d
            features.append(powered_feature)

    return np.concatenate(features, axis=1)



def count_color_intersections(image, color_pairs):
    intersections = []
    image_size = image.shape[0]

    for color_pair in color_pairs:
        count = 0
        for i in range(image_size):
            for j in range(image_size - 1):
                # Check horizontal pairs
                if image[i, j] == color_pair[0] and image[i, j + 1] == color_pair[1]:
                    count += 1
                # Check vertical pairs
                if i < image_size - 1 and image[i, j] == color_pair[0] and image[i + 1, j] == color_pair[1]:
                    count += 1
        intersections.append(count)

    return intersections

def preprocess_dataset_with_intersections(dataset, color_pairs):
    processed_features = []

    for _, row in dataset.iterrows():
        image = row['encoded_image']
        color_counts = [np.sum(image == color) for color in range(1, num_colors + 1)]
        color_intersections = count_color_intersections(image, color_pairs)

        feature_with_counts_and_intersections = np.concatenate([image, color_counts, color_intersections])
        processed_features.append(feature_with_counts_and_intersections)

    return np.array(processed_features)

def decode_image(encoded_image, image_size=20, num_classes=4):
    decoded_image = np.zeros((image_size, image_size), dtype=int)

    # Correctly reshape the encoded image to a 3D array (image_size x image_size x num_classes)
    encoded_image_reshaped = encoded_image.reshape((image_size, image_size, num_classes))

    for i in range(image_size):
        for j in range(image_size):
            # Find the color index with the maximum value (one-hot encoded)
            color_index = np.argmax(encoded_image_reshaped[i, j])
            decoded_image[i, j] = color_index + 1  # Colors are indexed from 1

    return decoded_image

def is_intersection(pixel, adjacent_pixels):
    return any(adjacent_pixel != pixel for adjacent_pixel in adjacent_pixels)

def get_3x3_area(matrix, row, col):
    return matrix[row-1:row+2, col-1:col+2].flatten()

def extract_intersection_features(image, image_size=20):
    features = []
    for i in range(1, image_size-1):
        for j in range(1, image_size-1):
            pixel = image[i, j]
            adjacent_pixels = [image[i-1, j], image[i+1, j], image[i, j-1], image[i, j+1]]  # Up, down, left, right

            if is_intersection(pixel, adjacent_pixels):
                area_3x3 = get_3x3_area(image, i, j)
                features.append(area_3x3)

    return features

def extract_intersection_features(encoded_image, image_size=20, color_mapping=None):
    if color_mapping is None:
        color_mapping = {0: 'Red', 1: 'Blue', 2: 'Yellow', 3: 'Green'}

    decoded_image = decode_image(encoded_image, image_size, len(color_mapping))

    features = []
    for i in range(1, image_size-1):
        for j in range(1, image_size-1):
            pixel = decoded_image[i, j]
            adjacent_pixels = [decoded_image[i-1, j], decoded_image[i+1, j], decoded_image[i, j-1], decoded_image[i, j+1]]

            if is_intersection(pixel, adjacent_pixels):
                area_3x3 = get_3x3_area(decoded_image, i, j)
                color_order = []

                for color_index in range(len(color_mapping)):
                    color_name = color_mapping[color_index]
                    if color_name in area_3x3:
                        color_order.append(color_name)

                if len(color_order) == 2:
                    features.append(color_order)

    features_array = np.array(features)

    return features_array

def decode_image(encoded_image, image_size=20, num_colors=4):
    decoded_image = np.argmax(encoded_image.reshape((image_size, image_size, num_colors)), axis=-1) + 1
    return decoded_image

def is_intersection(pixel, adjacent_pixels):
    return any(adj_pixel != pixel for adj_pixel in adjacent_pixels)

def get_3x3_area(image, row, col):
    return image[row-1:row+2, col-1:col+2]

def analyze_area(area):
    color_counts = np.bincount(area.flatten(), minlength=5)[1:]
    return color_counts

def extract_3x3_features(encoded_image, image_size=20, num_classes=4, fixed_length=100):
    decoded_image = decode_image(encoded_image, image_size, num_classes)
    features = []

    for i in range(1, image_size-1):
        for j in range(1, image_size-1):
            if len(features) >= fixed_length:
                break
            pixel = decoded_image[i, j]
            adjacent_pixels = [decoded_image[i-1, j], decoded_image[i+1, j], decoded_image[i, j-1], decoded_image[i, j+1]]

            if is_intersection(pixel, adjacent_pixels):
                area_3x3 = get_3x3_area(decoded_image, i, j)
                area_features = analyze_area(area_3x3)
                features.append(area_features.flatten())

    while len(features) < fixed_length:
        features.append(np.zeros_like(features[0]))

    return np.array(features).flatten()


encoded_image = one_hot_encode_labels(generate_wires()[0].flatten(), 4)
features = extract_3x3_features(encoded_image)

def raise_power(matrix, power):
    return np.power(matrix, power)

def extract_and_power_3x3_features(image, image_size=20, num_classes=4, degree=2, max_intersections=10):
    decoded_image = decode_image(image, image_size, num_classes)
    features = []

    for i in range(1, image_size-1):
        for j in range(1, image_size-1):
            if len(features) >= max_intersections:
                break
            pixel = decoded_image[i, j]
            adjacent_pixels = [decoded_image[i-1, j], decoded_image[i+1, j], decoded_image[i, j-1], decoded_image[i, j+1]]

            if is_intersection(pixel, adjacent_pixels):
                area_3x3 = get_3x3_area(decoded_image, i, j)

                for d in range(1, degree + 1):
                    powered_area = raise_power(area_3x3, d)
                    features.append(powered_area.flatten())

    while len(features) < max_intersections:
        features.append(np.zeros(9 * degree))

    return np.concatenate(features)

def decode_image1(encoded_image, image_size=20, num_classes=4):

    decoded_image = np.zeros((image_size, image_size), dtype=int)

    encoded_image_reshaped = encoded_image.reshape((image_size, image_size, num_classes))

    for i in range(image_size):
        for j in range(image_size):
            color_index = np.argmax(encoded_image_reshaped[i, j])
            decoded_image[i, j] = color_index + 1

    return decoded_image

import numpy as np

def run_multiple_trainings(num_iterations, dataset_size, test_size, learning_rate, training_epochs, num_classes, degree=2, image_size=20):
    accuracies = []

    for i in range(num_iterations):
        print(f"Running iteration {i+1}/{num_iterations}")

        dataset = generate_dangerous_dataset(dataset_size)

        X = np.array(dataset['encoded_image'].tolist())
        Y = one_hot_encode_labels(np.array(dataset['wire_to_cut']), num_classes)

        train_set, test_set = train_test_split(dataset, test_size=test_size)
        X_train = np.array(train_set['encoded_image'].tolist())
        Y_train = one_hot_encode_labels(np.array(train_set['wire_to_cut']), num_classes)
        X_test = np.array(test_set['encoded_image'].tolist())
        Y_test = one_hot_encode_labels(np.array(test_set['wire_to_cut']), num_classes)

        X_train_enhanced = []
        X_test_enhanced = []

        for image in X_train:
            image_decoded = decode_image1(image, image_size, num_classes)

            poly_features = add_polynomial_features(image_decoded, degree)

            color_intersections = count_color_intersections(image_decoded, [(1, 3), (2, 4)])

            area_features = extract_3x3_features(image, image_size, num_classes)
            area_features_flat = area_features.flatten()

            area_features_poly = extract_and_power_3x3_features(image, image_size=20, num_classes=4, degree=area_degree)
            area_features_poly_flat = area_features_poly.flatten()

            combined_features = np.concatenate([area_features_poly_flat, area_features_flat, poly_features.flatten(), color_intersections])
            X_train_enhanced.append(combined_features)

        for image in X_test:
            image_decoded = decode_image1(image, image_size, num_classes)

            poly_features = add_polynomial_features(image_decoded, degree)

            color_intersections = count_color_intersections(image_decoded, [(1, 3), (2, 4)])

            area_features = extract_3x3_features(image, image_size, num_classes)
            area_features_flat = area_features.flatten()

            area_features_poly = extract_and_power_3x3_features(image, image_size=20, num_classes=4, degree=area_degree)
            area_features_poly_flat = area_features_poly.flatten()

            combined_features = np.concatenate([area_features_poly_flat, area_features_flat, poly_features.flatten(), color_intersections])
            X_test_enhanced.append(combined_features)

        X_train_enhanced = np.array(X_train_enhanced)
        X_test_enhanced = np.array(X_test_enhanced)

        w_softmax, b_softmax, losses_softmax = train_softmax_regression(X_train_enhanced, Y_train, learning_rate, training_epochs)

        accuracy = test_softmax_regression(X_test_enhanced, Y_test, w_softmax, b_softmax)
        accuracies.append(accuracy)

        print(f"Iteration {i+1} completed with accuracy: {accuracy * 100:.2f}%")

    average_accuracy = sum(accuracies) / len(accuracies)
    print(f"Average accuracy over {num_iterations} iterations: {average_accuracy * 100:.2f}%")

    return accuracies

num_iterations = 10
dataset_size = 500
test_size = 0.2
learning_rate = 0.01
training_epochs = 2000
num_classes = 4
degree = 3
area_degree = 3

accuracies = run_multiple_trainings(num_iterations, dataset_size, test_size, learning_rate, training_epochs, num_classes, degree)

"""Bonus"""

import matplotlib.pyplot as plt

# Data for the graph
iterations = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
loss_values = [
    1.3862943611198904, 37.2774847717466, 26.882630612747906, 16.72560767845869,
    25.140207555251465, 7.612886601913144, 22.586959359928514, 4.363707707198381,
    21.931644752196053, 1.713137047647497, 0.9732450799317175, 0.05527768596299139,
    0.029268575154022373, 0.0006766768689506514, 0.0005694107900608564, 0.0005080717257410016,
    0.0004655260982679739, 0.0004332428890198554, 0.00040733612260008623, 0.00038574999713248363
]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, loss_values, marker='o')
plt.title("Loss Values for n=500")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Data for the graph
iterations = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
loss_values = [
    1.3862943611198904, 99.9866887748426, 83.0007465120711, 42.42125195870276,
    23.194949049337456, 33.535071062305065, 22.618864326239436, 44.22119773084451,
    4.6526256067961524, 18.18436377238152, 16.076602427834906, 22.663393736119225,
    23.471830085513947, 12.126178550126147, 10.63435644494223, 9.598171095298532,
    17.612238030153268, 12.629749664633957, 13.670228378633931, 5.05994123057616
]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, loss_values, marker='o')
plt.title("Loss Values for n=1000")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Data for the graph
iterations = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
loss_values = [
    1.3862943611198908, 70.39093486873398, 45.825028224041624, 71.70680384032443,
    23.7714248914984, 81.10794113628651, 42.906189529914684, 22.40888531949733,
    45.83289803365472, 38.05709708060146, 57.53901268983504, 24.330908753535766,
    40.99383804044721, 49.622574475746255, 38.0946040180697, 39.723389214063,
    55.50871141138299, 32.533264531990135, 25.97204893242888, 28.94074143438292
]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, loss_values, marker='o')
plt.title("Loss Values for n=2500")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()

import matplotlib.pyplot as plt

# Data for the graph
iterations = [0, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
loss_values = [
    1.3862943611198906, 84.97665571932879, 71.95022303633239, 39.71623201257891,
    84.2200315727115, 44.15007787956727, 56.041675678840754, 57.21664339126872,
    51.73691250845301, 20.658235844506464, 56.30555995679816, 41.250578667753324,
    45.11204539347357, 44.10206937885407, 40.217394097989676, 51.304659422641386,
    58.50636274837574, 38.50949440616784, 63.4203292426201, 36.875612137584454
]

# Creating the plot
plt.figure(figsize=(10, 6))
plt.plot(iterations, loss_values, marker='o')
plt.title("Loss Values for n=5000")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.grid(True)
plt.show()
