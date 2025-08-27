import eecs598
import torch
import torchvision
import matplotlib.pyplot as plt
import statistics
import random
from torchvision.utils import make_grid

# Increase the default figure size
plt.rcParams['figure.figsize'] = (10.0, 8.0)
plt.rcParams['font.size'] = 16

# Load entire cifar10 dataset
x_train, y_train, x_test, y_test = eecs598.data.cifar10()

print('Training set:', )
print('  data shape:', x_train.shape)
print('  labels shape: ', y_train.shape)
print('Test set:')
print('  data shape: ', x_test.shape)
print('  labels shape', y_test.shape)

# Visualize the dataset
classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
samples_per_class = 12
samples = []
for y, cls in enumerate(classes):
    plt.text(-4, 34 * y + 18, cls, ha='right')
    idxs, = (y_train == y).nonzero(as_tuple=True)
    for i in range(samples_per_class):
        idx = idxs[random.randrange(idxs.shape[0])].item()
        samples.append(x_train[idx])
img = torchvision.utils.make_grid(samples, nrow=samples_per_class)
plt.imshow(eecs598.tensor_to_image(img))
plt.axis('off')
plt.show()

# Subsample the dataset
num_train = 500
num_test = 250

x_train_sub, y_train_sub, x_test_sub, y_test_sub = eecs598.data.cifar10(num_train, num_test)

print('Training set:', )
print('  data shape:', x_train_sub.shape)
print('  labels shape: ', y_train_sub.shape)
print('Test set:')
print('  data shape: ', x_test_sub.shape)
print('  labels shape', y_test_sub.shape)

# Compute distances: Naive implementation
from knn import compute_distances_two_loops

torch.manual_seed(0)

dists = compute_distances_two_loops(x_train_sub, x_test_sub)
print('dists has shape: ', dists.shape)

# Visual debug
plt.imshow(dists.numpy(), cmap='gray', interpolation='none')
plt.colorbar()
plt.xlabel('test')
plt.ylabel('train')
plt.show()

# Compute distances: Vectorization
# One loop
from knn import compute_distances_one_loop

torch.manual_seed(0)
x_train_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)
x_test_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)

dists_one = compute_distances_one_loop(x_train_rand, x_test_rand)
dists_two = compute_distances_two_loops(x_train_rand, x_test_rand)
difference = (dists_one - dists_two).pow(2).sum().sqrt().item()
print('Difference: ', difference)
if difference < 1e-4:
    print('Good! The distance matrices match')
else:
    print('Uh-oh! The distance matrices are different')

# No loop
from knn import compute_distances_no_loops

torch.manual_seed(0)
x_train_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)
x_test_rand = torch.randn(100, 3, 16, 16, dtype=torch.float64)

dists_two = compute_distances_two_loops(x_train_rand, x_test_rand)
dists_none = compute_distances_no_loops(x_train_rand, x_test_rand)
difference = (dists_two - dists_none).pow(2).sum().sqrt().item()
print('Difference: ', difference)
if difference < 1e-4:
  print('Good! The distance matrices match')
else:
  print('Uh-oh! The distance matrices are different')

# Time comparison
import time

def timeit(f, *args):
    tic = time.time()
    f(*args)
    toc = time.time()
    return toc - tic

torch.manual_seed(0)
x_train_rand = torch.randn(500, 3, 32, 32)
x_test_rand = torch.randn(500, 3, 32, 32)

two_loop_time = timeit(compute_distances_two_loops, x_train_rand, x_test_rand)
print('Two loop version took %.2f seconds' % two_loop_time)

one_loop_time = timeit(compute_distances_one_loop, x_train_rand, x_test_rand)
speedup = two_loop_time / one_loop_time
print('One loop version took %.2f seconds (%.1fX speedup)'
      % (one_loop_time, speedup))

no_loop_time = timeit(compute_distances_no_loops, x_train_rand, x_test_rand)
speedup = two_loop_time / no_loop_time
print('No loop version took %.2f seconds (%.1fX speedup)'
      % (no_loop_time, speedup))

# Predict labels
from knn import predict_labels

torch.manual_seed(0)
dists = torch.tensor([
    [0.3, 0.4, 0.1],
    [0.1, 0.5, 0.5],
    [0.4, 0.1, 0.2],
    [0.2, 0.2, 0.4],
    [0.5, 0.3, 0.3],
])
y_train_1 = torch.tensor([0, 1, 0, 1, 2])
y_pred_expected_1 = torch.tensor([1, 0, 0])
y_pred_1 = predict_labels(dists, y_train, k=3)
print(y_pred_1)
correct = y_pred_1.tolist() == y_pred_expected_1.tolist()
print('Correct: ', correct)

# Knn Class
from knn import KnnClassifier

num_test = 10000
num_train = 20
num_classes = 5

# Generate random training and test data
torch.manual_seed(128)
x_train_rand_1 = torch.rand(num_train, 2)
y_train_rand_1 = torch.randint(num_classes, size=(num_train,))
x_test_rand_1 = torch.rand(num_test, 2)
classifier = KnnClassifier(x_train_rand_1, y_train_rand_1)

# Plot predictions for different values of k
for k in [1, 3, 5]:
    y_test_rand_1 = classifier.predict(x_test_rand_1, k=k)
    plt.gcf().set_size_inches(8, 8)
    class_colors = ['r', 'g', 'b', 'k', 'y']
    train_colors = [class_colors[c] for c in y_train_rand_1]
    test_colors = [class_colors[c] for c in y_test_rand_1]
    plt.scatter(x_test_rand_1[:, 0], x_test_rand_1[:, 1],
                color=test_colors, marker='o', s=32, alpha=0.05)
    plt.scatter(x_train_rand_1[:, 0], x_train_rand_1[:, 1],
                color=train_colors, marker='*', s=128.0)
    plt.title('Predictions for k = %d' % k, size=16)
    plt.show()

# Image classification on CIFAR-10
torch.manual_seed(0)
num_train = 5000
num_test = 500
x_train_sub, y_train_sub, x_test_sub, y_test_sub = eecs598.data.cifar10(num_train, num_test)

classifier = KnnClassifier(x_train_sub, y_train_sub)
# classifier.check_accuracy(x_test, y_test, k=1)
classifier.check_accuracy(x_test_sub, y_test_sub, k=5)

# Use cross-validation to choose hyperparameter k
from knn import knn_cross_validate

torch.manual_seed(0)

k_to_accuracies = knn_cross_validate(x_train_sub, y_train_sub, num_folds=5)

for k, accs in sorted(k_to_accuracies.items()):
  print('k = %d got accuracies: %r' % (k, accs))

ks, means, stds = [], [], []
torch.manual_seed(0)
for k, accs in sorted(k_to_accuracies.items()):
  plt.scatter([k] * len(accs), accs, color='g')
  ks.append(k)
  means.append(statistics.mean(accs))
  stds.append(statistics.stdev(accs))
plt.errorbar(ks, means, yerr=stds)
plt.xlabel('k')
plt.ylabel('Cross-validation accuracy')
plt.title('Cross-validation on k')
plt.show()

# Select the best k and rerun classifier on full dataset
from knn import knn_get_best_k

best_k = 1
torch.manual_seed(0)

best_k = knn_get_best_k(k_to_accuracies)
print('Best k is ', best_k)

classifier = KnnClassifier(x_train, y_train)
classifier.check_accuracy(x_test, y_test, k=best_k)
# Best k is 12
# Got 3428 / 10000 correct; accuracy is 34.28%