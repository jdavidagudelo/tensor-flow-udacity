from sklearn.linear_model import LogisticRegression
from utils import plot_utils


def train_and_predict(sample_size, train_data_set, train_labels, test_data_set, test_labels,  image_size=0):
    regression = LogisticRegression()
    x_train = train_data_set[:sample_size].reshape(sample_size, image_size * image_size)
    y_train = train_labels[:sample_size]
    regression.fit(x_train, y_train)
    x_test = test_data_set.reshape(test_data_set.shape[0], image_size * image_size)
    y_test = test_labels
    predicted_labels = regression.predict(x_test)
    print('Accuracy:', regression.score(x_test, y_test), 'when sample_size=', sample_size)
    plot_utils.display_sample_data_set(test_data_set, predicted_labels, 'sample_size=' + str(sample_size))