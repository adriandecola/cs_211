"""
Course:        DCS 211 Winter 2021 Module D
Assignment:    Project 4b
Topic:         k-NN Machine Learning
Purpose:       Implementing a functioning k-NN classifier for the provided
               digts.csv and iris.csv data.

Student Name: Adrian deCola

Other students outside my pair that I received help from ('N/A' if none):
N/A

Other students outside my pair that I gave help to ('N/A' if none):
N/A

Citations/links of external references used ('N/A' if none):
N/A

Comments on Results:
--------------------
    iris.csv:
        adjusted_random_score:  0.8185077302263052
        v_measure:              0.8389334444194805
        computed accuracy:      0.9286

    digits.csv:
        adjusted_random_score:  0.9755486568033007
        v_measure:              0.9776352988191485
        computed accuracy:      0.9887

        These are the scores for the iris.csv data and its predicted labels. The
        best value of k used was 4. As we can see, dispite correctly labeling
        about 92% of the data, as given from our compued accuracy, the adjusted
        random score and the v measure gave lower scores. This is due to the
        small amount of labels giving variance, or uncertainty in the accuracy
        of our predictions. For example, the adjusted random score, whose values
        range from -1 to 1, with zero representing a random score assignment,
        cannot be too sure that our accuracy was not somewhat random as there is
        not a lot of labels to predict. The V-measure score measure homogenuity
        and completeness, focusing more on how accurate each cluster is. Since
        there are only three clusters one label being incorrectly predicted causes
        more inaccuracy across clusters, as the cluster it is assigned to contains
        a member of another class. In our digits.csv data there was lots more data
        to test. The best value of k used was 1. All three of our scorings gave
        around the same score, in the high .9s.

Difference with Standard Scalar:
--------------------------------
    iris.csv:
        adjusted_random_score:  0.7436129323988243
        v_measure:              0.7448181781370871
        computed accuracy:      0.8929

    digits.csv:
        adjusted_random_score:  0.9580568857733347
        v_measure:              0.9627105959817531
        computed accuracy:      0.9802

        The results are different using the standard scalar instead min max
        scaler. I expect thta the lower score in the iris.csv data is due to the
        variance due to the low amount of test data. The digits data accuracy does
        not change to much likely due to the larger size of the data set. Something
        that suprised me was that the way the Principle Component Analysis worked,
        it significantly changed the way in which the dimensions were squashed.
        Therefore, the representations looked significantly different from the
        previous. The scales on the sides of the graphs were different, but this
        makes sense as we standardize the data in the standard scalar. I think the
        reason that using the standard scalar would make more sense is because
        the likeliness of a point being part of a certain cluster does not vary
        linearly with its displacement from a point. Rather realy obsurd components,
        lead have really unlikely probabilities and these distrobution probably
        look somewhat Gaussian. While we recieved lower accuracies this is not
        indicitive that one scalar is better than the other as these were only two
        datasets with one test each.

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from progress.bar import Bar
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

##########################################
def read_csv_data(filename, num_features, labels_to_remove = [], remap_labels = False):
    """This function reads a csv file into a pandas dataframe object, cleans it,
    and then returns two numpy arrays, one of the features and another of the labels.
    This function assumes that the first row of the csv file is ths contains the
    headers. It also accumes that the first columns contain the features, followed
    by the labels column. It therefore deletes any columns one past the number of
    features specified. It also deletes any rows/observations that contain no values.
    It removes any rows that contain the labels in the optional parameter
    labels_to_remove. If remap_labels is True then the function remaps the labels
    to integer vaules. The file referenced must be in the working directory.

    Parameters:
    -----------
        filename : str
            the csv filename
        num_features : int
            the number of features in each object
        labels_to_remove : list
            a list of the names of labels(row labels) to remove
        remap_labels : Bool
            a Boolean that if true means we need to remap the labels column to
            integers

    Returns:
    -------
        list
            a two-element list that contains first a numpy array of the features
            and then a numpy array of the labels
    """

    df = pd.read_csv(filename, header = 0)

    # grabbing the columns past (including) num_features + 1
    cols_to_remove = []
    for i in range(len(df.columns)):
        if i > num_features:
            cols_to_remove.append(df.columns[i])

    #removing the columns
    df_clean_cols = df.drop(columns = cols_to_remove)

    #droping rows with NA value
    df_cleaner = df_clean_cols.dropna()

    # dropping row whose names in the labels/taget col(numfeatures + 1)
    # is in labels_to_remove
    labels_col = df_cleaner.columns[-1]
    for label in labels_to_remove:
        df_cleaner = df_cleaner.loc[df_cleaner[labels_col]!= label]
    # making label values integers if necessary
    if remap_labels == True:
        labels = df_cleaner[labels_col]
        # gets unique labels
        labels_unique = labels.unique()
        labels_dict  = {labels_unique[i] : i for i in range(len(labels_unique))}
        # using that maps x to label_dict[x]
        y = labels.apply(lambda x : labels_dict[x])
    else:
        y = df_cleaner[labels_col]
    y = y.to_numpy()
    #all rows & all columns but labels column
    X = df_cleaner.drop(columns = labels_col)
    X = X.to_numpy()
    #returns a list of numpy arrays
    return [X, y]


def split_data(X, y, percent = .20, seed = None):
    """This function splits the passed in data into test and training sets. If
    there is a given seed, the data is scrabled according to that seed. If a
    percent is given it created the test size to be that proportion of the data
    (automatically 20%).

    Parameters:
    -----------
        X : numpy array
            a numpy array containing all of the features
        y : numpy array
            a numpy array containing all of the labels
        percent : float
            a floating point containing the proportion of data to be used as a
            test data, defaulting to .20
        seed : int
            an integer (between 0-999999) corresponding to a seed for
            randomization, defaulting to None if no seed is given

    Returns:
    -------
        list
            a four-element list that contains first a numpy array of the features
            of the training data, then a numpy array of the labels of the training
            data, then a numpy array of the features of the test data, and lastly
            a numpy array of the labels of the test data
    """

    #setting seed and scrambling data
    if seed != None:
        np.random.seed(seed)
        indices = np.random.permutation(len(y))
        X = X[indices]
        y = y[indices]

    NUM_ROWS = X.shape[0]     # the number of labeled rows
    TEST_SIZE = int(percent*NUM_ROWS)   # no harm in rounding down

    X_test = X[:TEST_SIZE]
    X_train = X[TEST_SIZE:]
    y_test = y[:TEST_SIZE]
    y_train = y[TEST_SIZE:]

    return [X_train, y_train, X_test, y_test]


def find_best_k(X_train, y_train, max_k):
    """This function finds the best k value given the training data and the
    maximum k value to find. It also print the best k value used and uses a
    progress bar.

    Parameters:
    -----------
        X_train : numpy array
            a numpy array containing the features of the training data
        y : numpy array
            a numpy array containing the labels of the training data
        max_k : int
            an integer coressponding to the maximum number of k to test

    Returns:
    -------
        int
            an integer corresponding to the best k found
    """

    #initializing
    best_accuracy = 0
    best_k = None

    bar = Bar("Calculating best k value:", max = max_k)
    for i in range(1, max_k + 1):
        k = i
        knn_cv_model = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn_cv_model, X_train, y_train, cv = 5)
        average_cv_accuracy = cv_scores.mean()
        if average_cv_accuracy > best_accuracy:
            best_accuracy = average_cv_accuracy
            best_k = k
        bar.next()

    print(f"\nThe best k was: {best_k}")

    return best_k


def k_nearest_neighbors_prediction(X, y, max_k, seed = None):
    """This function scales the features, splits the data into test and training
    sets, scrambling it if a seed is given, finds the best k value given the
    maximum k value to test, and then creates predicted labels for the test
    features, returning it in a list with the actual labels.

    Parameters:
    -----------
        X : numpy array
            a numpy array containing all the features
        y : numpy array
            a numpy array containing all the labels
        max_k : int
            an integer coressponding to the maximum number of k to test
        seed : int
            an integer (between 0-999999) corresponding to a seed for
            randomization, defaulting to None if no seed is given

    Returns:
    -------
        list
            a list containing first the predicted labels of the test set and then
            the actual labels of the test data
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    split_list = split_data(X_scaled, y, seed = seed)
    X_train = split_list[0]
    y_train = split_list[1]
    X_test = split_list[2]
    y_test = split_list[3]

    best_k = find_best_k(X_train, y_train, max_k)

    knn_model = KNeighborsClassifier(n_neighbors = best_k)
    knn_model.fit(X_train, y_train)
    y_predicted= knn_model.predict(X_test)

    return [y_predicted, y_test]



def visualize(X, y, num_clusters, seed = None):
    """This nonfruitful function first scales the features. It then uses principle
    component analysis to put the data into both 3 and 2 dimensions. Using k-means
    clustering, passing in the seed for the original random stata of the centroid
    and the number of clusters, the function creates predicted labels. It then
    graphs the cluster centers as Xs and color codes the estimated labels. It does
    this for both the two dimensional and three dimensional vizualizations. It also
    could have been possible to show which predicted labels were incorrect;
    however the question did not ask for this.

    Parameters:
    -----------
        X : numpy array
            a numpy array containing all the features
        y : numpy array
            a numpy array containing all the labels
        num_clusters : int
            an integer corresponding to the number of clusters to use in our k-means
            model
        seed : int
            an integer (between 0-999999) corresponding to a seed for
            randomization, defaulting to None if no seed is given

    Returns:
    -------
        None
    """

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components = 3)
    X_pca = pca.fit_transform(X_scaled)

    k_means = KMeans(init = 'k-means++', random_state = seed,
                     n_clusters = num_clusters, n_init = 5)
    y_km = k_means.fit_predict(X_pca)
    centroids = k_means.cluster_centers_

    cmap = plt.get_cmap('tab10')
    colors = [cmap(i) for i in y_km]  # color each predicted label

    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(111, projection = "3d")  # 111: 1 row, 1 col, 1st pos
    ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], c = colors)
    ax.scatter(centroids[:,0], centroids[:,1], centroids[:,2],
               c = "gold", marker = "X", edgecolor = "black", s = 250)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")
    plt.title("A vizualizations of k-means clustering using 3 components")
    plt.show()  

    #repeating for 2D
    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X_scaled)

    y_km = k_means.fit_predict(X_pca)
    centroids = k_means.cluster_centers_

    fig = plt.figure(figsize = (6,6))
    ax = fig.add_subplot(111) # 111: 1 row, 1 col, 1st pos
    ax.scatter(X_pca[:,0], X_pca[:,1], c = colors)
    ax.scatter(centroids[:,0], centroids[:,1],
               c = "gold", marker = "X", edgecolor = "black", s = 250)
    plt.title("A vizualizations of k-means clustering using 2 components")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.show()

def compare_labels(y_test, y_predicted):
    """This function compares the labels in y_test and y_predicted, returning the
    accuracy of the prediction rounded to 4 digits. It is assumed that all the
    labels have been mapped to integers.

    Parameters:
    -----------
        y_test : numpy array
            a numpy array containing the actual labels
        y_predicted : numpy array
            a numpy array containing the predicted labels

    Returns:
    -------
        float
            the computed accuracy of the prediction rounded to 4 digits
    """

    num = len(y_predicted)
    num_correct = 0 #accumulator

    for i in range(num):
        p = int(y_predicted[i])
        a = int(y_test[i])
        if p == a:
            num_correct += 1

    return round(num_correct / num, 4)


def print_scores(y_test, y_predicted):
    """This function prints the three different scorings for the given y_test
    and y_predicted. The first is the adjusted random score, follow by the v
    measure score, followed by the computed accuracy score from the function
    above.

    Parameters:
    -----------
        y_test : numpy array
            a numpy array containing the actual labels
        y_predicted : numpy array
            a numpy array containing the predicted labels

    Returns:
    -------
        float
            the computed accuracy of the prediction rounded to 4 digits
    """

    adjusted_random_score = metrics.adjusted_rand_score(y_test, y_predicted)
    v_measure_score = metrics.v_measure_score(y_test, y_predicted)
    computed_accuracy = compare_labels(y_test, y_predicted)

    print(f"adjusted_random_score:  {round(adjusted_random_score, 3)}")
    print(f"v_measure:              {round(v_measure_score, 3)}")
    print(f"computed accuracy:      {round(computed_accuracy, 3)}")


###########
def main():
    ########################## Testing iris.csv ###############################
    [X, y] = read_csv_data("iris.csv", 4, ["alieniris"], True)
    """
    print(X)
    print(y)
    print(len(X))
    print(len(y))
    """

    #testing without mixing the data with a seed
    [X_train, y_train, X_test, y_test] = split_data(X, y)
    """
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    print("*"*70)
    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))
    """

    #testing with a seed and percent
    [X_train, y_train, X_test, y_test] = split_data(X, y, percent = .10, seed = 678678)
    """
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    print("*"*70)
    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))
    """

    best_k = find_best_k(X_train, y_train, 50)

    [y_predicted, y_test] = k_nearest_neighbors_prediction(X, y, 50, 5551212)
    #print(y_predicted)
    #print(y_test)

    #for this data set:
    num_clusters = 3
    print()
    print("Visualizing the iris k-means clustering results. "
          "Each type of flower is colored a different color")
    visualize(X, y, num_clusters, seed = 5551212)
    print()

    print_scores(y_test, y_predicted)
    print()

    ########################## Testing digits.csv ###############################
    [X, y] = read_csv_data("digits.csv", 64)
    """
    print(X)
    print(y)
    print(len(X))
    print(len(y))
    """

    #testing without mixing the data with a seed
    [X_train, y_train, X_test, y_test] = split_data(X, y)
    """
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    print("*"*70)
    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))
    """

    #testing with a seed and percent
    [X_train, y_train, X_test, y_test] = split_data(X, y, percent = .10, seed = 678678)
    """
    print(X_train)
    print(y_train)
    print(X_test)
    print(y_test)
    print("*"*70)
    print(len(X_train))
    print(len(y_train))
    print(len(X_test))
    print(len(y_test))
    """

    best_k = find_best_k(X_train, y_train, 50)

    [y_predicted, y_test] = k_nearest_neighbors_prediction(X, y, 50, 5551212)
    #print(y_predicted)
    #print(y_test)

    #for this data set:
    num_clusters = 10
    print()
    print("Visualizing the digits k-means clustering results. "
          "Each digit is colored a different color")
    visualize(X, y, num_clusters, seed = 5551212)
    print()
    print_scores(y_test, y_predicted)


    """
    try:

        read_csv_data("", 0, [], True)
        split_data(None, None, 0.2, 5551212)
        find_best_k(None, None, 100)
        k_nearest_neighbors_prediction(None, None, 100, 5551212)
        visualize(None, None, 3, 5551212)
        print_scores(None, None)


    except Exception as error:
        print(f"Did not correctly implement function:\n\t{error}")
    """

if __name__ == "__main__":
    main()
