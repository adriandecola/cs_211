"""
Course:        DCS 211 Winter 2021 Module D
Assignment:    Project 4a
Topic:         k-NN Machine Learning
Purpose:       Implementing a functioning k-NN classifier for the provided
               digits.csv data.

Student Name: Adrian deCola

"""

import numpy as np      # numpy is Python's "array" library
import pandas as pd     # Pandas is Python's "data" library ("dataframe" == spreadsheet)
import seaborn as sns   # yay for Seaborn plots!
import matplotlib.pyplot as plt
import random
from progress.bar import Bar
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score

#################################################
def drawDigitHeatmap(pixels, showNumbers = True):
    '''
    Draws a heat map of a given digit based on its 8x8 set of pixel values.

    Parameters:
    -----------
        pixels : list
            a 2D list (8x8) of integers of the pixel values for the digit
        showNumbers : bool
            if True, shows the pixel value inside each square

    Returns:
    -------
        None
    '''

    (fig, axes) = plt.subplots(figsize = (4.5, 3))  # aspect ratio

    rgb = (0, 0, 0.5)  # each in (0,1), so darkest will be dark blue
    colormap = sns.light_palette(rgb, as_cmap=True)
    # all seaborn palettes: medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f

    # plot the heatmap;  see: https://seaborn.pydata.org/generated/seaborn.heatmap.html
    # (fmt = "d" indicates to show annotation with integer format)
    sns.heatmap(pixels, annot = showNumbers, fmt = "d", linewidths = 0.5, \
                ax = axes, cmap = colormap)
    plt.show(block = True)

##############################
def fetchDigit(df, which_row):
    '''
    For digits.csv data represented as a dataframe, this fetches the digit from
    the corresponding row, reshapes, and returns a tuple of the digit and a
    numpy array of its pixel values.

    Parameters:
    -----------
        df : pandas data frame
            expected to be obtained via pd.read_csv() on digits.csv
        which_row : int
            an integer in 0 to len(df)

    Returns:
    -------
        tuple
            returns a tuple containing the reprsented digit and a numpy array
            of the pixel values
    '''
    digit  = int(round(df.iloc[which_row, 64]))
    pixels = df.iloc[which_row, 0:64]   # dont want the rightmost rows
    pixels = pixels.values              # converts to numpy array
    pixels = pixels.astype(int)         # convert to integers for plotting
    pixels = np.reshape(pixels, (8,8))  # makes 8x8
    return (digit, pixels)

def cleanTheData(df):

    col65Name = df.columns[65]
    df_clean = df.drop(columns = [col65Name])

    array = df_clean.values
    array = array.astype('float64')
    return array

def predictive_model(features, array):
    #array doesnt include the actual label
    dist = np.linalg.norm

    num_vectors = len(array)
    closest_vector = array[0, 0:len(array[0])-1]
    closest_num = array[0, len(array[0])-1]
    closest_distance = dist(features-closest_vector)

    for i in range(num_vectors):
        current_vector = array[i, 0:len(array[i])-1]
        current_num = array[i, len(array[i])-1]
        current_distance = dist(features-current_vector)

        if current_distance < closest_distance:
            closest_distance = current_distance
            closest_num = current_num
    return closest_num

def splitData(allData):
    x_all = allData[:, 0:len(allData[0])-1]
    y_all = allData[:, len(allData[0])-1]

    NUM_ROWS = x_all.shape[0]     # the number of labeled rows
    TRAIN_PERCENT = 0.80
    TRAIN_SIZE = int(TRAIN_PERCENT*NUM_ROWS)   # no harm in rounding down

    train = allData[:TRAIN_SIZE]
    test = allData[TRAIN_SIZE:]

    x_train = train[:, 0:len(train[0])-1]
    y_train = train[:, len(train[0])-1]

    x_test = test[:, 0:len(test[0])-1]
    y_test = test[:, len(test[0])-1]

    return [x_test, y_test, x_train, y_train]

def compare_labels(predicted_digs, actual_digs):
    num_digs = len(predicted_digs)
    num_correct = 0 #accumulator

    for i in range(num_digs):
        p = int(predicted_digs[i])
        a = int(actual_digs[i])
        result = "incorrect"
        if p == a:
            result = ""
            num_correct += 1

        #print(f"row {i}  :  Actual = {a}, Predicted = {p}   {result}")

    print()
    print(f"The model correctly predicted {num_correct} digits out of {num_digs}.")
    print(f"That's an accuracy of {round(num_correct / num_digs, 3) * 100}%.")

def best_k(data):
    #this function returns the lowest k with the highest accuracy from all three
    #seeds
    best_accuracy = 0
    best_k = None

    org_data = data
    x_all = data[:, 0:len(data[0])-1]
    y_all = data[:, len(data[0])-1]

    np.random.seed(8675309)
    indices = np.random.permutation(len(y_all))
    x_labeled = x_all[indices]
    y_labeled = y_all[indices]


    for i in range(1, 50):
        k = i
        knn_cv_model = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn_cv_model, x_labeled, y_labeled, cv = 5)
        average_cv_accuracy = cv_scores.mean()
        print(f"k = {k}, accuracy = {average_cv_accuracy}")
        if average_cv_accuracy > best_accuracy:
            best_accuracy = average_cv_accuracy
            best_k = k

    x_all = org_data[:, 0:len(data[0])-1]
    y_all = org_data[:, len(data[0])-1]

    np.random.seed(5551212)
    indices = np.random.permutation(len(y_all))
    x_labeled = x_all[indices]
    y_labeled = y_all[indices]


    for i in range(1, 50):
        k = i
        knn_cv_model = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn_cv_model, x_labeled, y_labeled, cv = 5)
        average_cv_accuracy = cv_scores.mean()
        print(f"k = {k}, accuracy = {average_cv_accuracy}")
        if average_cv_accuracy > best_accuracy:
            best_accuracy = average_cv_accuracy
            best_k = k

    x_all = org_data[:, 0:len(data[0])-1]
    y_all = org_data[:, len(data[0])-1]

    np.random.seed(5921915)
    indices = np.random.permutation(len(y_all))
    x_labeled = x_all[indices]
    y_labeled = y_all[indices]
    #for all of these k= 1 is the best value.


    for i in range(1, 50):
        k = i
        knn_cv_model = KNeighborsClassifier(n_neighbors=k)
        cv_scores = cross_val_score(knn_cv_model, x_labeled, y_labeled, cv = 5)
        average_cv_accuracy = cv_scores.mean()
        print(f"k = {k}, accuracy = {average_cv_accuracy}")
        if average_cv_accuracy > best_accuracy:
            best_accuracy = average_cv_accuracy
            best_k = k

    return best_k

def train_test(df):
    df = cleanTheData(df)
    split_list = splitData(df)
    k = 1 #we deteremined above
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(split_list[2], split_list[3])
    predicted_digs = knn_model.predict(split_list[0])
    actual_digs = split_list[1]
    #prints
    compare_labels(predicted_digs, actual_digs)

###########
def main():

    # for read_csv, use header=0 when row 0 is a header row
    filename = 'digits.csv'
    df = pd.read_csv(filename, header = 0)



    """
    print(df.head())
    print(f"{filename} : file read into a pandas dataframe.")

    num_to_draw = 5
    for i in range(num_to_draw):
        # let's grab one row of the df at random, extract/shape the digit to be
        # 8x8, and then draw a heatmap of that digit
        random_row = random.randint(0, len(df) - 1)
        (digit, pixels) = fetchDigit(df, random_row)
        print(f"The digit is {digit}")
        print(f"The pixels are\n{pixels}")
        drawDigitHeatmap(pixels)
        plt.show()
    """
    #####################################################
    #
    # OK!  Onward to knn for digits! (based on your iris work...)
    #


    data = cleanTheData(df)

    train_test(df)

    """

    # Testing predictive_model
    feature = [0,0,12,8,8,7,0,0,0,3,16,16,11,7,0,0,0,2,14,1,0,0,0,0,0,5,14,5,0,0,0,0,0,2,15,16,9,0,0,0,0,0,0,2,16,2,0,0,0,0,4,8,16,4,0,0,0,0,11,14,9,0,0,0]
    print(predictive_model(feature, data))

    ########################## Question 3 ##########################
    x_all = data[:, 0:len(data[0])-1]
    y_all = data[:, len(data[0])-1]

    NUM_ROWS = x_all.shape[0]     # the number of labeled rows
    TRAIN_PERCENT = 0.80
    TRAIN_SIZE = int(TRAIN_PERCENT*NUM_ROWS)   # no harm in rounding down

    train = data[:TRAIN_SIZE]
    test = data[TRAIN_SIZE:]

    x_test = test[:, 0:len(test[0])-1]
    y_test = test[:, len(test[0])-1]

    # for loop
    correct_digits = 0

    bar = Bar("Predicting numbers...", max = len(x_test))
    for i in range(len(x_test)):
        # x_test[i] is our "features"
        predicted_dig = predictive_model(x_test[i], train)
        if int(predicted_dig) == int(y_test[i]):
            correct_digits += 1
        bar.next()
    bar.finish()
    accuracy = round(correct_digits / len(x_test), 3)
    print(f"The accuracy on our test data was {accuracy*100}%.")

    ########################## Question 4 ##########################

    x_all = data[:, 0:len(data[0])-1]
    y_all = data[:, len(data[0])-1]

    NUM_ROWS = x_all.shape[0]     # the number of labeled rows
    TEST_PERCENT = 0.2
    TEST_SIZE = int(TEST_PERCENT*NUM_ROWS)   # no harm in rounding down

    train = data[TEST_SIZE:]
    test = data[:TEST_SIZE]

    x_test = test[:, 0:len(test[0])-1]
    y_test = test[:, len(test[0])-1]

    # for loop
    correct_digits = 0

    bar = Bar("Predicting numbers...", max = len(x_test))
    for i in range(len(x_test)):
        # x_test[i] is our "features"
        predicted_dig = predictive_model(x_test[i], train)
        if int(predicted_dig) == int(y_test[i]):
            correct_digits += 1
        bar.next()
    bar.finish()
    ############################# WHY ISNT ROUND WORKING???????????????
    accuracy = round(correct_digits / len(x_test), 3)
    print(f"The accuracy on our test data was {accuracy*100}%.")

    ########################## Question 5 ##########################
    wrong_digits = 0
    for i in range(len(x_test)):
        # x_test[i] is our "features"
        predicted_dig = predictive_model(x_test[i], train)
        if int(predicted_dig) != int(y_test[i]):
            pixels = x_test[i].astype(int)         # convert to integers for plotting
            pixels = np.reshape(pixels, (8,8))
            print(f"This digit was a {y_test[i]}.")
            print(f"It was guessed to be a {predicted_dig}.")
            drawDigitHeatmap(pixels)
            wrong_digits += 1
        if wrong_digits == 5:
            break

    # these were hard for even us to tell so it makes sense that our model might fail
    ########################## Question 6 ##########################
    split_list = splitData(data)
    #print(split_list)
    #explain split_list***********
    #checked first pixel values and digit values and it makes sense
    ########################## Question 7 ##########################
    k = 60
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(split_list[2], split_list[3])
    predicted_digs = knn_model.predict(split_list[0])
    actual_digs = split_list[1]
    compare_labels(predicted_digs, actual_digs)

    # I just wanted to try a much larger k**********
    print(best_k(data))
    """

if __name__ == "__main__":
    main()
