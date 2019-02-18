from sklearn.model_selection import train_test_split

def split_data_to_sets(data=[], label=[], test_ratio = 0.33, random_state = 42):
    # TODO! If you want random samples, but now we use a fixed seed
    X_train, X_test, y_train, y_test = train_test_split(data, label,
                                                        test_size = test_ratio,
                                                        random_state = random_state)
    return(X_train, y_train, X_test, y_test)

def get_CV_folds(X_train, y_train, k = 10):
    return(X_train, y_train, [], [])

def feature_selector(data=[], label=[]):
    # Placeholder if you want to remove some of the non-explaining features
    # with your method of choice
    return(data)
