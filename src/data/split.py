def split_data(X, y):
    # time-based split
    train_mask = X["valid_time"] < "2016-01-01"
    val_mask = (X["valid_time"] >= "2016-01-01") & (X["valid_time"] < "2019-01-01")
    test_mask = X["valid_time"] >= "2019-01-01"

    X_train, y_train = (
        X[train_mask].drop(
            columns=["valid_time", "drought_class", "latitude", "longitude"]
        ),
        y[train_mask],
    )
    X_val, y_val = (
        X[val_mask].drop(
            columns=["valid_time", "drought_class", "latitude", "longitude"]
        ),
        y[val_mask],
    )
    X_test, y_test = (
        X[test_mask].drop(
            columns=["valid_time", "drought_class", "latitude", "longitude"]
        ),
        y[test_mask],
    )

    # y_train = y_train.replace("moderate", "normal").replace("watch", "normal")
    # y_val = y_val.replace("moderate", "normal").replace("watch", "normal")
    # y_test = y_test.replace("moderate", "normal").replace("watch", "normal")

    return (X_train, y_train, X_val, y_val, X_test, y_test)
