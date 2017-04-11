from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt




#%% 

if __name__ == '__main__':

     # Type of the data to read ("stride", "dssp")
    data_type = "stride"
    # File name
    file_name = "../data/pred_result_" + data_type + ".csv"
    # Read the prediction data
    pred_data = pd.read_csv(file_name)
    
    # Take out the proteins with "other"
    # pred_data = pred_data.drop(pred_data[pred_data["overall_structure_real"].isin(["None"])].index).reset_index(drop=True)

    
    features = [1] + list(range(4, 7)) + list(range(10, 30)) + list(range(31, 36))
    x_train = pred_data.iloc[:, features]
    y_train = pred_data.overall_structure_real
    
   
    
    #%% Training

    clf = RandomForestClassifier(n_estimators=200, n_jobs = 2, criterion="entropy", bootstrap=False, min_samples_split=2, min_samples_leaf=2, max_depth=9, max_features=10) 
    
#    param_grid = {
#                     'n_estimators': [15, 100, 200, 300]
#                 }
#    
#    grid_clf = GridSearchCV(clf, param_grid, cv=6, verbose=2, n_jobs=10)
#    grid_clf.fit(x_train, y_train)
#    
#    print("\n-------- BEST ESTIMATOR --------\n")
#    print(grid_clf.best_estimator_)
#    print("\n-------- BEST PARAMS --------\n")
#    print(grid_clf.best_params_)
#    print("\n-------- BEST SCORE --------\n")
#    print(grid_clf.best_score_)


            
    clf.fit(x_train, y_train)
    
    # Feature importance plot of Random Forest;
    # see http://scikit-learn.org/stable/auto_examples/ensemble/plot_forest_importances.html
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    # Print the feature ranking
    print("Feature ranking:")
    
    for f in range(x_train.shape[1]):
        print("%d. feature %d (%f):" % (f + 1, indices[f], importances[indices[f]]), x_train.columns[indices[f]])
    
    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(x_train.shape[1]), importances[indices],
           color="#5A9ACC", yerr=std[indices], align="center")
    plt.xticks(range(x_train.shape[1]), x_train.columns[indices], rotation='vertical')
    plt.xlim([-1, x_train.shape[1]])
    plt.show()
    
    
    #%% Prediction
    pred_data.overall_pred = clf.predict(x_train)
    
    # Accuracy on the test set. It's very high, but it doesn't matter much.
    # What we want is to have a general method to classify, which we have, with accuracy > 80%
    print(sum(pred_data.overall_pred == pred_data.overall_structure_real) / pred_data.shape[0])
    # Which predictions were wrong?
    wrong_pred = pred_data.loc[pred_data.overall_pred != pred_data.overall_structure_real]
    
    #%% Save the predictions
    pred_data.to_csv("../data/pred_result_" + data_type + ".csv", index=False)
    