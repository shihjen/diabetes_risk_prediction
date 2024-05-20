# function to plot histogram for displaying distribution of each variable in the dataset
def plot_histogram(var, data):
    import plotly.express as px
    for k,v in var.items():
        fig = px.histogram(data, x=k, height=600, nbins=50, title=f'Distribution of {v}')
        fig.update_layout(xaxis_title=v, yaxis_title='Frequency', margin=dict(l=50, r=50, t=50, b=50))
        fig.update_layout(template='plotly_dark', xaxis=dict(showticklabels=False))
        fig.update_traces(hovertemplate=f'{v}: %{{x}}<br>Frequency: %{{y}}')
        fig.show()
        
# function to compare distibution of each variable between diabetic and non-diabetic groups         
def compare_groups(var, data):
    import matplotlib.pyplot as plt
    import seaborn as sns
    colors = ['#FCA636FF','#C0369D']
    labels = ['Without Diabetets','Diabetes']
    # visualization
    for k,v in var.items():
        figure, axes = plt.subplots(1,2, figsize=(15,5), gridspec_kw={'width_ratios':[1.5,1]})
        # Set figure and axes background colors
        # figure.patch.set_facecolor('black')
        # axes[0].patch.set_facecolor('black')
        # axes[1].patch.set_facecolor('black')
        sns.histplot(data=data, x=data[k], hue=data['Outcome'], kde=True, ax=axes[0], palette=colors)
        axes[0].set_xlabel('Frequency')
        axes[0].legend(labels=['Diabetes','Without Diabetes'])
        sns.boxplot(data=data, x=data['Outcome'], y=data[k], ax=axes[1], palette=colors)
        axes[1].set_xticks([0,1],labels)
        axes[1].set_xlabel(' ')
        #axes[1].grid(alpha=0.4)
        figure.suptitle(f'Distribution of {v} between Patients with and without Diabetes', fontsize=15)
        plt.tight_layout(pad=1)
        plt.show()

# function to plot scatter plot
def plot_scatter(x,y):
    import plotly.express as px
    fig = px.scatter(train, x=x, y=y, color='Outcome', height=600)
    fig.update_layout(template='plotly_dark')
    fig.show()

# function to perform student t-test
def ttest(var, data):
    from scipy.stats import ttest_ind
    data_copy = data[~data[var].isna()]
    group_data = data_copy.groupby('Outcome')
    diabetes = group_data.get_group(1)
    no_diabetes = group_data.get_group(0)
    res = ttest_ind(diabetes[var], no_diabetes[var])
    test_stat = round(res[0], 4)
    pvalue = round(res[1], 4)
    return test_stat, pvalue


# helper function to perform hyperparameters tuning via grid search and return the tuned model
def hyperparameterTuning(model, param_grid, Xtrain, ytrain):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    stratifiedCV = StratifiedKFold(n_splits=3)
    # create the GridSearchCV object
    grid_search = GridSearchCV(model,                  # model to be tuned
                               param_grid,             # search grid for the parameters
                               cv=stratifiedCV,        # stratified K-fold cross validation to evaluate the model performance
                               scoring='roc_auc',      # metric to assess the model performance, weighted F1 score (consider the proportion of classes in the dataset)
                               n_jobs=-1)              # use all cpu cores to speed-up CV search

    # fit the data into the grid search space
    grid_search.fit(Xtrain, ytrain)

    # print the best parameters and the corresponding ROC_AUC score
    print('Best Hyperparameters from Grid Search : ', grid_search.best_params_)
    print('Best AUROC Score: ', grid_search.best_score_)
    print()

    # get the best model
    best_model = grid_search.best_estimator_
    
    # return the hyperparameters tuned model
    return best_model

# function to get the performance of all trained models
def model_performance(models, X, y):
    from sklearn.metrics import precision_score, recall_score, f1_score, matthews_corrcoef
    import pandas as pd
    precision_list = []
    recall_list = []
    f1_list = []
    mcc_list = []
    cont = []
    for model_name, model in models.items():
        ypred = model.predict(X)
        precision = precision_score(y, ypred, average='macro')
        recall = recall_score(y, ypred, average='macro')
        f1 = f1_score(y, ypred, average='macro')
        mcc = matthews_corrcoef(y, ypred)
        res = [model_name, precision, recall, f1, mcc]
        cont.append(res)
    res_df = pd.DataFrame(cont, columns=['Model','Precision','Recall','F1','MCC'])
    return res_df

# function to plot confusion matrix
def plot_confusion_matrix(model, Xtrain, ytrain, Xtest, ytest):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    label = ['Non-diabetic', 'Diabetic']

    ypred_train = model.predict(Xtrain)
    ypred_test = model.predict(Xtest)

    cm_train = confusion_matrix(ytrain, ypred_train)
    cm_test = confusion_matrix(ytest, ypred_test)
    
    figure, axes = plt.subplots(1,2, figsize=(12,5))
    sns.heatmap(cm_train, annot=True, cmap='plasma', cbar=False, ax=axes[0], xticklabels=label, yticklabels=label, fmt='d')
    axes[0].set_title('Training Data')
    sns.heatmap(cm_test, annot=True, cmap='plasma', cbar=False, ax=axes[1], xticklabels=label, yticklabels=label, fmt='d')
    axes[1].set_title('Test Data')
    plt.tight_layout()
    plt.show()