import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def histograms_plot(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        dataframe[feature].hist(bins=20,ax=ax)
        ax.set_title(feature+" Distribution")

    fig.tight_layout()  
    plt.show()
    
def histograms_log_plot(dataframe, features, rows, cols):
    fig=plt.figure(figsize=(20,20))
    for i, feature in enumerate(features):
        ax=fig.add_subplot(rows,cols,i+1)
        try:
            np.log(dataframe[feature]).hist(bins=50,ax=ax)
        except:
            pass
        ax.set_title(feature+" Distribution")

    fig.tight_layout()  
    plt.show()



def get_plot_facies (data, lith):
    lithofacies = {30000: 'Sandstone', 65030: 'Sandstone/Shale',
               65000: 'Shale', 80000: 'Marl', 74000: 'Dolomite', 70000: 'Limestone', 70032: 'Chalk', 88000: 'Halite',
               86000: 'Anhydrite', 99000: 'Tuff', 90000: 'Coal', 93000: 'Basement'}

    data['Lith']= data["FORCE_2020_LITHOFACIES_LITHOLOGY"].map(lithofacies)

    facies = data[data["Lith"] == lith]
     
    col_list = facies.columns.values[7:-4]
    

    nan = facies[col_list].isna().mean().sort_values() *100

    plt.figure(figsize=(12,4))
    splot = sns.barplot(x=nan.index,y=nan)
    splot.set_title("Percentage of null" + ' ' + str(lith), fontsize=20)
    plt.axhline(y=30, color='red', lw=2, linestyle='--',label="missing data Threshold")
    splot.set_ylabel('Percentage (%) Missing Value', fontsize=12)
    splot.set_xlabel('')
    
    plt.show()
    
    return facies


def use_logs(data, *args):

    columns = []
    for _ in args:
        columns.append(_)
        
    data = data.loc[:, (columns)]
    #data.fillna(data.mean(), inplace=True)
    data.fillna(-999, inplace=True)
        
    return data