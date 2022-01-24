import pandas as pd
import subprocess
from st_aggrid.grid_options_builder import GridOptionsBuilder
from sklearn.metrics import accuracy_score, precision_score, recall_score

def computeAPR(data, predLabel):
        confMatrix = pd.crosstab(data['True Sentiment'], data[predLabel])
        accuracy = accuracy_score(data['True Sentiment'], data[predLabel])
        precision = precision_score(data['True Sentiment'], data[predLabel], average='weighted')
        recall = recall_score(data['True Sentiment'], data[predLabel], average='weighted')
        return accuracy, precision, recall
# to get environment variables
def getEnvironVar(varname):
    CMD = 'echo $(source myscript.sh; echo $%s)' % varname
    p = subprocess.Popen(CMD, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    return (p.stdout.readlines()[0].strip())

def buildGridOptionAgGrid(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    # gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gridOptions = gb.build()
    return gridOptions
