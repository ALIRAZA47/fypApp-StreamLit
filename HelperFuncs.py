import pandas as pd
import subprocess
from st_aggrid.grid_options_builder import GridOptionsBuilder


def computeAccuracy(data, predLabel):
        confMatrix = pd.crosstab(data['True Sentiment'], data[predLabel])
        accuracy = (confMatrix.iloc[0,0] + confMatrix.iloc[1,1] + confMatrix.iloc[2,2]) \
        / (confMatrix.iloc[0,0] + confMatrix.iloc[1,1] + confMatrix.iloc[0,1] + confMatrix.iloc[1,0] + \
        confMatrix.iloc[2,2] + confMatrix.iloc[2,0] + confMatrix.iloc[0,2])
        
        return accuracy
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
