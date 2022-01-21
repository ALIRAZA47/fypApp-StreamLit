import pandas as pd
import subprocess


def generateConfusionMatrix(data):
        confMatrix = pd.crosstab(data['True Sentiment'], data['Predicted Sentiment'])

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
