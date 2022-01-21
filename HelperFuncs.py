import pandas as pd
import subprocess


def generateConfusionMatrix(data):
        confMatrix = pd.crosstab(data['True Sentiment'], data['Predicted Sentiment'])

# to get environment variables
def getEnvironVar(varname):
    CMD = 'echo $(source myscript.sh; echo $%s)' % varname
    p = subprocess.Popen(CMD, stdout=subprocess.PIPE, shell=True, executable='/bin/bash')
    return (p.stdout.readlines()[0].strip())


print(getEnvironVar('OPENAI_API_KEY'))