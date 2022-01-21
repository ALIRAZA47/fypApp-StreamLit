import pandas as pd
from st_aggrid.grid_options_builder import GridOptionsBuilder


def generateConfusionMatrix(data):
        confMatrix = pd.crosstab(data['True Sentiment'], data['Predicted Sentiment'])


def buildGridOptionAgGrid(df):
    gb = GridOptionsBuilder.from_dataframe(df)
    # gb.configure_pagination()
    gb.configure_side_bar()
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, aggFunc="sum", editable=True)
    gridOptions = gb.build()
    return gridOptions