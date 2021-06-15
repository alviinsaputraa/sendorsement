from django.shortcuts import render
import pandas as pd
import json

# Create your views here.
def dash(request):
    dfLabel = pd.read_excel('labelling.xls')
    df2 = dfLabel['tweet'].values.tolist()
    df1 = dfLabel['label'].values.tolist()
    dfNet = df1[0]
    dfPos = df1[1]
    dfNeg = df1[2]
    return render(request, 'dash.html', {'df': json.dumps(df2), 'label': json.dumps(df1),
                                         'dfNet': dfNet, 'dfNeg': dfNeg, 'dfPos': dfPos})