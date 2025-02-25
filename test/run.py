

import pandas as pd
import numpy as np
import json
import os

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns',None)

from pyecharts import options as opts
from pyecharts.charts import *
from pyecharts.options.global_options import ThemeType
from pyecharts.commons.utils import JsCode
from pyecharts.globals import CurrentConfig,NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
from pyecharts.globals import ThemeType
from jinja2 import Markup, Environment, FileSystemLoader

from utils.visualization import *

CurrentConfig.GLOBAL_ENV=Environment(loader=FileSystemLoader('./templates')) #注意要把loader加上


from flask import render_template, url_for
import flask

dataset=pd.read_csv('data/dataset.csv')



pie_gender = demo_pie('Gender Distribution', list(dataset.gender.value_counts().index), [int(value) for value in dataset.gender.value_counts().values])
pie_gender.render('gender1.html')


app = flask.Flask(__name__, static_folder='templates')
app.jinja_env.variable_start_string = '{{ '
app.jinja_env.variable_end_string = ' }}'

@app.route('/')
def get_sankey():
    return render_template('index.html',sankey=Markup(pie_gender.render_embed()),title="Learning Dashboard")


if __name__ == "__main__":
    app.run(port=5002,debug = True)












import pandas as pd
import numpy as np
import json
import os

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
pd.set_option('display.max_columns',None)

from pyecharts import options as opts
from pyecharts.charts import *
from pyecharts.options.global_options import ThemeType
from pyecharts.commons.utils import JsCode
from pyecharts.globals import CurrentConfig,NotebookType
CurrentConfig.NOTEBOOK_TYPE = NotebookType.JUPYTER_LAB
from pyecharts.globals import ThemeType
from jinja2 import Markup, Environment, FileSystemLoader

from utils.visualization import *
from utils.HistogramBins import *
from utils.featureCorrelation import *



CurrentConfig.GLOBAL_ENV=Environment(loader=FileSystemLoader('./templates')) #注意要把loader加上


from flask import render_template, url_for
import flask

app = flask.Flask(__name__,static_folder='templates')
# app = Flask(__name__)
app.jinja_env.variable_start_string = '{{ '
app.jinja_env.variable_end_string = ' }}'

dataset = pd.read_csv('data/dataset.csv')



@app.route('/')
# @app.route('/index',methods=['GET'])
def getVisual():
    pie_gender = demo_pie('Gender Distribution', list(dataset.gender.value_counts().index), [int(value) for value in dataset.gender.value_counts().values])
    return render_template('index.html', genderDist=Markup(pie_gender.render_embed()))


def main():
    app.run(host='0.0.0.0',port=5002,debug=True)


if __name__ == '__main__':
    main()