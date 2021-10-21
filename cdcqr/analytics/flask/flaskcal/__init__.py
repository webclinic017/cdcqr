import os
from flask import Flask, render_template, request
from croqr.analytics.flask.utils import plotstart, plotend
import pandas as pd
import matplotlib.pyplot as plt


img=plotstart()
figs=[]
df=pd.read_pickle('C:\\Users\\Wang\\Documents\\GitHub\\data\\sample_price.pickle')
df.plot()
plt.title('df')
figs.append(plotend(img))
df_str = figs[0]


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskcal.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    
    @app.route('/')
    def main():
        return render_template('app.html')


    @app.route('/send', methods=['POST'])
    def send(sum=sum):
        if request.method == 'POST':
            num1 = request.form['num1']
            num2 = request.form['num2']
            operation = request.form['operation']

            if operation == 'add':
                sum = float(num1) + float(num2)
                return render_template('app.html', sum=sum, fig_str = df_str)

            elif operation == 'subtract':
                sum = float(num1) - float(num2)
                return render_template('app.html', sum=sum, fig_str = df_str)

            elif operation == 'multiply':
                sum = float(num1) * float(num2)
                return render_template('app.html', sum=sum, fig_str = df_str)

            elif operation == 'divide':
                sum = float(num1) / float(num2)
                return render_template('app.html', sum=sum, fig_str = df_str)
            else:
                return render_template('app.html')

    from . import db
    db.init_app(app)
    return app
