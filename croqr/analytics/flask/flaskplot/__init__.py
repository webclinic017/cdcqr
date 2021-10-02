from flask import Flask, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


app = Flask(__name__)


df = pd.read_pickle('C:\\Users\\Wang\\Documents\\GitHub\\data\\sample_price.pickle')


@app.route('/test')
def chartTest():

    plt.plot(df)
    return render_template('app.html', name = plt.show())

if __name__ == '__main__':

    app.run(debug = False)
    