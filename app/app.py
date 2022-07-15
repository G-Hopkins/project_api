from flask import Flask, render_template, request, url_for, redirect
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, FloatField, SubmitField
from wtforms.validators import DataRequired
import numpy as np
import os
from functions import *

# creation of Flask session
app = Flask(__name__, template_folder='./public')

# this is for Flask-WTF
app.config['SECRET_KEY'] = 'cdfgh104j2j1ki43j2'

# flask-bootstrap needs this
Bootstrap(app)

params = np.array([])
params2 = np.array([])

# form for Black-Scholes inputs
class bsForm(FlaskForm):
    spot_price = FloatField('Spot price: ', validators=[DataRequired()])
    strike_price = FloatField('Strike price: ', validators=[DataRequired()])
    risk_free_rate = FloatField('Risk-free rate: ', validators=[DataRequired()])
    time_to_maturity = FloatField('Time-to-maturity: ', validators=[DataRequired()])
    vol = FloatField('Volatility: ', validators=[DataRequired()])
    submit = SubmitField('Get Prices')

# form for Heston inputs
class hestonForm(FlaskForm):
    spot_price = FloatField('Spot price: ', validators=[DataRequired()])
    strike_price = FloatField('Strike price: ', validators=[DataRequired()])
    risk_free_rate = FloatField('Risk-free rate: ', validators=[DataRequired()])
    time_to_maturity = FloatField('Time-to-maturity: ', validators=[DataRequired()])
    vol = FloatField('Volatility: ', validators=[DataRequired()])
    vol_of_vol = FloatField('Volatility of volatility: ', validators=[DataRequired()])
    long_average_variance = FloatField('Long average variance: ', validators=[DataRequired()])
    reversion_speed = FloatField('Reversion speed: ', validators=[DataRequired()])
    correlation = FloatField('Correlation: ', validators=[DataRequired()])
    submit2 = SubmitField('Get Prices')

# button to go back to the home page
class back(FlaskForm):
    submit3 = SubmitField('Return to pricing calculator')

# button to fill with random test data
class fill_bs(FlaskForm):
    submit4 = SubmitField('Fill Black-Scholes with random test inputs')

class fill_heston(FlaskForm):
    submit5 = SubmitField('Fill Heston with random test inputs')

# homepage
@app.route('/', methods=['GET', 'POST'])
def render():
    form = bsForm()
    form2 = hestonForm()
    filler_bs = fill_bs()
    filler_heston = fill_heston()
    message = ""
    if form.submit.data:
        spot = form.spot_price.data
        strike = form.strike_price.data
        r = form.risk_free_rate.data
        t = form.time_to_maturity.data
        v = form.vol.data
        global params
        params = np.array([spot,strike,r,t,v])
        return redirect(url_for('bs_pricing', params=params, form=form, form2=form2))
        
    if form2.submit2.data:
        spot = form2.spot_price.data
        strike = form2.strike_price.data
        r = form2.risk_free_rate.data
        t = form2.time_to_maturity.data
        v = form2.vol.data
        vol_of_v = form2.vol_of_vol.data
        theta = form2.long_average_variance.data
        k = form2.reversion_speed.data
        p = form2.correlation.data
        global params2
        params2 = np.array([spot, strike, r, t, v, vol_of_v, theta, k, p])
        return redirect(url_for('heston_pricing', params2=params2, form=form, form2=form2))
    
    if filler_heston.submit5.data:
        s, k, r, t, vol, vol_of_vol, long_average_variance, reversion_speed, correlation = heston_generation()
        form2.spot_price.data = s
        form2.strike_price.data = k
        form2.risk_free_rate.data = r
        form2.time_to_maturity.data = t
        form2.vol.data = vol
        form2.vol_of_vol.data = vol_of_vol
        form2.long_average_variance.data = long_average_variance
        form2.reversion_speed.data = reversion_speed
        form2.correlation.data = correlation

    if filler_bs.submit4.data:
        s,k,r,t,vol = bs_generation()
        form.spot_price.data = s
        form.strike_price.data = k
        form.risk_free_rate.data = r
        form.time_to_maturity.data = t
        form.vol.data = vol
    

    return render_template('index.html', form=form, form2=form2, message=message, filler_bs=filler_bs, filler_heston = filler_heston)

# black-scholes page
@app.route('/bs_price', methods=['GET', 'POST'])
def bs_pricing():
    back_button = back()
    ann_1_prices = ann_1_pricing(params)
    ann_1_call = round(ann_1_prices[0],4)
    ann_1_put = round(ann_1_prices[1],4)
    ann_1_time = round(ann_1_prices[2],4)
    ann_2_prices = ann_2_pricing(params)
    ann_2_call = round(ann_2_prices[0],4)
    ann_2_time = round(ann_2_prices[1],4)
    ann_3_prices = ann_3_pricing(params)
    ann_3_put = round(ann_3_prices[0],4)
    ann_3_time = round(ann_3_prices[1],4)
    bs_prices = black_scholes(params)
    bs_call = round(bs_prices[0],4)
    bs_put = round(bs_prices[1],4)
    bs_time = round(bs_prices[2],4)
    if back_button.submit3.data:
        return redirect(url_for('render'))
    return render_template('pricing.html', ann_1_call = ann_1_call, ann_1_put = ann_1_put, bs_call = bs_call,
    bs_put = bs_put, ann_2_call = ann_2_call, ann_3_put = ann_3_put, params=params, back_button = back_button, 
    spot_price = params[0], strike_price=params[1], r = params[2], t=params[3], v=params[4], 
    ann_1_time = ann_1_time, ann_2_time = ann_2_time, ann_3_time = ann_3_time, bs_time = bs_time)

# heston page
@app.route('/heston_price', methods=['GET', 'POST'])
def heston_pricing():
    back_button = back()
    heston_monte_carlo_prices = heston_monte_carlo(params2)
    heston_call = round(heston_monte_carlo_prices[0],4)
    heston_put = round(heston_monte_carlo_prices[1], 4)
    heston_time = round(heston_monte_carlo_prices[2],4)
    ann_4_prices = ann_4_pricing(params2)
    ann_4_call = round(ann_4_prices[0],4)
    ann_4_put = round(ann_4_prices[1],4)
    ann_4_time = round(ann_4_prices[2],4)
    if back_button.submit3.data:
        return redirect(url_for('render'))
    return render_template('pricing2.html', heston_call = heston_call, heston_put = heston_put, 
    ann_4_call = ann_4_call, ann_4_put = ann_4_put,params2=params2, back_button=back_button,spot_price = params2[0], strike_price=params2[1],
    r = params2[2], t=params2[3], v=params2[4], vol_of_v = params2[5], theta=params2[6], k = params2[7], p = params2[8],
    heston_time = heston_time, ann_4_time = ann_4_time)

if __name__ == '__main__':
    #port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', debug=True)
