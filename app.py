import joblib
from flask import Flask, render_template, redirect, url_for, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, BooleanField
from wtforms.validators import InputRequired, Email, Length
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
import pandas as pd
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

filename = 'diabetes-prediction-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
model1 = pickle.load(open('model1.pkl', 'rb'))
insurancemodel = pickle.load(open('rf_tuned.pkl', 'rb'))
model_lung= pickle.load(open('lungmodel.pkl', 'rb'))
model_breast = pickle.load(open('breastmodel.pkl', 'rb'))
modelmedical = pickle.load(open('rf_tuned.pkl', 'rb'))



app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database.db'
bootstrap = Bootstrap(app)
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(15), unique=True)
    email = db.Column(db.String(50), unique=True)
    password = db.Column(db.String(80))


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class LoginForm(FlaskForm):
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])
    remember = BooleanField('remember me')


class RegisterForm(FlaskForm):
    email = StringField('Email', validators=[InputRequired(), Email(message='Invalid email'), Length(max=50)])
    username = StringField('Username', validators=[InputRequired(), Length(min=4, max=15)])
    password = PasswordField('Password', validators=[InputRequired(), Length(min=8, max=80)])




@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/help')
def help():
    return render_template("help.html")


@app.route('/terms')
def terms():
    return render_template("tc.html")


@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user:
            if check_password_hash(user.password, form.password.data):
                login_user(user, remember=form.remember.data)
                return redirect(url_for('dashboard'))

        return render_template("login.html", form=form)
    return render_template("login.html", form=form)


@app.route('/', methods=['GET', 'POST'])
def signup():
    form = RegisterForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='sha256')
        new_user = User(username=form.username.data, email=form.email.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        return redirect("/login")
    return render_template('signup.html', form=form)


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("dashboard.html")


@app.route("/disindex")

def disindex():
    return render_template("disindex.html")



@app.route("/prostate")
@login_required
def prostate():
    return render_template("prostate.html")


@app.route("/lungcancer")
@login_required
def lungcancer():
    return render_template("lunghome.html")



@app.route('/lungpredict', methods=['POST'])
def lungpredict():
    if request.method == 'POST':
        gen = request.form['GENDER']
        age = request.form['AGE']
        smoke = request.form['SMOKING']
        yell = request.form['YELLOW_FINGERS']
        anx = request.form['ANXIETY']
        peer = request.form['PEER_PRESSURE']
        chronic= request.form['CHRONIC DISEASE']
        fat = request.form['FATIGUE']
        aller = request.form['ALLERGY']
        wheez = request.form['WHEEZING']
        alco = request.form['ALCOHOL CONSUMING']
        cough = request.form['COUGHING']
        breath = request.form['SHORTNESS OF BREATH']
        swall = request.form['SWALLOWING DIFFICULTY']
        chest = request.form['CHEST PAIN']

        mypred = np.array([[gen,age,smoke,yell,anx,peer,chronic,fat,aller,wheez,alco,cough,breath,swall,chest]])
        my_prediction = model_lung.predict(mypred)

        return render_template('lungop.html', prediction=my_prediction)



@app.route("/diabetes")
@login_required
def diabetes():
    return render_template("diabetes.html")


@app.route("/heart")
@login_required
def heart():
    return render_template("heart.html")

##################################################################################
# Kidney Disease prediction

@app.route("/kidney")
@login_required
def kidney():
    return render_template("kidney.html")


def ValuePredictor(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1, size)
    if size == 7:
        loaded_model = joblib.load('kidney_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route("/predictkidney",  methods=['GET', 'POST'])
def predictkidney():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePredictor(to_predict_list, 7)
    if(int(result) == 1):
        prediction = "Patient has a high risk of Kidney Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Kidney Disease"
    return render_template("kidney_result.html", prediction_text=prediction)

##################################################################################
# Liver Disease prediction


@app.route("/liver")
@login_required
def liver():
    return render_template("liver.html")


def ValuePred(to_predict_list, size):
    to_predict = np.array(to_predict_list).reshape(1,size)
    if(size==7):
        loaded_model = joblib.load('liver_model.pkl')
        result = loaded_model.predict(to_predict)
    return result[0]


@app.route('/predictliver', methods=["POST"])
def predictliver():
    if request.method == "POST":
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(float, to_predict_list))
        if len(to_predict_list) == 7:
            result = ValuePred(to_predict_list, 7)

    if int(result) == 1:
        prediction = "Patient has a high risk of Liver Disease, please consult your doctor immediately"
    else:
        prediction = "Patient has a low risk of Kidney Disease"
    return render_template("liver_result.html", prediction_text=prediction)


@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


##################################################################################
# breastcancer prediction


@app.route("/breastcancer")
@login_required
def breastcancer():
    return render_template("breast.html")



@app.route('/breastpredict',methods=['POST'])
def breastpredict():
  input_features = [int(x) for x in request.form.values()]
  features_value = [np.array(input_features)]

  features_name = ['clump_thickness', 'uniform_cell_size', 'uniform_cell_shape',
       'marginal_adhesion', 'single_epithelial_size', 'bare_nuclei',
       'bland_chromatin', 'normal_nucleoli', 'mitoses']

  df = pd.DataFrame(features_value, columns=features_name)
  output = model_breast.predict(df)

  if output == 4:
      res_val = "Breast cancer"
      return render_template('breastoutput.html',  prediction_text='Patient has {}'.format(res_val))
  else:
      res_val = "no Breast cancer"
      return render_template('breastoutput.html',  prediction_text='Patient has {}'.format(res_val))


##################################################################################

df1 = pd.read_csv('diabetes.csv')

# Renaming DiabetesPedigreeFunction as DPF
df1 = df1.rename(columns={'DiabetesPedigreeFunction': 'DPF'})

# Replacing the 0 values from ['Glucose','BloodPressure','SkinThickness','Insulin','BMI'] by NaN
df_copy = df1.copy(deep=True)
df_copy[['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']] = df_copy[['Glucose', 'BloodPressure',
                                                                                    'SkinThickness', 'Insulin',
                                                                                    'BMI']].replace(0, np.NaN)

# Replacing NaN value by mean, median depending upon distribution
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)

# Model Building

X = df1.drop(columns='Outcome')
y = df1['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

# Creating Random Forest Model

classifier = RandomForestClassifier(n_estimators=20)
classifier.fit(X_train, y_train)

# Creating a pickle file for the classifier
filename = 'diabetes-prediction-rfc-model.pkl'
pickle.dump(classifier, open(filename, 'wb'))

#####################################################################


@app.route('/predictt', methods=['POST'])
def predictt():
    if request.method == 'POST':
        preg = request.form['pregnancies']
        glucose = request.form['glucose']
        bp = request.form['bloodpressure']
        st = request.form['skinthickness']
        insulin = request.form['insulin']
        bmi = request.form['bmi']
        dpf = request.form['dpf']
        age = request.form['age']

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = classifier.predict(data)

        return render_template('diab_result.html', prediction=my_prediction)


############################################################################################################
# Medical Insurance Prediction


@app.route('/insurance')
@login_required
def insurance():
    return render_template('insurance.html')



    
@app.route('/ipredict', methods=['POST','GET'])
def ipredict():
    features = [int(x) for x in request.form.values()]

    print(features)
    final = np.array(features).reshape((1,6))
    print(final)
    pred = insurancemodel.predict(final)[0]
    print(pred)

    
    if pred < 0:
        return render_template('insuranceresult.html', pred='Error calculating Amount!')
    else:
        return render_template('insuranceresult.html', pred='Expected amount is Rs.{0:.3f}'.format(pred))



############################################################################################################
# prostate cancer code


df=pd.read_csv('Prostate_cancer_data.csv')
df=df.dropna(axis=1)#Drop the column with empty data
df=df.drop(['id'],axis=1)

#Encoding first column
from sklearn.preprocessing import LabelEncoder
labelencoder_X=LabelEncoder()#Calling LabelEncoder
df.iloc[:,0]=labelencoder_X.fit_transform(df.iloc[:,0].values)#Encoding the values of diagnosis column to values


#Splitting data for dependence
X=df.iloc[:,1:].values#Features of cancerous and non cancerous patients
Y=df.iloc[:,0].values#Whether patient has cancer or not

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25,random_state=1)

forest=RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
forest.fit(X_train,Y_train)

filename = 'model3.pkl'
pickle.dump(forest, open(filename, 'wb'))



@app.route('/prostatepredict', methods=['POST'])
def prostatepredict():
    if request.method == 'POST':
        rad = request.form['radius']
        tex = request.form['texture']
        par = request.form['perimeter']
        area = request.form['area']
        smooth = request.form['smoothness']
        compact = request.form['compactness']
        symme= request.form['symmetry']
        frac = request.form['fractal_dimension']

        mypred = np.array([[rad, tex, par, area, smooth, compact,symme, frac]])
        my_prediction = forest.predict(mypred)

        return render_template('prostate_result.html', prediction=my_prediction)

  


############################################################################################################

@app.route('/predictheart', methods=['POST'])
def predictheart():
    input_features = [float(x) for x in request.form.values()]
    features_value = [np.array(input_features)]

    features_name = ["age", "trestbps", "chol", "thalach", "oldpeak", "sex_0",
                     "  sex_1", "cp_0", "cp_1", "cp_2", "cp_3", "  fbs_0",
                     "restecg_0", "restecg_1", "restecg_2", "exang_0", "exang_1",
                     "slope_0", "slope_1", "slope_2", "ca_0", "ca_1", "ca_2", "thal_1",
                     "thal_2", "thal_3"]

    df = pd.DataFrame(features_value, columns=features_name)
    output = model1.predict(df)

    if output == 1:
        res_val = "a high risk of Heart Disease"
    else:
        res_val = "a low risk of Heart Disease"

    return render_template('heart_result.html', prediction_text='Patient has {}'.format(res_val))


############################################################################################################

if __name__ == "__main__":
    app.run(debug=True)

