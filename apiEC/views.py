import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd #Sirve para manipular el dataframe
from sklearn.preprocessing import MinMaxScaler #Sirve para transformar el dataframe a valores entre 0 y 1
import joblib #Es para cargar el modelo con formato h5





#Cargamos los modelos
mrf_diabetes = joblib.load('apiEC/diabetes/random_forest_model.h5')
mrf_hipertension = joblib.load('apiEC/hipertension/random_forest_model.h5')
mrf_er = joblib.load('apiEC/respiratoria/random_forest_model.h5')

#Cargamos los datasets con que se realizo el entrenamiento por cada modelo
datos_e_diabetes = pd.read_csv("apiEC/diabetes/datos.csv",sep=',')
datos_e_hipertension = pd.read_csv("apiEC/hipertension/datos.csv",sep=',')
datos_e_er = pd.read_csv("apiEC/respiratoria/datos.csv",sep=',')

#Obtenemos solo los datos de entrada de cada dataset
entradas_diabetes = datos_e_diabetes.drop(['qecp'], axis=1)
entradas_hipertension = datos_e_hipertension.drop(['qecp'],axis=1)
entradas_er = datos_e_er.drop(['qecp'],axis=1)





@csrf_exempt #Esto es para que no pida el token csrf y me permita hacer el POST
def vistaDiabetes(request):

    if request.method == 'POST':

        # Decodificar los datos de request.body a tipo string
        data = request.body.decode('utf-8')

        # Cargar los datos como objeto JSON
        json_data = json.loads(data)

        #Se asigna un vector de entrada distinto para hacer la prediccion
        #Los datos se obtienen desde la solicitud enviada por el cliente
        entradas_diabetes.loc[286] = [ json_data.get('edad'),json_data.get('genero'),json_data.get('ec'),json_data.get('ocupacion'),json_data.get('tresd'),json_data.get('sefvr'),json_data.get('ecaaqla'),json_data.get('cvcspeu'),json_data.get('uoafpc'),json_data.get('ecarh'),json_data.get('eccoccl'),json_data.get('ecuoaf'),json_data.get('esfhacap'),json_data.get('oapb'),json_data.get('oapc'),json_data.get('ecraca'),json_data.get('cqvuve'),json_data.get('saonacctet'),json_data.get('ccsa'),json_data.get('hpisa'),json_data.get('scmn'),json_data.get('seacr') ]

        #Transformamos el dataset que ya incluye el vector distinto
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(entradas_diabetes)

        #Realizar la prediccion
        predicciones = mrf_diabetes.predict(normalized_data)

        #La prediccion lo transformamos a un dato seriable y legible para el json y lo asignamos a 'prediccion'
        prediccion =  predicciones[286].tolist()

        return JsonResponse({'prediccion':prediccion})





@csrf_exempt #Esto es para que no pida el token csrf y me permita hacer el POST
def vistaHipertension(request):

    if request.method == 'POST':

        # Decodificar los datos de request.body a tipo string
        data = request.body.decode('utf-8')

        # Cargar los datos como objeto JSON
        json_data = json.loads(data)

        #Se asigna un vector de entrada distinto para hacer la prediccion
        #Los datos se obtienen desde la solicitud enviada por el cliente
        entradas_hipertension.loc[326] = [ json_data.get('edad'),json_data.get('genero'),json_data.get('ec'),json_data.get('ocupacion'),json_data.get('tresd'),json_data.get('sefvr'),json_data.get('ecaaqla'),json_data.get('cvcspeu'),json_data.get('uoafpc'),json_data.get('ecarh'),json_data.get('eccoccl'),json_data.get('ecuoaf'),json_data.get('esfhacap'),json_data.get('oapb'),json_data.get('oapc'),json_data.get('ecraca'),json_data.get('cqvuve'),json_data.get('saonacctet'),json_data.get('ccsa'),json_data.get('hpisa'),json_data.get('scmn'),json_data.get('seacr') ]

        #Transformamos el dataset que ya incluye el vector distinto
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(entradas_hipertension)

        #Realizar la prediccion
        predicciones = mrf_hipertension.predict(normalized_data)

        #La prediccion lo transformamos a un dato seriable y legible para el json y lo asignamos a 'prediccion'
        prediccion =  predicciones[326].tolist()

        return JsonResponse({'prediccion':prediccion})





@csrf_exempt #Esto es para que no pida el token csrf y me permita hacer el POST
def vistaER(request):

    if request.method == 'POST':

        # Decodificar los datos de request.body a tipo string
        data = request.body.decode('utf-8')

        # Cargar los datos como objeto JSON
        json_data = json.loads(data)

        #Se asigna un vector de entrada distinto para hacer la prediccion
        #Los datos se obtienen desde la solicitud enviada por el cliente
        entradas_er.loc[58] = [ json_data.get('edad'),json_data.get('genero'),json_data.get('ec'),json_data.get('ocupacion'),json_data.get('tresd'),json_data.get('sefvr'),json_data.get('ecaaqla'),json_data.get('cvcspeu'),json_data.get('uoafpc'),json_data.get('ecarh'),json_data.get('eccoccl'),json_data.get('ecuoaf'),json_data.get('esfhacap'),json_data.get('oapb'),json_data.get('oapc'),json_data.get('ecraca'),json_data.get('cqvuve'),json_data.get('saonacctet'),json_data.get('ccsa'),json_data.get('hpisa'),json_data.get('scmn'),json_data.get('seacr') ]

        #Transformamos el dataset que ya incluye el vector distinto
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(entradas_er)

        #Realizar la prediccion
        predicciones = mrf_er.predict(normalized_data)

        #La prediccion lo transformamos a un dato seriable y legible para el json y lo asignamos a 'prediccion'
        prediccion =  predicciones[58].tolist()

        return JsonResponse({'prediccion':prediccion})