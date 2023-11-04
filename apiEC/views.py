import json
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
import pandas as pd #Sirve para manipular el dataframe
import numpy as np
from sklearn.preprocessing import MinMaxScaler #Sirve para transformar el dataframe a valores entre 0 y 1
import joblib #Es para cargar el modelo con formato h5





#Cargamos los modelos
mrf_diabetes = joblib.load('apiEC/diabetes/random_forest_model.h5')
mrf_hipertension = joblib.load('apiEC/hipertension/random_forest_model.h5')
mrf_er = joblib.load('apiEC/respiratoria/random_forest_model.h5')
mlstm_sistolica = joblib.load('apiEC/pasistolica/sistolica.h5')
mlstm_diastolica = joblib.load('apiEC/padiastolica/diastolica.h5')

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
        entradas_diabetes.loc[286] = [ json_data.get('edad'),json_data.get('genero'),json_data.get('ec'),json_data.get('ocupacion'),json_data.get('tresd'),json_data.get('sefvr'),json_data.get('ecaaqla'),json_data.get('cvcspeu'),json_data.get('uoafpc'),json_data.get('ecarh'),json_data.get('eccoccl'),json_data.get('ecuoaf'),json_data.get('esfhacap'),json_data.get('oapb'),json_data.get('oapc'),json_data.get('ecraca'),json_data.get('cqvuve'),json_data.get('saonacctet'),json_data.get('ccsa'),json_data.get('hpisa'),json_data.get('scmn'),json_data.get('seacr'),json_data.get('paecda') ]

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
        entradas_hipertension.loc[326] = [ json_data.get('edad'),json_data.get('genero'),json_data.get('ec'),json_data.get('ocupacion'),json_data.get('tresd'),json_data.get('sefvr'),json_data.get('ecaaqla'),json_data.get('cvcspeu'),json_data.get('uoafpc'),json_data.get('ecarh'),json_data.get('eccoccl'),json_data.get('ecuoaf'),json_data.get('esfhacap'),json_data.get('oapb'),json_data.get('oapc'),json_data.get('ecraca'),json_data.get('cqvuve'),json_data.get('saonacctet'),json_data.get('ccsa'),json_data.get('hpisa'),json_data.get('scmn'),json_data.get('seacr'),json_data.get('paecda') ]

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
        entradas_er.loc[58] = [ json_data.get('edad'),json_data.get('genero'),json_data.get('ec'),json_data.get('ocupacion'),json_data.get('tresd'),json_data.get('sefvr'),json_data.get('ecaaqla'),json_data.get('cvcspeu'),json_data.get('uoafpc'),json_data.get('ecarh'),json_data.get('eccoccl'),json_data.get('ecuoaf'),json_data.get('esfhacap'),json_data.get('oapb'),json_data.get('oapc'),json_data.get('ecraca'),json_data.get('cqvuve'),json_data.get('saonacctet'),json_data.get('ccsa'),json_data.get('hpisa'),json_data.get('scmn'),json_data.get('seacr'),json_data.get('paecda') ]

        #Transformamos el dataset que ya incluye el vector distinto
        scaler = MinMaxScaler(feature_range=(0, 1))
        normalized_data = scaler.fit_transform(entradas_er)

        #Realizar la prediccion
        predicciones = mrf_er.predict(normalized_data)

        #La prediccion lo transformamos a un dato seriable y legible para el json y lo asignamos a 'prediccion'
        prediccion =  predicciones[58].tolist()

        return JsonResponse({'prediccion':prediccion})





@csrf_exempt #Esto es para que no pida el token csrf y me permita hacer el POST
def vistaPASistolica(request,numero_dias):

    if request.method == 'POST':

        # Decodificar los datos de request.body a tipo string
        data = request.body.decode('utf-8')

        # Cargar los datos como objeto JSON
        json_data = json.loads(data)
        df = pd.DataFrame(json_data)#Eso lo paso a convertir en un dataframe

        #Ajustes a los datos para tenerlo mas organizado
        df['Fecha'] = pd.to_datetime(df['Fecha']) #Convertir la columna 'Fecha' a tipo datetime64
        #ordenar de forma ascendente las fechas, desde el pasado hasta el presente
        df = df.sort_values(by='Fecha', ascending=True)
        #voy a reasignar indicesal dataframe
        df = df.reset_index(drop=True)
        # Establece la columna 'Fecha' como el índice del DataFrame
        df.set_index('Fecha', inplace=True)
        #Interpolando el dataset
        df = df.resample('D').mean().interpolate()


        # Preprocesamiento de datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df["PA Sistolica"].values.reshape(-1, 1))

        #Cargar el modelo

        # Realizar predicciones
        datos_de_entrada = scaled_data.reshape(-1,1,1)  # Reformatear los datos
        predicciones = mlstm_sistolica.predict(datos_de_entrada)
        predicciones = scaler.inverse_transform(predicciones.reshape(-1,1))
        valores_reales = predicciones

        predicciones_mostrar = []
        for i in range(numero_dias):  # Predecir los próximos 10 días

            scaler = MinMaxScaler(feature_range=(0, 1))
            valores_reales = scaler.fit_transform(valores_reales)
            last_value = valores_reales[-1, 0]#66.8 -> 0.4439338235294117

            #next_date = last_date + pd.DateOffset(days=1)  # Añadir un día a la última fecha
            next_sequence = np.array([[last_value]])

            # Utilizar el modelo para realizar la predicción para el siguiente día
            next_sequence = next_sequence.reshape(-1, 1, 1)  # Reformatear para que tenga forma (None, 1, 1)
            next_prediction = mlstm_sistolica.predict(next_sequence)
            next_prediction = next_prediction.reshape(-1, 1)
            next_prediction = scaler.inverse_transform(next_prediction)#aqui se tranforma a real o numero real
            valores_reales =  scaler.inverse_transform(valores_reales)#aqui se tranforma a real o numero real

            #pronostico = np.append(pronostico, next_prediction, axis=0)
            valores_reales = np.append(valores_reales, next_prediction, axis=0)

            #Aqui agrego los numeros de las predicciones que voy a mandar en formato json
            predicciones_mostrar.append(next_prediction[0, 0])

            # Imprimir el pronóstico para el próximo día
            #print(f"Predicción para el próximo día: {next_prediction[0, 0]}")
            # Convierte los valores float32 a float
            predicciones_serializables = [float(value) for value in predicciones_mostrar]

        return JsonResponse({'prediccion':predicciones_serializables})







@csrf_exempt #Esto es para que no pida el token csrf y me permita hacer el POST
def vistaPADiastolica(request,numero_dias):

    if request.method == 'POST':

        # Decodificar los datos de request.body a tipo string
        data = request.body.decode('utf-8')

        # Cargar los datos como objeto JSON
        json_data = json.loads(data)
        df = pd.DataFrame(json_data)#Eso lo paso a convertir en un dataframe

        #Ajustes a los datos para tenerlo mas organizado
        df['Fecha'] = pd.to_datetime(df['Fecha']) #Convertir la columna 'Fecha' a tipo datetime64
        #ordenar de forma ascendente las fechas, desde el pasado hasta el presente
        df = df.sort_values(by='Fecha', ascending=True)
        #voy a reasignar indicesal dataframe
        df = df.reset_index(drop=True)
        # Establece la columna 'Fecha' como el índice del DataFrame
        df.set_index('Fecha', inplace=True)
        #Interpolando el dataset
        df = df.resample('D').mean().interpolate()


        # Preprocesamiento de datos
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df["PA Diastolica"].values.reshape(-1, 1))

        #Cargar el modelo

        # Realizar predicciones
        datos_de_entrada = scaled_data.reshape(-1,1,1)  # Reformatear los datos
        predicciones = mlstm_diastolica.predict(datos_de_entrada)
        predicciones = scaler.inverse_transform(predicciones.reshape(-1,1))
        valores_reales = predicciones

        predicciones_mostrar = []
        for i in range(numero_dias):  # Predecir los próximos 10 días

            scaler = MinMaxScaler(feature_range=(0, 1))
            valores_reales = scaler.fit_transform(valores_reales)
            last_value = valores_reales[-1, 0]#66.8 -> 0.4439338235294117

            #next_date = last_date + pd.DateOffset(days=1)  # Añadir un día a la última fecha
            next_sequence = np.array([[last_value]])

            # Utilizar el modelo para realizar la predicción para el siguiente día
            next_sequence = next_sequence.reshape(-1, 1, 1)  # Reformatear para que tenga forma (None, 1, 1)
            next_prediction = mlstm_diastolica.predict(next_sequence)
            next_prediction = next_prediction.reshape(-1, 1)
            next_prediction = scaler.inverse_transform(next_prediction)#aqui se tranforma a real o numero real
            valores_reales =  scaler.inverse_transform(valores_reales)#aqui se tranforma a real o numero real

            #pronostico = np.append(pronostico, next_prediction, axis=0)
            valores_reales = np.append(valores_reales, next_prediction, axis=0)

            #Aqui agrego los numeros de las predicciones que voy a mandar en formato json
            predicciones_mostrar.append(next_prediction[0, 0])

            # Imprimir el pronóstico para el próximo día
            #print(f"Predicción para el próximo día: {next_prediction[0, 0]}")
            # Convierte los valores float32 a float
            predicciones_serializables = [float(value) for value in predicciones_mostrar]

        return JsonResponse({'prediccion':predicciones_serializables})