# Lab02_SDS

## Instrucciones:
El archivo lab01.py contiene los metodos para el preprocesamiento, entrenamiento del modelo y la muestra de las metricas, ejecutanto el comando python lab01.py

IMPORTANTE: No se pudo obtener la matriz de correlacion, ya que esta tardaba mas de 8 horas en compilar, junto con el reporte de profile.

Se adjunta el modelo obtenido en el archivo model.pkl


### Modelo TF-IDF

Accuracy Score: 0.5357291233901121

Matriz de confusion

                 precision    recall    f1-score      support

    spam           0.59        0.60       0.60          2741
    legitimate     0.46        0.45       0.45          2073

    accuracy                              0.54          4814
    macro avg      0.52        0.52       0.52          4814
    weighted avg   0.53        0.54       0.53          4814

Debido a la gran cantidad de plabras que habia en los mensajes, se tuvo que ajustar el modelo TD-IDF a un minimo de frecuencias de 0.05 y un maximo de 1. Esto debido a que la cantidad de frecuencias cambiaba drasticamente de 0 a 0.05: se redujo de 151919 columnas a 178. Por otra parte, cuando se reducia el maximo de frecuencas de 1 a 0.5, solo reducia 1 columna. Por lo tanto, se trabajadon con 178 frecuencias. Ejemplo:

    mailing  message    need      new     people       pm     send     sent     thanks     time     today    use     want     way     work
    0.0       0.00      0.00      0.15     0.0         0.00   0.0      0.00      0.00       0.00    0.0      0.00     0.00    0.0     0.00    
    0.0       0.00      0.00      0.00     0.0         0.00   0.0      0.00      0.00       0.00    0.0      0.35     0.00    0.0     0.00    
    0.0       0.00      0.00      0.00     0.0         0.00   0.0      0.00      0.00       0.00    0.0      0.50     0.00    0.0     0.00    
    0.0       0.00      0.00      0.08     0.0         0.00   0.0      0.00      0.00       0.00    0.0      0.00     0.00    0.0     0.00    
    0.0       0.00      0.00      0.00     0.0         0.00   0.0      0.00      0.00       0.00    0.0      0.36     0.00    0.0     0.00    
 
Debido a esta cantidad tan grande de frecuencias, es congruente que los valore de las metricas hayan sido tan "bajos", al rededor de 0.6.

## Conclusion

¿Qué representación numérica produjo el mejor resultado?

