# Sistema conversor de habla a texto basado en redes neuronales
En este repositorio se encuentra la implementación de un sistema 
conversor de habla a texto mediante la utilización de redes neuronales, 
mediante dos de las arquitecturas que componen hoy en día el 
estado del arte: 
- CTC (Connectionist Temporal Classification) 
- Mecanismos de atención.  

Ambos sistemas fueron implementados en TensorFlow.  
Este trabajo fue realizado como trabajo de tesis final para la 
carrera de Ingeniería Electrónica en la Facultad de Ingeniería de la de UBA.

## Configuración del proyecto
La clase [`ProjectData`](src/utils/ProjectData.py) contiene la configuración global del entorno.
En esta clase se encuentran definidas las rutas de donde se levantan los audios y las transcripciones
para ser procesadas, las rutas de guardado de los archivos procesados, los pesos de las redes, etc.
Estas rutas deben ser configuradas acorde al proyecto. 

## Generación de datos
Para el entrenamiento se utilizaron 500 horas de audios de la base de 
datos de <cite>[LibriSpeech][1]</cite>. Se realizaron también ciertas 
validaciones con <cite>[TIMIT][2]</cite>, pero dicha base de datos 
demostró ser muy reducida como para permitir alcanzar buenos resultados.

El formato utilizado para los archivos de entrenamiento es el de **tfrecords**,
los cuales son generados mediante los scripts [`tfrecord_from_librispeech.py`](src/tfrecord_from_librispeech.py) y 
[`tfrecord_from_timit.py`](src/tfrecord_from_timit.py). Dada la gran cantidad de auidos, es conveniente separar
la base de datos en varios archivos `.tfrecord` diferentes. Durante el desarrollo de este
trabajo se dividió el conjunto de audios en archivos de aproximadamente 1000 audios cada uno.  
Se debe tener en cuenta que el formato de los archivos `.wav` al descargar las bases de datos
no siempre es el correcto. Para esto se crearon los scripts [`flac2wav.sh`](utils/flac2wav.sh) y [`FixWav.sh`](utils/FixWav/FixWav.sh) en la 
carpeta `utils` (también se crearon scripts para el copiado de archivos en una estructura 
estándar: [`CopyLibrispeech.sh`](utils/CopyScript/CopyLibrispeech.sh) y [`CopyTimit.sh`](utils/CopyScript/CopyTimit.sh)).  
Una vez que se tienen los audios en el formato adecuado, se deben configurar los features. 
Para esto se deben modificar los parámetros de la clase `FeatureConfig`:
```
# Configuration of the features
feature_config = FeatureConfig()
feature_config.feature_type = 'deep_speech_mfcc'    # 'mfcc', 'spec', 'log_spec', 'deep_speech_mfcc'
feature_config.nfft = 1024
feature_config.winlen = 20
feature_config.winstride = 10
feature_config.preemph = 0.98
feature_config.num_filters = 40
feature_config.num_ceps = 26
feature_config.mfcc_window = np.hanning
```
Estos parámetros fueron los utilizados para obtener los resultados aquí presentados. Se debe aclarar 
que los features calculados en el modo `deep_speech_mfcc` coiciden con los utilizados en
<cite>[DeepSpeech][3]</cite>.  
Es importante también especificar el tipo de label. Durante las primeras
pruebas se utilizó una representación diferente para la red CTC (`ClassicLabel`) y para
la red LAS (`LASLabel`). Sin embargo, se logró unificar las representaciones en ambos modelos,
por lo que se recomienda la utilización de `ClassicLabel`.  

**Importante**: Se debe tener en cuenta los audios y las transcripciones deben estar
almacenados en la carpeta `data` de acuerdo con lo establecido en la clase `ProjectData`.

## Configuración de los modelos
En el inicio del proyecto se crearon clases para implementar los diferentes modelos. Sin embargo,
a medida que éstos se fueron complejizando y se incrementaron la cantidad de datos, se optó por 
utilizar una implementación basada en `tf.estimator`. Los scripts que se deben ejecutar son:
- Red CTC &rightarrow; [`train_zorznet_estimator.py`](src/Estimators/zorznet/train_zorznet_estimator.py)
- Red LAS &rightarrow; [`train_las_estimator.py`](src/Estimators/las/train_las_estimator.py)

Los modelos y las funciones de parseo de datos de ambas redes se encuentran definidos en los 
archivos `model_fn.py` y `data_input_fn.py` en cada una de sus respectivas carpetas. 

Los hiperparámetros deben ser especificados configurando las clases 
[`ZorzNetData.py`](src/neural_network/ZorzNet/ZorzNetData.py) y 
[`LASNetData.py`](src/neural_network/LAS/LASNetData.py) respectivamente. Un ejemplo de 
configuración se puede observar en [best_models/zorznet](best_models/zorznet/README.md) y
[best_models/lasnet](best_models/lasnet/README.md)

Una vez configurados los hiperparámetros ya es posible comenzar a entrenar y validar las
distintas redes.

## Resultados
Para evaluar el error de 
los distintos modelos se deben ejecutar los scripts [`train_zorznet_estimator.py`](src/Estimators/zorznet/train_zorznet_estimator.py) y 
[`train_las_estimator.py`](src/Estimators/las/train_las_estimator.py) con la opción `save_predictions=True` de manera de que 
se almacenen las predicciones de la red en un archivo. Luego se debe ejecutar el script [`evaluate_metrics.py`](src/evaluate_metrics.py) 
para comparar dichas predicciones con las oraciones reales.  
El script se debe ejecutar como `python3.6 evaluate_metrics.py -p predictions.txt -t truth.txt`  

En la siguiente tabla se presenta un resumen de los resultados obtenidos mediante estas redes.

|Red|Dataset|LER|WER|LM|
|:---:|:---:|:---:|:---:|:---:|
|ZorzNet|100hs LS|8.6%|25.1%|-|
|**ZorzNet**|**500hs LS**|**7.2%**|**21.7%**|**-**|
|LASNet|100hs LS|11.1%|27%|-|
|**LASNet**|**500hs LS**|**7.2%**|**17.9%**|**-**|
|[LAS](https://arxiv.org/pdf/1508.01211.pdf)|Google voice|-|16.2%|-|
|[LAS](https://arxiv.org/pdf/1508.01211.pdf)|Google voice|-|10.3%|lm+sampling+rescoring|
|DeepSpeech|300hs SWB|-|25.9%|5-gram|
|DeepSpeech|5000hs|-|11.8%|5-gram|

Se puede ver que los resultados obtenidos son comparables con los de los modelos que representan
el estado del arte, cuando éstos no utilizan gran cantidad de audios ni modelos de lenguaje.

[1]:http://www.openslr.org/12
[2]:https://catalog.ldc.upenn.edu/LDC93S1
[3]:https://github.com/mozilla/DeepSpeech

