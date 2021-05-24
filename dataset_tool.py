# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Tool for creating multi-resolution TFRecords datasets for StyleGAN and ProGAN."""

# pylint: disable=too-many-lines
import os
import sys
import glob
import argparse
import threading
import six.moves.queue as Queue # pylint: disable=import-error
import traceback
import numpy as np
import tensorflow as tf
import PIL.Image
import dnnlib.tflib as tflib

from training import dataset

#----------------------------------------------------------------------------

# Función que muestra en pantalla un mensaje de error, si sucede un problema con el dataset
def error(msg):
    print('Error: ' + msg)
    exit(1)

#----------------------------------------------------------------------------

# Definimos una clase para utilizar el dataset para hacer uso de multiples tamaños de datos
class TFRecordExporter:
    # Función para la inicialización de la clase 
    # El parámetro self hace referencia a si mismo(en c es como this), 
    def __init__(self, tfrecord_dir, expected_images, print_progress=True, progress_interval=10):
        # Obtenemos la dirección del tfrecord_dir haciendo referencia a tensorflow y obtenemos su prefijo
        self.tfrecord_dir       = tfrecord_dir
        self.tfr_prefix         = os.path.join(self.tfrecord_dir, os.path.basename(self.tfrecord_dir))

        # Definimos la cantidad de imagenes esperadas, esto puede variar según la dataset, tienen que ser como 
        # mínimo 10 imagenes
        self.expected_images    = expected_images

        # Asignamos 0 para imagenes cortadas
        self.cur_images         = 0

        # No definimos el tamaño, ya que deseamos hacer que la dataset sea multi-tamaño
        self.shape              = None

        # No definimos el tamaño, ya que la resolución puede variar
        self.resolution_log2    = None

        # Definimos una variable vacia del tensor para escribir, es un arreglo.
        self.tfr_writers        = []

        # Definimos una variable ue determina si el proceso se puede imprimir o no
        self.print_progress     = print_progress

        # Definimos el intervalo que toma la dataset para avanzar, en este caso se hace uso de 10, pero esto puede variar
        self.progress_interval  = progress_interval

        # Verificamos que el proceso se haya terminado para poder imprimir en pantalla que se cargo
        # los datos desde la dirección que se proporciono
        if self.print_progress:
            print('Creating dataset "%s"' % tfrecord_dir)
        # Se valida si la dirección es correcta o erronea, en caso de serlo se finaliza el programa
        if not os.path.isdir(self.tfrecord_dir):
            os.makedirs(self.tfrecord_dir)
        assert os.path.isdir(self.tfrecord_dir)

    # Función que permite cerrar o finalizar el programa al cargar la dataset 
    def close(self):
        # Verificamos si el proceso se encuentra en proceso, de ser así se imprime el borrado de los datos
        # definimos previamente
        if self.print_progress:
            print('%-40s\r' % 'Flushing data...', end='', flush=True)
        # Hacemos uso del self que almacena los datos previamente inicializados,  recorremos el arreglo
        # cerrando el tensor e inicializando con un arreglo vacio los datos
        for tfr_writer in self.tfr_writers:
            tfr_writer.close()
        self.tfr_writers = []
        # En el caso que el proceso continue, se imprime las imagenes cortadas privamente en el tensor
        if self.print_progress:
            print('%-40s\r' % '', end='', flush=True)
            print('Added %d images.' % self.cur_images)

    # Función que termina un orden aleatoriamente, se pasa como parámetro los datos previamente
    # inicializados
    def choose_shuffled_order(self): # Note: Images and labels must be added in shuffled order.
        # Asignamos el rango definido en donde se encontrarán los datos aleatorios
        order = np.arange(self.expected_images)
        # Obtenemos los datos aleatorios y regresamos el orden
        np.random.RandomState(123).shuffle(order)
        return order

    # Adicionamos las imagenes que deseamos pasar mediante la dataset
    def add_image(self, img):
        # Verificamos que el proceso se encuentre activo, además de verificar que las imagenes
        # cortadas se encuentren adecuadas con el intervalo, es decir, si se tiene
        # 1000 imágenes y un intervalo de 10 este como resultado tendrá 0, ya que cumple que 
        # es un múltiplo 
        if self.print_progress and self.cur_images % self.progress_interval == 0:
            # Se imprimen las imagenes cortadas y la cantidad esperada
            print('%d / %d\r' % (self.cur_images, self.expected_images), end='', flush=True)
        # Ya que no conocemos el tamaño de las dimensiones de las imagenes, se define mejor
        if self.shape is None:
            # Definimos eel tamaño según las imagenes que se ingresan
            self.shape = img.shape
            # Se asigna la resolución, definida mediante alguna imagen del conjunto obtenido
            self.resolution_log2 = int(np.log2(self.shape[1]))

            # Nos aseguramos que el resultado se encuentre en el rango de 1 a 3
            assert self.shape[0] in [1, 3]
            # Verificamos que el tamaño de las imagenes sean correctas
            assert self.shape[1] == self.shape[2]
            # De ser así, al aplicarse la resolución de vuelta a la imagen seleccionada debe ser verdadero
            assert self.shape[1] == 2**self.resolution_log2
            # Optimizamos el tensor de la imagen de entrada
            tfr_opt = tf.python_io.TFRecordOptions(tf.python_io.TFRecordCompressionType.NONE)
            # Recorremos la resolución de la imagen, recordemos que empezamos con 0, así que tenemos un dato 
            # por lo cual se resta
            for lod in range(self.resolution_log2 - 1):
                # Hacemos uso del archivo que contiene un prefijo(etiqueta) de la imagen a utilizar y su resolución
                tfr_file = self.tfr_prefix + '-r%02d.tfrecords' % (self.resolution_log2 - lod)
                # Adicionamos en arreglo del tensor los datos de la imagen
                self.tfr_writers.append(tf.python_io.TFRecordWriter(tfr_file, tfr_opt))
        # Se realiza solo en caso de coincidir con el mismo tamaño las imagenes
        assert img.shape == self.shape
        # Recorremos el arreglo del tensor escrito previamente, obteniendo etiqueta e imagen
        for lod, tfr_writer in enumerate(self.tfr_writers):
            # Verificamos la existencia de la etiqueta
            if lod:
                # Si existe entonces definimos la imagen en una variable
                img = img.astype(np.float32)
                # Dimensionamos la imagen previamente obtenida
                img = (img[:, 0::2, 0::2] + img[:, 0::2, 1::2] + img[:, 1::2, 0::2] + img[:, 1::2, 1::2]) * 0.25
            # Definimos formato a la imagen dimensionada anteriormente
            quant = np.rint(img).clip(0, 255).astype(np.uint8)
            # anexamos los datos que se entrenaran en la red, usando la dimensionalidad y el dato
            ex = tf.train.Example(features=tf.train.Features(feature={
                'shape': tf.train.Feature(int64_list=tf.train.Int64List(value=quant.shape)),
                'data': tf.train.Feature(bytes_list=tf.train.BytesList(value=[quant.tostring()]))}))
            tfr_writer.write(ex.SerializeToString())
        self.cur_images += 1

    # Función para adicionar etiquetas a las imagenes 
    def add_labels(self, labels):
        # Verificamos si la impresión continua en proceso en caso de ser así
        if self.print_progress:
            # Mostramo en pantalla el mensaje de que se estan guardando las etiquetas
            print('%-40s\r' % 'Saving labels...', end='', flush=True)
        # En caso de no coincidir la dimensionalidad de las imagenes con etiqueta y las cortadas marcara error
        assert labels.shape[0] == self.cur_images
        # Abrimos las imagenes con su prefijo obtenido previamente
        with open(self.tfr_prefix + '-rxx.labels', 'wb') as f:
            # Guardamos las etiquetas
            np.save(f, labels.astype(np.float32))

    # Retornamos el puntero con la inicialización
    def __enter__(self):
        return self

    # Definimos la salida del programa mediante un argumento en consola
    def __exit__(self, *args):
        self.close()

#----------------------------------------------------------------------------
# Clase para excepción de información
class ExceptionInfo(object):
    # Función que inicializa la información excepcional
    def __init__(self):
        self.value = sys.exc_info()[1]
        self.traceback = traceback.format_exc()

#----------------------------------------------------------------------------
# Clase que realiza el hilo o hilos del trabajo
class WorkerThread(threading.Thread):
    # Función para inicializar la clase, que consiste en hacer uso del puntero en el
    # el hilo de rosca
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    # Función que corre los hilos previamente definidos en la inicialización
    def run(self):
        # Repetimos el ciclo infinitamente
        while True:
            # Obtenemos los datos de la cola hecha previamente
            func, args, result_queue = self.task_queue.get()
            # Si la función es nula entonces se rompe la consola
            if func is None:
                break
            # verificamos el argumento y lo metemos en el resultado de la cola
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))

#----------------------------------------------------------------------------
# Clase del grupo de hilos
class ThreadPool(object):
    # Función que inicializa la estructura de los hilos
    def __init__(self, num_threads):
        # Verificamos que el número de hilos no sean cero, es decir, que exista uno por lo menos
        assert num_threads >= 1
        # Determinamos en una variable del puntero las tareas que se ejecutan en la cola
        self.task_queue = Queue.Queue()
        # Hacemos que las colas se resguarden según la tarea que se ejecuta
        self.result_queues = dict()
        # Definimos la cantidad de hilos que tendra
        self.num_threads = num_threads
        # Realizamos un recorrido del número de hilos que deseamos
        for _idx in range(self.num_threads):
            # En la clase WorkerThread se anexa las tareas que se desean correr
            thread = WorkerThread(self.task_queue)
            # Definimos como verdadero la variable e iniciamos los hilos
            thread.daemon = True
            thread.start()
    
    # Función que adiciona las tareas que deseamos utilizar
    def add_task(self, func, args=()):
        assert hasattr(func, '__call__') # must be a function
        # verificamos si la tarea que deseamos anexar no se encuentra en la cola, en caso de no serlo
        if func not in self.result_queues:
            # Definimos un espacio en la cola
            self.result_queues[func] = Queue.Queue()
        # Adicionamos en la cola laa tarea anterior
        self.task_queue.put((func, args, self.result_queues[func]))

    # Función que obtiene los resultados de los hilos
    def get_result(self, func): # returns (result, args)
        # Obtenemos resultado y argumento del resultado de la cola
        result, args = self.result_queues[func].get()
        # Se realiza una excepción
        if isinstance(result, ExceptionInfo):
            print('\n\nWorker thread caught an exception:\n' + result.traceback)
            raise result.value
        # Retornamos el resultado y argumento
        return result, args

    # Función que finaliza las tareas
    def finish(self):
        # Realizamos un recorrido del número de hilos
        for _idx in range(self.num_threads):
            # Metemos en la cola de tareas los valores nulos
            self.task_queue.put((None, (), None))

    # Función que retorna un puntero de datos que hemos definido
    def __enter__(self): # for 'with' statement
        return self

    # Función que finaliza el puntero
    def __exit__(self, *excinfo):
        self.finish()

    # Función que define las acciones que se procesan en tiempo real
    def process_items_concurrently(self, item_iterator, process_func=lambda x: x, pre_func=lambda x: x, post_func=lambda x: x, max_items_in_flight=None):
        # Se verifica que el máximo de elementos sea nulo, cabe mencionar que de esa forma se le puede
        # adicionar la cantidad que se desea
        if max_items_in_flight is None: max_items_in_flight = self.num_threads * 4
        # Se comprueba que los elementos sean por lo menos 1
        assert max_items_in_flight >= 1
        # En caso de serlo se crea un arreglo vacias con los resultados y otro arreglo de un elemento
        results = []
        retire_idx = [0]

        # Función que hace uso de la función lambda que retorna un valor
        def task_func(prepared, _idx):
            return process_func(prepared)

        # Función que obtiene el resultado de la tarea que se desea realizar y su proceso
        def retire_result():
            # Anexamos en una variable sola y una con 2 elementos el resultado obteneido al ingresar
            # lambda al obtener resultado
            processed, (_prepared, idx) = self.get_result(task_func)
            # Anexamos en el arreglo previamente generada results el resultado del proceso que lleva la tarea
            results[idx] = processed
            # Se repite un proceso de manera que se tome en cuenta el primer valor anexado
            # en retire_idx que es cero hasta la longitud del resultado previamente definido y la
            # existencia de que el resultado no sea nullo
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                # En caso de serlo se añade la post función, recordemos que se ha aplicado la pre función
                yield post_func(results[retire_idx[0]])
                # Si cumple se anexa como Ninguno en la posición actual limpiandolo y a su vez avanzando 
                # en uno en el arreglo de un elemento
                results[retire_idx[0]] = None
                retire_idx[0] += 1

        # Recorremos los elementos que queremos utilizar
        for idx, item in enumerate(item_iterator):
            # Preparamos el proceso con su respectiva pre función
            prepared = pre_func(item)
            # A los resultados se le anexa ninguno
            results.append(None)
            # Adicionamos las tareas que deseamos usar, recordando que estas están enlistadas para iterarse
            self.add_task(func=task_func, args=(prepared, idx))
            # Recorremos según el tamaño de los elementos menos el maximo de item previamente definido más 
            # dos omitiendo el primer y último valor
            while retire_idx[0] < idx - max_items_in_flight + 2:
                # Dentro del recorrido se realiza un ciclo para obtener los resultados
                for res in retire_result(): yield res
        # Una vez obteneido los resultados se recorre de nuevo ahora con los valores definidos previamente
        while retire_idx[0] < len(results):
            for res in retire_result(): yield res

#----------------------------------------------------------------------------
# Función display la dirección del tensor en pantalla
def display(tfrecord_dir):
    # Se muestra que se esta cargando la dataset
    print('Loading dataset "%s"' % tfrecord_dir)
    # Comprobamos la existencia de gpu para el proceso, en caso contrario el programa terminara
    tflib.init_tf({'gpu_options.allow_growth': True})
    # Obtenemos las funciones de la dataset de su respectiva clase
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size='full', repeat=False, shuffle_mb=0)
    # Finalizamos el uso de la gpu
    tflib.init_uninitialized_vars()
    import cv2  # pip install opencv-python

    # Declaramos variable en cero y usamos un ciclo infinito
    idx = 0
    while True:
        # Se recorre hasta concluir , solo se rompe en caso de error
        try:
            # Anexamos el batch a todas las imagenes cargadas previamente, obteniendo el resultado y etiquetas
            images, labels = dset.get_minibatch_np(1)
        # Se tenia una excepción si existía un error rompiendo el proceso de obtener las imagenes de manera
        # correcta
        except tf.errors.OutOfRangeError:
            break
        # En caso de cumplirse la obtención correcta de la dataset se compara que el idx no se haya alterado y sea 0
        if idx == 0:
            # Imprimimos un mensaje en pantalla
            print('Displaying images')
            # Con ayuda de openCV nombramos al conjunto de datos de la dataset, esto nos permite
            # utilizarlo después
            cv2.namedWindow('dataset_tool')
            # Imprimimos un mensaje en pantalla
            print('Press SPACE or ENTER to advance, ESC to exit')
        # Imprimimos las etiquetas y el indice respectivo
        print('\nidx = %-8d\nlabel = %s' % (idx, labels[0].tolist()))
        # Imprimimos con ayuda de openCV la dataset previamente nombrada
        cv2.imshow('dataset_tool', images[0].transpose(1, 2, 0)[:, :, ::-1]) # CHW => HWC, RGB => BGR
        # Incrementamos idx para evitar repetir el proceso
        idx += 1
        # Hacemos uso de waitKey para finalizar en caso de necesitarlo
        if cv2.waitKey() == 27:
            break
    # Imprimimos en el display las imagenes 
    print('\nDisplayed %d images.' % idx)

#----------------------------------------------------------------------------
# Función para la extración de la dirección del tensor y su salida
def extract(tfrecord_dir, output_dir):
    # Imprimimos la dirección del tensor
    print('Loading dataset "%s"' % tfrecord_dir)
    # Comprobamos la existencia de gpu para el proceso, en caso contrario el programa terminara
    tflib.init_tf({'gpu_options.allow_growth': True})
    # Obtenemos las funciones de la dataset de su respectiva clase
    dset = dataset.TFRecordDataset(tfrecord_dir, max_label_size=0, repeat=False, shuffle_mb=0)
    # Finalizamos el uso de la gpu
    tflib.init_uninitialized_vars()
    # Observamos la dirección de salida donde se extraeran las imagenes
    print('Extracting images to "%s"' % output_dir)
    # Verificamos que el path exista
    if not os.path.isdir(output_dir):
        # Hacemos uso de la dirección de salida
        os.makedirs(output_dir)
    idx = 0
    # Usamos un ciclo infinito
    while True:
        # Verificamso la existencia de multiplos de 10, ya que recordemos es la base que tenemos en la dataset
        if idx % 10 == 0:
            # En caso de serlo se imprime la cantidad
            print('%d\r' % idx, end='', flush=True)
        try:
            # Anexamos en dos variables la imagen y su etiqueta respectivamente de la dataset cargada
            images, _labels = dset.get_minibatch_np(1)
        # Si se genera un error se rompe el programa
        except tf.errors.OutOfRangeError:
            break
        # Validamos el alto de las imagenes que sean uno
        if images.shape[1] == 1:
            # En caso se serlo se consideran imagenes a blanco y negro
            img = PIL.Image.fromarray(images[0][0], 'L')
        else:
            # En caso contrario hacen referencia a imagenes a color RGB
            img = PIL.Image.fromarray(images[0].transpose(1, 2, 0), 'RGB')
        # Se resguarda las imagenes en la dirección de salida predefinida con el idx e incrementamos para 
        # cambiar de imagen
        img.save(os.path.join(output_dir, 'img%08d.png' % idx))
        idx += 1
    print('Extracted %d images.' % idx)

#----------------------------------------------------------------------------
# Función para comparar los tensores de una dirección a y una dirección b, ignorando o no etiquetas
def compare(tfrecord_dir_a, tfrecord_dir_b, ignore_labels):
    
    max_label_size = 0 if ignore_labels else 'full'
    print('Loading dataset "%s"' % tfrecord_dir_a)
    tflib.init_tf({'gpu_options.allow_growth': True})
    dset_a = dataset.TFRecordDataset(tfrecord_dir_a, max_label_size=max_label_size, repeat=False, shuffle_mb=0)
    print('Loading dataset "%s"' % tfrecord_dir_b)
    dset_b = dataset.TFRecordDataset(tfrecord_dir_b, max_label_size=max_label_size, repeat=False, shuffle_mb=0)
    tflib.init_uninitialized_vars()

    print('Comparing datasets')
    idx = 0
    identical_images = 0
    identical_labels = 0
    while True:
        if idx % 100 == 0:
            print('%d\r' % idx, end='', flush=True)
        try:
            images_a, labels_a = dset_a.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_a, labels_a = None, None
        try:
            images_b, labels_b = dset_b.get_minibatch_np(1)
        except tf.errors.OutOfRangeError:
            images_b, labels_b = None, None
        if images_a is None or images_b is None:
            if images_a is not None or images_b is not None:
                print('Datasets contain different number of images')
            break
        if images_a.shape == images_b.shape and np.all(images_a == images_b):
            identical_images += 1
        else:
            print('Image %d is different' % idx)
        if labels_a.shape == labels_b.shape and np.all(labels_a == labels_b):
            identical_labels += 1
        else:
            print('Label %d is different' % idx)
        idx += 1
    print('Identical images: %d / %d' % (identical_images, idx))
    if not ignore_labels:
        print('Identical labels: %d / %d' % (identical_labels, idx))

#----------------------------------------------------------------------------

def create_mnist(tfrecord_dir, mnist_dir):
    print('Loading MNIST from "%s"' % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    with gzip.open(os.path.join(mnist_dir, 'train-labels-idx1-ubyte.gz'), 'rb') as file:
        labels = np.frombuffer(file.read(), np.uint8, offset=8)
    images = images.reshape(-1, 1, 28, 28)
    images = np.pad(images, [(0,0), (0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 1, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (60000,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_mnistrgb(tfrecord_dir, mnist_dir, num_images=1000000, random_seed=123):
    print('Loading MNIST from "%s"' % mnist_dir)
    import gzip
    with gzip.open(os.path.join(mnist_dir, 'train-images-idx3-ubyte.gz'), 'rb') as file:
        images = np.frombuffer(file.read(), np.uint8, offset=16)
    images = images.reshape(-1, 28, 28)
    images = np.pad(images, [(0,0), (2,2), (2,2)], 'constant', constant_values=0)
    assert images.shape == (60000, 32, 32) and images.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255

    with TFRecordExporter(tfrecord_dir, num_images) as tfr:
        rnd = np.random.RandomState(random_seed)
        for _idx in range(num_images):
            tfr.add_image(images[rnd.randint(images.shape[0], size=3)])

#----------------------------------------------------------------------------

def create_cifar10(tfrecord_dir, cifar10_dir):
    print('Loading CIFAR-10 from "%s"' % cifar10_dir)
    import pickle
    images = []
    labels = []
    for batch in range(1, 6):
        with open(os.path.join(cifar10_dir, 'data_batch_%d' % batch), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        images.append(data['data'].reshape(-1, 3, 32, 32))
        labels.append(data['labels'])
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype == np.int32
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_cifar100(tfrecord_dir, cifar100_dir):
    print('Loading CIFAR-100 from "%s"' % cifar100_dir)
    import pickle
    with open(os.path.join(cifar100_dir, 'train'), 'rb') as file:
        data = pickle.load(file, encoding='latin1')
    images = data['data'].reshape(-1, 3, 32, 32)
    labels = np.array(data['fine_labels'])
    assert images.shape == (50000, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (50000,) and labels.dtype == np.int32
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 99
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_svhn(tfrecord_dir, svhn_dir):
    print('Loading SVHN from "%s"' % svhn_dir)
    import pickle
    images = []
    labels = []
    for batch in range(1, 4):
        with open(os.path.join(svhn_dir, 'train_%d.pkl' % batch), 'rb') as file:
            data = pickle.load(file, encoding='latin1')
        images.append(data[0])
        labels.append(data[1])
    images = np.concatenate(images)
    labels = np.concatenate(labels)
    assert images.shape == (73257, 3, 32, 32) and images.dtype == np.uint8
    assert labels.shape == (73257,) and labels.dtype == np.uint8
    assert np.min(images) == 0 and np.max(images) == 255
    assert np.min(labels) == 0 and np.max(labels) == 9
    onehot = np.zeros((labels.size, np.max(labels) + 1), dtype=np.float32)
    onehot[np.arange(labels.size), labels] = 1.0

    with TFRecordExporter(tfrecord_dir, images.shape[0]) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            tfr.add_image(images[order[idx]])
        tfr.add_labels(onehot[order])

#----------------------------------------------------------------------------

def create_lsun(tfrecord_dir, lmdb_dir, resolution=256, max_images=None):
    print('Loading LSUN dataset from "%s"' % lmdb_dir)
    import lmdb # pip install lmdb # pylint: disable=import-error
    import cv2 # pip install opencv-python
    import io
    with lmdb.open(lmdb_dir, readonly=True).begin(write=False) as txn:
        total_images = txn.stat()['entries'] # pylint: disable=no-value-for-parameter
        if max_images is None:
            max_images = total_images
        with TFRecordExporter(tfrecord_dir, max_images) as tfr:
            for _idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.asarray(PIL.Image.open(io.BytesIO(value)))
                    crop = np.min(img.shape[:2])
                    img = img[(img.shape[0] - crop) // 2 : (img.shape[0] + crop) // 2, (img.shape[1] - crop) // 2 : (img.shape[1] + crop) // 2]
                    img = PIL.Image.fromarray(img, 'RGB')
                    img = img.resize((resolution, resolution), PIL.Image.ANTIALIAS)
                    img = np.asarray(img)
                    img = img.transpose([2, 0, 1]) # HWC => CHW
                    tfr.add_image(img)
                except:
                    print(sys.exc_info()[1])
                if tfr.cur_images == max_images:
                    break

#----------------------------------------------------------------------------

def create_lsun_wide(tfrecord_dir, lmdb_dir, width=512, height=384, max_images=None):
    assert width == 2 ** int(np.round(np.log2(width)))
    assert height <= width
    print('Loading LSUN dataset from "%s"' % lmdb_dir)
    import lmdb # pip install lmdb # pylint: disable=import-error
    import cv2 # pip install opencv-python
    import io
    with lmdb.open(lmdb_dir, readonly=True).begin(write=False) as txn:
        total_images = txn.stat()['entries'] # pylint: disable=no-value-for-parameter
        if max_images is None:
            max_images = total_images
        with TFRecordExporter(tfrecord_dir, max_images, print_progress=False) as tfr:
            for idx, (_key, value) in enumerate(txn.cursor()):
                try:
                    try:
                        img = cv2.imdecode(np.fromstring(value, dtype=np.uint8), 1)
                        if img is None:
                            raise IOError('cv2.imdecode failed')
                        img = img[:, :, ::-1] # BGR => RGB
                    except IOError:
                        img = np.asarray(PIL.Image.open(io.BytesIO(value)))

                    ch = int(np.round(width * img.shape[0] / img.shape[1]))
                    if img.shape[1] < width or ch < height:
                        continue

                    img = img[(img.shape[0] - ch) // 2 : (img.shape[0] + ch) // 2]
                    img = PIL.Image.fromarray(img, 'RGB')
                    img = img.resize((width, height), PIL.Image.ANTIALIAS)
                    img = np.asarray(img)
                    img = img.transpose([2, 0, 1]) # HWC => CHW

                    canvas = np.zeros([3, width, width], dtype=np.uint8)
                    canvas[:, (width - height) // 2 : (width + height) // 2] = img
                    tfr.add_image(canvas)
                    print('\r%d / %d => %d ' % (idx + 1, total_images, tfr.cur_images), end='')

                except:
                    print(sys.exc_info()[1])
                if tfr.cur_images == max_images:
                    break
    print()

#----------------------------------------------------------------------------

def create_celeba(tfrecord_dir, celeba_dir, cx=89, cy=121):
    print('Loading CelebA from "%s"' % celeba_dir)
    glob_pattern = os.path.join(celeba_dir, 'img_align_celeba_png', '*.png')
    image_filenames = sorted(glob.glob(glob_pattern))
    expected_images = 202599
    if len(image_filenames) != expected_images:
        error('Expected to find %d images' % expected_images)

    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order()
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
            assert img.shape == (218, 178, 3)
            img = img[cy - 64 : cy + 64, cx - 64 : cx + 64]
            img = img.transpose(2, 0, 1) # HWC => CHW
            tfr.add_image(img)

#----------------------------------------------------------------------------

def create_from_images(tfrecord_dir, image_dir, shuffle):
    print('Loading images from "%s"' % image_dir)
    image_filenames = sorted(glob.glob(os.path.join(image_dir, '*')))
    if len(image_filenames) == 0:
        error('No input images found')

    img = np.asarray(PIL.Image.open(image_filenames[0]))
    resolution = img.shape[0]
    channels = img.shape[2] if img.ndim == 3 else 1
    if img.shape[1] != resolution:
        error('Input images must have the same width and height')
    if resolution != 2 ** int(np.floor(np.log2(resolution))):
        error('Input image resolution must be a power-of-two')
    if channels not in [1, 3]:
        error('Input images must be stored as RGB or grayscale')

    with TFRecordExporter(tfrecord_dir, len(image_filenames)) as tfr:
        order = tfr.choose_shuffled_order() if shuffle else np.arange(len(image_filenames))
        for idx in range(order.size):
            img = np.asarray(PIL.Image.open(image_filenames[order[idx]]))
            if channels == 1:
                img = img[np.newaxis, :, :] # HW => CHW
            else:
                img = img.transpose([2, 0, 1]) # HWC => CHW
            tfr.add_image(img)

#----------------------------------------------------------------------------

def create_from_hdf5(tfrecord_dir, hdf5_filename, shuffle):
    print('Loading HDF5 archive from "%s"' % hdf5_filename)
    import h5py # conda install h5py
    with h5py.File(hdf5_filename, 'r') as hdf5_file:
        hdf5_data = max([value for key, value in hdf5_file.items() if key.startswith('data')], key=lambda lod: lod.shape[3])
        with TFRecordExporter(tfrecord_dir, hdf5_data.shape[0]) as tfr:
            order = tfr.choose_shuffled_order() if shuffle else np.arange(hdf5_data.shape[0])
            for idx in range(order.size):
                tfr.add_image(hdf5_data[order[idx]])
            npy_filename = os.path.splitext(hdf5_filename)[0] + '-labels.npy'
            if os.path.isfile(npy_filename):
                tfr.add_labels(np.load(npy_filename)[order])

#----------------------------------------------------------------------------

def execute_cmdline(argv):
    prog = argv[0]
    parser = argparse.ArgumentParser(
        prog        = prog,
        description = 'Tool for creating multi-resolution TFRecords datasets for StyleGAN and ProGAN.',
        epilog      = 'Type "%s <command> -h" for more information.' % prog)

    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    def add_command(cmd, desc, example=None):
        epilog = 'Example: %s %s' % (prog, example) if example is not None else None
        return subparsers.add_parser(cmd, description=desc, help=desc, epilog=epilog)

    p = add_command(    'display',          'Display images in dataset.',
                                            'display datasets/mnist')
    p.add_argument(     'tfrecord_dir',     help='Directory containing dataset')

    p = add_command(    'extract',          'Extract images from dataset.',
                                            'extract datasets/mnist mnist-images')
    p.add_argument(     'tfrecord_dir',     help='Directory containing dataset')
    p.add_argument(     'output_dir',       help='Directory to extract the images into')

    p = add_command(    'compare',          'Compare two datasets.',
                                            'compare datasets/mydataset datasets/mnist')
    p.add_argument(     'tfrecord_dir_a',   help='Directory containing first dataset')
    p.add_argument(     'tfrecord_dir_b',   help='Directory containing second dataset')
    p.add_argument(     '--ignore_labels',  help='Ignore labels (default: 0)', type=int, default=0)

    p = add_command(    'create_mnist',     'Create dataset for MNIST.',
                                            'create_mnist datasets/mnist ~/downloads/mnist')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'mnist_dir',        help='Directory containing MNIST')

    p = add_command(    'create_mnistrgb',  'Create dataset for MNIST-RGB.',
                                            'create_mnistrgb datasets/mnistrgb ~/downloads/mnist')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'mnist_dir',        help='Directory containing MNIST')
    p.add_argument(     '--num_images',     help='Number of composite images to create (default: 1000000)', type=int, default=1000000)
    p.add_argument(     '--random_seed',    help='Random seed (default: 123)', type=int, default=123)

    p = add_command(    'create_cifar10',   'Create dataset for CIFAR-10.',
                                            'create_cifar10 datasets/cifar10 ~/downloads/cifar10')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'cifar10_dir',      help='Directory containing CIFAR-10')

    p = add_command(    'create_cifar100',  'Create dataset for CIFAR-100.',
                                            'create_cifar100 datasets/cifar100 ~/downloads/cifar100')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'cifar100_dir',     help='Directory containing CIFAR-100')

    p = add_command(    'create_svhn',      'Create dataset for SVHN.',
                                            'create_svhn datasets/svhn ~/downloads/svhn')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'svhn_dir',         help='Directory containing SVHN')

    p = add_command(    'create_lsun',      'Create dataset for single LSUN category.',
                                            'create_lsun datasets/lsun-car-100k ~/downloads/lsun/car_lmdb --resolution 256 --max_images 100000')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'lmdb_dir',         help='Directory containing LMDB database')
    p.add_argument(     '--resolution',     help='Output resolution (default: 256)', type=int, default=256)
    p.add_argument(     '--max_images',     help='Maximum number of images (default: none)', type=int, default=None)

    p = add_command(    'create_lsun_wide', 'Create LSUN dataset with non-square aspect ratio.',
                                            'create_lsun_wide datasets/lsun-car-512x384 ~/downloads/lsun/car_lmdb --width 512 --height 384')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'lmdb_dir',         help='Directory containing LMDB database')
    p.add_argument(     '--width',          help='Output width (default: 512)', type=int, default=512)
    p.add_argument(     '--height',         help='Output height (default: 384)', type=int, default=384)
    p.add_argument(     '--max_images',     help='Maximum number of images (default: none)', type=int, default=None)

    p = add_command(    'create_celeba',    'Create dataset for CelebA.',
                                            'create_celeba datasets/celeba ~/downloads/celeba')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'celeba_dir',       help='Directory containing CelebA')
    p.add_argument(     '--cx',             help='Center X coordinate (default: 89)', type=int, default=89)
    p.add_argument(     '--cy',             help='Center Y coordinate (default: 121)', type=int, default=121)

    p = add_command(    'create_from_images', 'Create dataset from a directory full of images.',
                                            'create_from_images datasets/mydataset myimagedir')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'image_dir',        help='Directory containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    p = add_command(    'create_from_hdf5', 'Create dataset from legacy HDF5 archive.',
                                            'create_from_hdf5 datasets/celebahq ~/downloads/celeba-hq-1024x1024.h5')
    p.add_argument(     'tfrecord_dir',     help='New dataset directory to be created')
    p.add_argument(     'hdf5_filename',    help='HDF5 archive containing the images')
    p.add_argument(     '--shuffle',        help='Randomize image order (default: 1)', type=int, default=1)

    args = parser.parse_args(argv[1:] if len(argv) > 1 else ['-h'])
    func = globals()[args.command]
    del args.command
    func(**vars(args))

#----------------------------------------------------------------------------

if __name__ == "__main__":
    execute_cmdline(sys.argv)

#----------------------------------------------------------------------------
