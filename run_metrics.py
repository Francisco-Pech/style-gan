# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Main entry point for training StyleGAN and ProGAN networks."""

import dnnlib
from dnnlib import EasyDict
import dnnlib.tflib as tflib

import config
from metrics import metric_base
from training import misc

#----------------------------------------------------------------------------
# Función para correr el metric pickle, tiene varios párametros
def run_pickle(submit_config, metric_args, network_pkl, dataset_args, mirror_augment):
    # Corremos la configuración submit
    ctx = dnnlib.RunContext(submit_config)
    # Inicializamos configuración del tensor
    tflib.init_tf()
    # Imprimimos el nombre de la metrica y la red network
    print('Evaluating %s metric on network_pkl "%s"...' % (metric_args.name, network_pkl))
    # Llamamos a la metrica que deseemos utilizar mediante el argumento
    metric = dnnlib.util.call_func_by_name(**metric_args)
    # Imprimimos espacio
    print()
    # Corremos metricas con base a la red definido pkl, los argumentos de dataset y mirror, junto
    # el número de gpus necesarios para configurar 
    metric.run(network_pkl, dataset_args=dataset_args, mirror_augment=mirror_augment, num_gpus=submit_config.num_gpus)
    # Imprimir espacio
    print()
    # Cerramos la configuración de submit
    ctx.close()

#----------------------------------------------------------------------------
# Función para correr las métricas instantenamentes, se pasan varios párametros
def run_snapshot(submit_config, metric_args, run_id, snapshot):
    # Corremos la configuración submit
    ctx = dnnlib.RunContext(submit_config)
    # Inicializamos configuración del tensor
    tflib.init_tf()
    # Imprimimos el nombre de la metrica, su id en la cual corre y el párametro instantáneo 
    print('Evaluating %s metric on run_id %s, snapshot %s...' % (metric_args.name, run_id, snapshot))
    # Resguardamos la localización del id de la métrica que corre
    run_dir = misc.locate_run_dir(run_id)
    # Resguardamos la dirección donde se encuentra la métrica a usar, más su párametro instantáneo,
    # todo esto en una red pkl
    network_pkl = misc.locate_network_pkl(run_dir, snapshot)
    # Llamamos por su nombre del argunto las métricas a utilizar
    metric = dnnlib.util.call_func_by_name(**metric_args)
    # Imprimimos espacio
    print()
    # Corremos la red definida con el id de la métrica previamente, su dirección y la configuración de gpus
    metric.run(network_pkl, run_dir=run_dir, num_gpus=submit_config.num_gpus)
    # Imprimimos espacio
    print()
    # Cerramos configuración del tensor
    ctx.close()

#----------------------------------------------------------------------------
# Función que corre todas las métricas instantáneas, se pasan varios párametros
def run_all_snapshots(submit_config, metric_args, run_id):
    # Corremos la configuración submit
    ctx = dnnlib.RunContext(submit_config)
    # Inicializamos configuración del tensor
    tflib.init_tf()
    # Imprimimos el nombre de la metrica y su id en la cual corre 
    print('Evaluating %s metric on all snapshots of run_id %s...' % (metric_args.name, run_id))
    # Resguardamos la localización del id de la métrica que corre
    run_dir = misc.locate_run_dir(run_id)
    # Resguardamos la dirección donde se encuentra la métrica a usar, más su párametro instantáneo,
    # todo esto en una red pkl
    network_pkls = misc.list_network_pkls(run_dir)
    # Llamamos por su nombre del argunto las métricas a utilizar
    metric = dnnlib.util.call_func_by_name(**metric_args)
    # Imprimimos espacio en blanco
    print()
    # Recorremos la red pkl, se obtienen 2 párametros el indice y el valor de la red pkl
    for idx, network_pkl in enumerate(network_pkls):
        # Actualizamos la longitud de las redes pkls con base a su indice
        ctx.update('', idx, len(network_pkls))
        # Se corre la métrica de la red pkl, junto a su dirección y la cantidad de gpus
        metric.run(network_pkl, run_dir=run_dir, num_gpus=submit_config.num_gpus)
    # Imprimos espacio
    print()
    # Cerramos el tensor
    ctx.close()

#----------------------------------------------------------------------------
# Función principal
def main():
    # Hacemos uso de la configuración 
    submit_config = dnnlib.SubmitConfig()

    # Which metrics to evaluate?
    # Creamos un arreglo de la métrica vacio
    metrics = []
    # Hacemos uso de la métrica base y la sumamos al arreglo vacio, esto puede variar según la métrica a
    # utilizar
    metrics += [metric_base.fid50k]
    #metrics += [metric_base.ppl_zfull]
    #metrics += [metric_base.ppl_wfull]
    #metrics += [metric_base.ppl_zend]
    #metrics += [metric_base.ppl_wend]
    #metrics += [metric_base.ls]
    #metrics += [metric_base.dummy]

    # Which networks to evaluate them on?
    # Creamos un arreglo de tareas vacios
    tasks = []
    # Anexamos al arreglo vacio los argumentos necesarios para hacer uso de la red network del drive
    tasks += [EasyDict(run_func_name='run_metrics.run_pickle', network_pkl='https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ', dataset_args=EasyDict(tfrecord_dir='ffhq', shuffle_mb=0), mirror_augment=True)] # karras2019stylegan-ffhq-1024x1024.pkl
    #tasks += [EasyDict(run_func_name='run_metrics.run_snapshot', run_id=100, snapshot=25000)]
    #tasks += [EasyDict(run_func_name='run_metrics.run_all_snapshots', run_id=100)]

    # How many GPUs to use?
    # Determinamos cantidad de gpus a utilizar esto puede variar
    submit_config.num_gpus = 1
    #submit_config.num_gpus = 2
    #submit_config.num_gpus = 4
    #submit_config.num_gpus = 8

    # Execute.
    # Definimos la ruta raíz de donde se resguarda el resultado de la métrica
    submit_config.run_dir_root = dnnlib.submission.submit.get_template_from_path(config.result_dir)
    # Se ignoran algunas configuraciones que contienen datos exceptos 
    submit_config.run_dir_ignore += config.run_dir_ignore
    # Recorremos las tareas
    for task in tasks:
        # Recorremos las métricas
        for metric in metrics:
            # Resguardamos en una variable que contienen el nombre de la tarea a correr y métricas
            submit_config.run_desc = '%s-%s' % (task.run_func_name, metric.name)
            # Llamamos a la tarea por su nombre y vemos que existe
            if task.run_func_name.endswith('run_snapshot'):
                # Resguardamos en la misma variable de la tarea y métricas el id de la tarea y de la 
                # tarea instantánea
                submit_config.run_desc += '-%s-%s' % (task.run_id, task.snapshot)
            # Llamamos a todas las tareas y veremos si existe
            if task.run_func_name.endswith('run_all_snapshots'):
                # En caso de serlo se resguarda la tarea y su id
                submit_config.run_desc += '-%s' % task.run_id
            # configuramos la cantidad de gpus
            submit_config.run_desc += '-%dgpu' % submit_config.num_gpus
            # Corremos el submit de la configuración de las métricas y tareas
            dnnlib.submit_run(submit_config, metric_args=metric, **task)

#----------------------------------------------------------------------------
# Corremos la función principal
if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
