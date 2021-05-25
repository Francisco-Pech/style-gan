# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Minimal script for reproducing the figures of the StyleGAN paper using pre-trained generators."""

import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config

#----------------------------------------------------------------------------
# Helpers for loading and using pre-trained generators.
# Hacemos uso de varias redes para preentrenar, en este caso se manejan 5 de distintas dimensionalidades, estás imagenes
# están definidas 
url_ffhq        = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ' # karras2019stylegan-ffhq-1024x1024.pkl
url_celebahq    = 'https://drive.google.com/uc?id=1MGqJl28pN4t7SAtSrPdSRJSQJqahkzUf' # karras2019stylegan-celebahq-1024x1024.pkl
url_bedrooms    = 'https://drive.google.com/uc?id=1MOSKeGF0FJcivpBI7s63V9YHloUTORiF' # karras2019stylegan-bedrooms-256x256.pkl
url_cars        = 'https://drive.google.com/uc?id=1MJ6iCfNtMIRicihwRorsM3b7mmtmK9c3' # karras2019stylegan-cars-512x384.pkl
url_cats        = 'https://drive.google.com/uc?id=1MQywl0FNt6lHu8E_EUqnRbviagS7fbiJ' # karras2019stylegan-cats-256x256.pkl

#-----------------------------------------------------------------------------------------------------------
# Preentrenamiento de objetos
url_objects     = 'https://drive.google.com/uc?id=1mYx2GFc58B2aW6Qz-B7WaRlV0gMGdXjF'

#-----------------------------------------------------------------------------------------------------------

# Hacemos uso de diccionarios, esto quiere decir que ordenamos valores según deseemos, en este caso
# ordenamos primero una variable que contiene un dicionario con 2 variables y de segunda variable el bash
synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)

# Llamamos a un diccionario vacio 
_Gs_cache = dict()

# Función que carga los datos pre-entrenados, tiene como parámetro su url
def load_Gs(url):
    # Verificamos que el url que contiene la dataset pre-entrenada no se encuentre en algun diccionario
    if url not in _Gs_cache:
        # En el caso de no encontrarse en algun diccionario, abrimos la url 
        with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
            # Obtenemos la información de la url 
            _G, _D, Gs = pickle.load(f)
        # Cargamos en el diccionario vacio la url para determinar cual url era y su respectivo valor
        _Gs_cache[url] = Gs
    # Retornamos los datos obtenidos
    return _Gs_cache[url]

#----------------------------------------------------------------------------
# Figures 2, 3, 10, 11, 12: Multi-resolution grid of uncurated result images.
# Función para dibujar sin cortes una imagen, se le asigna varios párametros 
def draw_uncurated_result_figure(png, Gs, cx, cy, cw, ch, rows, lods, seed):
    # Determinamos un arreglo para visualizar la existencia de imagen
    print(png)
    # Respaldamos en una variable números random, definidos en un rango, este rando se determina
    # entre filas y un arreglo cargado, además de hacer uso de una imagen de las imagenes cargadas o 
    # dataset precargadas
    latents = np.random.RandomState(seed).randn(sum(rows * 2**lod for lod in lods), Gs.input_shape[1])
    # Creamos una variable que tiene las imagenes, se hacer uso de la dataset cargada y bien el diccionario que 
    # previamente se cargo, cabe mencionar que se hace uso de los números aleatorios para filtrado
    images = Gs.run(latents, None, **synthesis_kwargs) # [seed, y, x, rgb]

    # Se genera una nueva imagene RGB según los datos previamente ingresados
    canvas = PIL.Image.new('RGB', (sum(cw // 2**lod for lod in lods), ch * rows), 'white')
    # Listamos las imagenes con el diccionario
    image_iter = iter(list(images))
    # Cargos el arreglo que deseamos en este caso [0,1,2,2,3,3] y obntenemos 2 valores la posición
    # y el valor
    for col, lod in enumerate(lods):
        # Realizamos un ciclo for que recorra los datos del lods, estos son los valores,
        # esto quiere decir que cada vez que recorremos los datos se hará un multiplo
        # de la fila en este caso 3 por la cantidad definida en el lods
        for row in range(rows * 2**lod):
            # Recorre un arreglo del listado de un diccinario en RGB
            image = PIL.Image.fromarray(next(image_iter), 'RGB')
            # Determina una porción de imagenes
            image = image.crop((cx, cy, cx + cw, cy + ch))
            # Redimensionamos la imagen
            image = image.resize((cw // 2**lod, ch // 2**lod), PIL.Image.ANTIALIAS)
            # Usamos la imagen previamentte creada, con una imagen creada por el arreglo  y pegamos
            canvas.paste(image, (sum(cw // 2**lod for lod in lods[:col]), row * ch // 2**lod))
    # Guardamos la imagen 
    canvas.save(png)

#----------------------------------------------------------------------------
# Figure 3: Style mixing.
# Función que dibuja imagenes mezcladas con diversos párametros
def draw_style_mixing_figure(png, Gs, w, h, src_seeds, dst_seeds, style_ranges):
    # Visualizamos una imagen
    print(png)
    # Respaldamos en una variable números random en una pila, definidos en un rango, este rando se determina
    # entre filas y un arreglo cargado, además de hacer uso de una imagen de las imagenes cargadas o 
    # dataset precargadas
    src_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in src_seeds)
    # Realizamos el mismo proceso anterior pero con el dst
    dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
    # Mapeamos los datos obtenidos en una red pre-entrenada y respaldamos el src aleatorio
    src_dlatents = Gs.components.mapping.run(src_latents, None) # [seed, layer, component]
    # Realizamos el mismo proceso con el dst
    dst_dlatents = Gs.components.mapping.run(dst_latents, None) # [seed, layer, component]
    # Se hace uso de una sintasis definido y correlos en la src y dst
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)
    # Se genera una nueva imagene RGB según los datos previamente ingresados
    canvas = PIL.Image.new('RGB', (w * (len(src_seeds) + 1), h * (len(dst_seeds) + 1)), 'white')
    # Cargos el arreglo que deseamos en este caso el obtenido en src y obtenemos 2 valores la posición
    # y el valor
    for col, src_image in enumerate(list(src_images)):
        # Pegamos el valor de la nueva imagen generada mediante el valor de src_image para las columnas
        canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), ((col + 1) * w, 0))
    # Cargamos el arreglo deseado en este caso dst y obtenemos 2 valores posición y el valor, cabe
    # mencionar que se tratan de listados
    for row, dst_image in enumerate(list(dst_images)):
        # Se realiza el mismop procedimiento que con src_image para columnas, pero en este caso para filas
        canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), (0, (row + 1) * h))
        # Anexamos a la pila el resultado mapeado recorriendolo y multiplicandolo por su logitud
        row_dlatents = np.stack([dst_dlatents[row]] * len(src_seeds))
        # Ahora hacemos uso de columnas y filas y respaldarlas en row_dlatents de todas las filas
        row_dlatents[:, style_ranges[row]] = src_dlatents[:, style_ranges[row]]
        # Renderizamos de nueva forma como sintasis la dataset
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        # Recorremos el resultado obtenido
        for col, image in enumerate(list(row_images)):
            # Copiamos la imagen en cavas y guardamos
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, (row + 1) * h))
    canvas.save(png)

#----------------------------------------------------------------------------
# Figure 4: Noise detail.
# Función que dibuja una imagen o la plasma con el ruido obtenido, no se le da ningun filtro
def draw_noise_detail_figure(png, Gs, w, h, num_samples, seeds):
    # Imprimimos arreglo de la imagen seleccionada
    print(png)
    # Generamos una nueva variable que contiene una nueva imagen generada RGB
    canvas = PIL.Image.new('RGB', (w * 3, h * len(seeds)), 'white')
    # Recorremos las semillas que deseamos o necesitamos obteniendo 2 párametros la posición y el resultado
    for row, seed in enumerate(seeds):
        # Obtenemos el valor actual aleatoriamente teniendo en cuenta la carga de la dataset
        latents = np.stack([np.random.RandomState(seed).randn(Gs.input_shape[1])] * num_samples)
        # Asignamos a una variable la carga corriente del valor aleatoria actual en un diccionario
        images = Gs.run(latents, None, truncation_psi=1, **synthesis_kwargs)
        # Hacemos uso de la imagen generada y le anexamos el valor de una imagen con sus dimensiones
        canvas.paste(PIL.Image.fromarray(images[0], 'RGB'), (0, row * h))
        # Recorremos un rango definido 
        for i in range(4):
            # Se realiza crea en un arreglo las imagenes esto para cuadrarlos 
            crop = PIL.Image.fromarray(images[i + 1], 'RGB')
            # Se asigna los valores para cuadrarlos
            crop = crop.crop((650, 180, 906, 436))
            # Se redimensiona las imagenes
            crop = crop.resize((w//2, h//2), PIL.Image.NEAREST)
            # Se resguardan las imagenes cabe mencionar que esto se hace solamente con los
            # primeros 4 valores, esto debido a que es más factible ver como si las imagenes son correctas
            canvas.paste(crop, (w + (i%2) * w//2, row * h + (i//2) * h//2))
        # Se realiza la media de std
        diff = np.std(np.mean(images, axis=3), axis=0) * 4
        # Se realiza la diferencia con base a la media y una formula
        diff = np.clip(diff + 0.5, 0, 255).astype(np.uint8)
        # Se copia el arreglo en canvas
        canvas.paste(PIL.Image.fromarray(diff, 'L'), (w * 2, row * h))
    # Se guarda el resultado final en canvas con base a la imagen seleccionada
    canvas.save(png)

#----------------------------------------------------------------------------
# Figure 5: Noise components.
# Función para dibujar los componentes que contienen ruido, se le pasa diversos párametros
def draw_noise_components_figure(png, Gs, w, h, seeds, noise_ranges, flips):
    # Imprimimos el arreglo de la imagen seleccionada
    print(png)
    # Realizamos una copia de la dataset cargada
    Gsc = Gs.clone()
    # Obtenemos un arreglo del ruido de componentes de la sitasis de la clonación y nombre
    noise_vars = [var for name, var in Gsc.components.synthesis.vars.items() if name.startswith('noise')]
    # Listamos la compresión del ruido previamente localizado en el arreglo
    noise_pairs = list(zip(noise_vars, tflib.run(noise_vars))) # [(var, val), ...]
    # Generamos un arreglo aleatoriamente que sea la más actual
    latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds)
    # Creamos un arreglo donde se pondrán todas las imagenes
    all_images = []
    # Recorremos los rangos predefinidos en un arreglo
    for noise_range in noise_ranges:
        # Envíamos las variables del componente con ruido obtenido
        tflib.set_vars({var: val * (1 if i in noise_range else 0) for i, (var, val) in enumerate(noise_pairs)})
        # Definimos un rango para las imagenes clonadas y el diccionario
        range_images = Gsc.run(latents, None, truncation_psi=1, randomize_noise=False, **synthesis_kwargs)
        # Definimos que las imagenes obtenidas se localicen algunas filas, todas la columnas y todos los
        # rango definidos
        range_images[flips, :, :] = range_images[flips, :, ::-1]
        # Adicionamos en el arreglo que creamos antes el resultado enlistado
        all_images.append(list(range_images))

    # Generamos una imagen nueva
    canvas = PIL.Image.new('RGB', (w * 2, h * 2), 'white')
    # Recorremos el resultado del arreglo previamente calculado obteniendo 2 párametros
    for col, col_images in enumerate(zip(*all_images)):
        # Hacemos uso de los rangos definos en las imagenes a generar y las cuadramos, además se copian
        canvas.paste(PIL.Image.fromarray(col_images[0], 'RGB').crop((0, 0, w//2, h)), (col * w, 0))
        canvas.paste(PIL.Image.fromarray(col_images[1], 'RGB').crop((w//2, 0, w, h)), (col * w + w//2, 0))
        canvas.paste(PIL.Image.fromarray(col_images[2], 'RGB').crop((0, 0, w//2, h)), (col * w, h))
        canvas.paste(PIL.Image.fromarray(col_images[3], 'RGB').crop((w//2, 0, w, h)), (col * w + w//2, h))
    # Resguardamos las imagenes obtenidas o imagen
    canvas.save(png)

#----------------------------------------------------------------------------
# Figure 8: Truncation trick.
# Función para dibujar imagen truncada
def draw_truncation_trick_figure(png, Gs, w, h, seeds, psis):
    # Imprimimos el arreglo de la imagen seleccionada
    print(png)
    # Generamos aleatoriamente un arreglo y lo ponemos en la pila, esta es la actual
    latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in seeds)
    # Mapeamos el resultado actual y corremos
    dlatents = Gs.components.mapping.run(latents, None) # [seed, layer, component]
    # Obtenemos resultado de algun componente del dataset
    dlatent_avg = Gs.get_var('dlatent_avg') # [component]

    # Generamos una nueva imagen para modificar
    canvas = PIL.Image.new('RGB', (w * len(psis), h * len(seeds)), 'white')
    # Recorremos el listado de los valores actuales mapeados mediante los 2 párametros definidos
    # uno recorre el indice conocido como fila y el otro el resultado
    for row, dlatent in enumerate(list(dlatents)):
        # Hacemos uso de una fila de un valor actual mapeado y redimensionado, además de filtrado
        row_dlatents = (dlatent[np.newaxis] - dlatent_avg) * np.reshape(psis, [-1, 1, 1]) + dlatent_avg
        # Usamos la dataset previamente cargada y la corremos con una sintasis, cabe mencionar que 
        # hacemos uso de la fila de valores actual mapeados
        row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=False, **synthesis_kwargs)
        # Hacemos uso de las filas de las imagenes y obtenemos sus valores con sus 2 párametros,
        # estos son el indice en la columna y el resultado en image
        for col, image in enumerate(list(row_images)):
            # Resguardamos un arreglo de las imagenes definidos por una columna y fila
            canvas.paste(PIL.Image.fromarray(image, 'RGB'), (col * w, row * h))
    # Guardamos el resultado en la imagen seleccionada 
    canvas.save(png)

#----------------------------------------------------------------------------
# Main program.
# Función que es la principal del programa
def main():
    # Inicializamos el tensor
    tflib.init_tf()
    # Definimos ruta de salida en este caso esta en result
    os.makedirs(config.result_dir, exist_ok=True)
    # Se hace uso de las funciones previamente creadas, se menciona que se usa las imagenes de alta calidad 
    # de caras, sin embargo, se usan para otros casos como habitaciones, carros, gatos otros tipos de
    # dibujos
    draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure02-uncurated-ffhq.png'), load_Gs(url_ffhq), cx=0, cy=0, cw=1024, ch=1024, rows=3, lods=[0,1,2,2,3,3], seed=5)
    draw_style_mixing_figure(os.path.join(config.result_dir, 'figure03-style-mixing.png'), load_Gs(url_ffhq), w=1024, h=1024, src_seeds=[639,701,687,615,2268], dst_seeds=[888,829,1898,1733,1614,845], style_ranges=[range(0,4)]*3+[range(4,8)]*2+[range(8,18)])
    draw_noise_detail_figure(os.path.join(config.result_dir, 'figure04-noise-detail.png'), load_Gs(url_ffhq), w=1024, h=1024, num_samples=100, seeds=[1157,1012])
    draw_noise_components_figure(os.path.join(config.result_dir, 'figure05-noise-components.png'), load_Gs(url_ffhq), w=1024, h=1024, seeds=[1967,1555], noise_ranges=[range(0, 18), range(0, 0), range(8, 18), range(0, 8)], flips=[1])
    draw_truncation_trick_figure(os.path.join(config.result_dir, 'figure08-truncation-trick.png'), load_Gs(url_ffhq), w=1024, h=1024, seeds=[91,388], psis=[1, 0.7, 0.5, 0, -0.5, -1])
    draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure10-uncurated-bedrooms.png'), load_Gs(url_bedrooms), cx=0, cy=0, cw=256, ch=256, rows=5, lods=[0,0,1,1,2,2,2], seed=0)
    draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure11-uncurated-cars.png'), load_Gs(url_cars), cx=0, cy=64, cw=512, ch=384, rows=4, lods=[0,1,2,2,3,3], seed=2)
    draw_uncurated_result_figure(os.path.join(config.result_dir, 'figure12-uncurated-cats.png'), load_Gs(url_cats), cx=0, cy=0, cw=256, ch=256, rows=5, lods=[0,0,1,1,2,2,2], seed=1)

#----------------------------------------------------------------------------
# Se llama la función principal para ejecutar
if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
